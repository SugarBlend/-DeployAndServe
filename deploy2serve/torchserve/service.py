import aiohttp
import asyncio
from argparse import Namespace, ArgumentParser
import cv2
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import FileResponse, Response
import os
from starlette.background import BackgroundTask
from starlette.exceptions import HTTPException
import pickle
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest, make_asgi_app, Summary, Counter
import numpy as np
from queue import Queue, Empty
from typing import List
from threading import Thread
import time
import tempfile
import uvicorn

from deploy2serve.triton.base_service import collect_frames
from deploy2serve.utils.logger import get_logger


class Service(object):
    def __init__(self, fastapi_url: str, inference_server_url: str) -> None:
        self.fastapi_url: str = fastapi_url
        self.inference_server_url: str = inference_server_url
        self.logger = get_logger("torchserve")

    async def send_requests(
            self,
            session: aiohttp.ClientSession,
            model_name: str,
            frame_queue: Queue[np.ndarray],
            thread_collector: Thread
    ) -> List[List[np.ndarray]]:
        tasks: List[asyncio.Task] = []
        results: List[asyncio.Future] = []
        max_concurrent_tasks: int = 128
        counter: int = 0

        while not frame_queue.empty() or thread_collector.is_alive():
            if len(tasks) >= max_concurrent_tasks:
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                tasks = list(pending)
                results.extend(done)

            try:
                frame = frame_queue.get_nowait()
                _, img_encoded = cv2.imencode(".jpg", frame)
                task = asyncio.create_task(
                    session.post(
                        f"http://{self.inference_server_url}/predictions/{model_name}",
                        data=img_encoded.tobytes()
                    ),
                    name=f"Task:{counter}"
                )
                tasks.append(task)
                counter += 1
            except Empty:
                await asyncio.sleep(0.01)

        if tasks:
            done, _ = await asyncio.wait(tasks)
            results.extend(done)

        tasks.clear()
        sorted_results = sorted(results, key=lambda x: int(x.get_name().split(":")[-1]))
        return [await task.result().json(content_type=None) for task in sorted_results]

    async def request_manager(self, content_type: str, data: bytes, model_name: str):
        try:
            async with aiohttp.ClientSession() as session:
                media_type, container = content_type.split("/")
                if media_type == "video":
                    frame_queue = Queue(maxsize=256)
                    collector_thread = Thread(target=collect_frames, args=(data, container, frame_queue))
                    collector_thread.start()
                    return await self.send_requests(session, model_name, frame_queue, collector_thread)
                elif media_type == "image":
                    response = await session.post(f"http://{self.inference_server_url}/predictions/{model_name}", data=data)
                    return await response.json(content_type=None)
                else:
                    raise NotImplementedError("Passed unsupported data type of file.")
        except aiohttp.ClientError as error:
            self.logger.critical(f"Request failed: {error}.")
        except Exception as error:
            self.logger.critical(f"Unexpected error: {error}.")

    def create(self) -> uvicorn.Server:
        app = FastAPI()

        metrics_app = make_asgi_app()
        app.mount("/metrics", metrics_app)

        REQUEST_TIME = Summary(
            "predict_request_processing_seconds",
            "Time spent processing predict request"
        )
        REQUEST_COUNT = Counter(
            "predict_request_count_total",
            "Total predict requests count"
        )

        @app.get("/metrics")
        async def metrics() -> Response:
            return Response(
                content=generate_latest(),
                media_type=CONTENT_TYPE_LATEST
            )

        @app.post("/predict")
        async def predict(
                file: UploadFile = File(description="Uploaded file."),
                model_name: str = Query(description="Model name.")
        ) -> FileResponse:
            REQUEST_COUNT.inc()
            start_time = time.time()

            try:
                contents = await file.read()
                result = await self.request_manager(file.content_type, contents, model_name)
                REQUEST_TIME.observe(time.time() - start_time)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pickle") as temp:
                    pickle.dump(result, temp)

                return FileResponse(temp.name, filename="data.pickle", media_type="application/octet-stream",
                                    background=BackgroundTask(lambda: os.unlink(temp.name)))

            except Exception as error:
                raise HTTPException(status_code=500, detail=str(error))

        host, port = self.fastapi_url.split(":")
        config = uvicorn.Config(app, host=host, port=int(port), log_level="error")
        server = uvicorn.Server(config)
        return server


def parse_options() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--service_host", type=str, default="localhost", help="Service host.")
    parser.add_argument("--service_port", type=int, default=5001, help="Service port.")
    parser.add_argument("--torchserve_host", type=str, default="localhost", help="TorchServe host.")
    parser.add_argument("--torchserve_port", type=int, default=8080, help="TorchServe port.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_options()
    api = Service(
        fastapi_url=f"{args.service_host}:{args.service_port}",
        inference_server_url=f"{args.torchserve_host}:{args.torchserve_port}"
    )
    server = api.create()
    server.run()
