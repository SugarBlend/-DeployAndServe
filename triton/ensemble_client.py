import asyncio
from argparse import ArgumentParser, Namespace
import base64
import cv2
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import FileResponse
import numpy as np
import pickle
from queue import Queue, Empty
from importlib import import_module
from imutils.video import FileVideoStream
import tempfile
from tqdm import tqdm
from threading import Thread
from typing import Union, Callable, Optional, List, Coroutine, Type, Any
import time
import uvicorn

from triton.configs import ServiceConfig, Url, Formats, ProtocolType
from triton.base import TritonRemote
from utils.logger import get_logger
from utils.runner import ServerRunner


class Service(object):
    def __init__(
            self,
            inference_server_cls: Union[TritonRemote, Callable],
            fastapi: Url,
            triton: Url,
            protocol: ProtocolType,
            show: bool = False
    ) -> None:
        self.inference_server_cls: Union[TritonRemote, Callable] = inference_server_cls
        self.fastapi: Url = fastapi
        self.triton: Url = triton
        self.protocol: ProtocolType = protocol
        self.show: bool = show

        self.inference_server: Optional[TritonRemote] = None
        self.runner: ServerRunner = self.create()
        self.logger = get_logger("service")

    @staticmethod
    def collect_frames(data: bytes, container: str, frame_queue: Queue) -> None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{container}") as tmp:
            tmp.write(data)
            source_path = tmp.name

        cap = FileVideoStream(source_path)
        cap.start()
        frames_number = int(cap.stream.get(cv2.CAP_PROP_FRAME_COUNT))

        with tqdm(total=frames_number, desc="Read frames") as bar:
            while True:
                frame = cap.read()
                if frame is None:
                    break
                frame_queue.put(frame)
                bar.update()
        cap.stop()
        tmp.close()

    async def send_requests_concurrently(
            self,
            frame_queue: Queue[np.ndarray],
            thread_collector: Thread
    ) -> List[List[np.ndarray]]:
        tasks: List[Coroutine[Any, Any, List[np.ndarray]]] = []
        results: List[List[np.ndarray]] = []
        while not frame_queue.empty() or thread_collector.is_alive():
            try:
                frame = frame_queue.get()
            except Empty:
                continue
            _, img_encoded = cv2.imencode(".jpg", frame)
            tasks.append(self.inference_server.infer(feed_input=[np.stack([img_encoded])]))

        if tasks:
            results = await asyncio.gather(*tasks)
        return results

    async def request_manager(self, content_type: str, data: bytes) -> List[List[np.ndarray]]:
        media_type, container = content_type.split("/")
        if media_type == Formats.VIDEO:
            frame_queue = Queue()
            collector_thread = Thread(target=self.collect_frames, args=(data, container, frame_queue))
            collector_thread.start()
            results = await self.send_requests_concurrently(frame_queue, collector_thread)
            collector_thread.join()
            return results
        elif media_type == Formats.IMAGE:
            buffer = np.frombuffer(data, np.uint8)
            result = await self.inference_server.infer(feed_input=[np.stack([buffer])])
            return [result]
        else:
            raise NotImplementedError("Passed unsupported data type of file.")

    def create(self) -> ServerRunner:
        app = FastAPI()

        @app.post("/predict")
        async def predict(
                file: UploadFile = File(description="Uploaded file."),
                model_name: str = Query(description="Model name.")
        ) -> FileResponse:
            contents = await file.read()
            data = base64.b64decode(base64.b64encode(contents).decode("utf-8"))

            if self.inference_server is None:
                self.inference_server = self.inference_server_cls(self.triton.get_url(), model_name, self.protocol)
                await self.inference_server.initialize()

            result = await self.request_manager(file.content_type, data)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pickle") as temp:
                with open(temp.name, "wb") as file:
                    pickle.dump(np.array(result), file)

            return FileResponse(temp.name, filename="data.pickle", media_type="application/octet-stream")

        config = uvicorn.Config(app, self.fastapi.host, self.fastapi.port, log_level="info")
        server = uvicorn.Server(config)
        return ServerRunner(server)


def parse_options() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--service_config", default="./custom/yolo/ensemble_yolo.yaml",
                        help="Path to service configuration.")
    return parser.parse_args()


def get_callable_from_string(path: str) -> Type[Any]:
    module_path, name = path.split(':')
    module = import_module(module_path)
    return getattr(module, name)


if __name__ == "__main__":
    args = parse_options()
    config = ServiceConfig.from_file(args.service_config)
    service = Service(
        inference_server_cls=get_callable_from_string(config.server),
        fastapi=config.fastapi,
        triton=config.triton,
        protocol=config.protocol
    )
    while service.runner.thread.is_alive():
        time.sleep(0.1)
