import asyncio
from argparse import ArgumentParser, Namespace
import base64
import cv2
from functools import partial
from fastapi.responses import FileResponse
from fastapi import FastAPI, File, UploadFile, Query
import numpy as np
import pickle
from importlib import import_module
from imutils.video import FileVideoStream
from queue import Queue
import tempfile
from tqdm import tqdm
from threading import Thread
from typing import Dict, Union, Callable, Optional, List, Type, Any
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

    async def request_manager(self, content_type: str, data: bytes) -> List[Dict[str, Any]]:
        media_type, container = content_type.split("/")
        if media_type == Formats.VIDEO:
            frame_queue = Queue(maxsize=32)
            collector_thread = Thread(target=self.collect_frames, args=(data, container, frame_queue))
            collector_thread.start()

            results: Dict[int, Dict[str, Any]] = {}
            tasks = []
            counter = 0
            while collector_thread.is_alive() or not frame_queue.empty():
                if not frame_queue.empty():
                    frame = frame_queue.get()
                    _, img_encoded = cv2.imencode(".jpg", frame)
                    task = asyncio.create_task(self.predict_handler(img_encoded.tobytes()))
                    callback = partial(lambda c, t: results.update({c: t.result()}), counter)
                    task.add_done_callback(callback)
                    tasks.append(task)
                    counter += 1
                else:
                    await asyncio.gather(*tasks)
                    tasks.clear()

            await asyncio.gather(*tasks)
            collector_thread.join()
            tasks.clear()
            return [value for key, value in sorted(results.items())]
        elif media_type == Formats.IMAGE:
            result = await self.predict_handler(data)
            return [result]
        else:
            raise NotImplementedError("Passed unsupported data type of file.")

    async def predict_handler(self, data: bytes) -> Dict[str, Any]:
        try:
            buffer = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            tensors = self.inference_server.preprocess(frame)
            result = await self.inference_server.infer(feed_input=tensors)
            return self.inference_server.postprocess(result)
        except Exception as error:
            self.logger.error(f"Processing error: {error}")

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
    parser.add_argument("--service_config", default="./custom/yolo/yolo.yaml", help="Path to service configuration")
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
