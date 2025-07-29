from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
import base64
import cv2
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import FileResponse, Response
import pickle
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest, make_asgi_app, Summary, Counter
import os
from queue import Queue
from imutils.video import FileVideoStream
from starlette.background import BackgroundTask
from starlette.exceptions import HTTPException
import tempfile
from tqdm import tqdm
from typing import Union, Callable, Optional, Any
import time
import uvicorn

from deploy2serve.triton.core.configs import Url, ProtocolType
from deploy2serve.triton.core.base.inference_server import TritonRemote
from deploy2serve.utils.logger import get_logger
from deploy2serve.utils.runner import ServerRunner


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


class BaseService(ABC):
    def __init__(
            self,
            inference_server_cls: Union[TritonRemote, Callable],
            fastapi: Url,
            triton: Url,
            protocol: ProtocolType
    ) -> None:
        self.inference_server_cls: Union[TritonRemote, Callable] = inference_server_cls
        self.fastapi: Url = fastapi
        self.triton: Url = triton
        self.protocol: ProtocolType = protocol

        self.inference_server: Optional[TritonRemote] = None
        self.runner: ServerRunner = self.create()
        self.logger = get_logger("service")

    @abstractmethod
    async def send_requests(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    async def request_manager(self, *args, **kwargs) -> Any:
        pass

    def create(self) -> ServerRunner:
        app = FastAPI()

        metrics_app = make_asgi_app()
        app.mount("/metrics", metrics_app)

        REQUEST_TIME = Summary(
            'predict_request_processing_seconds',
            'Time spent processing predict request'
        )
        REQUEST_COUNT = Counter(
            'predict_request_count_total',
            'Total predict requests count'
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
                data = base64.b64decode(base64.b64encode(contents).decode("utf-8"))

                self.inference_server = self.inference_server_cls(self.triton.get_url(), model_name, self.protocol)
                await self.inference_server.initialize()

                result = await self.request_manager(file.content_type, data)
                REQUEST_TIME.observe(time.time() - start_time)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pickle") as temp:
                    pickle.dump(result, temp)

                await self.inference_server.triton_client.close()

                return FileResponse(temp.name, filename="data.pickle", media_type="application/octet-stream",
                                    background=BackgroundTask(lambda: os.unlink(temp.name)))
            except Exception as error:
                raise HTTPException(status_code=500, detail=str(error))

        config = uvicorn.Config(app, self.fastapi.host, self.fastapi.port, log_level="error")
        server = uvicorn.Server(config)
        return ServerRunner(server)


def parse_options() -> Namespace:
    parser = ArgumentParser()
    from deploy2serve.utils.logger import get_project_root
    parser.add_argument("--service_config", default=f"{get_project_root()}/deploy2serve/triton/overrides/yolo/configs/ensemble.yaml", help="Path to service configuration.")
    return parser.parse_args()
