import base64
import cv2
from fastapi import FastAPI, File, UploadFile, Query
import numpy as np
from utils.runner import ServerRunner
import tempfile
from tqdm import tqdm
from typing import Dict, Any, Union, Callable, Optional
import uvicorn

from configs import Url, Formats, ProtocolType
from triton.base import TritonRemote


class Service(object):
    def __init__(
            self,
            inference_server_cls: Union[TritonRemote, Callable],
            fastapi: Url,
            triton: Url,
            protocol: ProtocolType,
            show: bool = False,
            visualize_func: Optional[Callable] = None
    ) -> None:
        self.inference_server_cls: Union[TritonRemote, Callable] = inference_server_cls
        self.fastapi: Url = fastapi
        self.triton: Url = triton
        self.protocol: ProtocolType = protocol
        self.show: bool = show
        self.visualize_func: Optional[Callable] = visualize_func

        self.inference_server: Optional[TritonRemote] = None
        self.runner: ServerRunner = self.create()

    async def video_predict(self, data: bytes, container: str, *args, **kwargs) -> Dict[str, Any]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{container}") as tmp:
            tmp.write(data)
            source_path = tmp.name

        await self.inference_server.initialize()
        cap = cv2.VideoCapture(source_path, cv2.CAP_FFMPEG)

        with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as bar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                tensors = self.inference_server.preprocess(frame)
                feed_input = {self.inference_server.metadata["inputs"][idx]["name"]: tensor
                              for idx, tensor in enumerate(tensors)}
                result = await self.inference_server.infer(feed_input)
                outputs = self.inference_server.postprocess(result)
                if self.show and self.visualize_func:
                    self.visualize_func(frame, outputs, wait_time=1)
                bar.update()
        return {}

    async def image_predict(self, data: bytes, *args, **kwargs) -> Dict[str, Any]:
        buffer = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        await self.inference_server.initialize()
        tensors = self.inference_server.preprocess(frame)
        feed_input = {self.inference_server.metadata["inputs"][idx]["name"]: tensor
                      for idx, tensor in enumerate(tensors)}
        result = await self.inference_server.infer(feed_input)
        outputs = self.inference_server.postprocess(result)
        if self.show and self.visualize_func:
            self.visualize_func(frame, outputs)
        return {}

    def create(self) -> ServerRunner:
        app = FastAPI()

        funcs = {
            Formats.VIDEO: self.video_predict,
            Formats.IMAGE: self.image_predict
        }

        @app.post("/predict")
        async def predict(
                file: UploadFile = File(description="Uploaded file."),
                model_name: str = Query(description="Name of folder model which triton use as default load model.")
        ):
            media_type, container = file.content_type.split(sep="/")
            contents = await file.read()
            file_base64 = base64.b64encode(contents).decode("utf-8")
            data = base64.b64decode(file_base64)

            self.inference_server = self.inference_server_cls(self.triton.get_url(), model_name, self.protocol)
            result = await funcs[Formats(media_type)](data, container)
            cv2.destroyAllWindows()
            return result

        config = uvicorn.Config(app, self.fastapi.host, self.fastapi.port)
        server = uvicorn.Server(config)
        runner = ServerRunner(server)
        return runner
