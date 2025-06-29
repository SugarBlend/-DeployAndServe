import asyncio
import cv2
from functools import partial
import numpy as np
from queue import Queue, Empty
from threading import Thread
from typing import Dict, Union, Callable, List, Any
import time

from triton.configs import ServiceConfig, Url, Formats, ProtocolType
from triton.base import TritonRemote
from base_service import BaseService, collect_frames, parse_options, get_callable_from_string


class Service(BaseService):
    def __init__(
            self,
            inference_server_cls: Union[TritonRemote, Callable],
            fastapi: Url,
            triton: Url,
            protocol: ProtocolType
    ) -> None:
        super(Service, self).__init__(inference_server_cls, fastapi, triton, protocol)

    async def send_request(self, data: bytes) -> np.ndarray:
        try:
            buffer = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            tensors = self.inference_server.preprocess(frame)
            result = await self.inference_server.infer(feed_input=tensors)
            return self.inference_server.postprocess(result)
        except Exception as error:
            self.logger.error(error)

    async def send_requests(
            self,
            frame_queue: Queue[np.ndarray],
            thread_collector: Thread
    ) -> Dict[int, List[np.ndarray]]:
        tasks: List[asyncio.tasks.Task] = []
        results: Dict[int, List[np.ndarray]] = {}
        max_concurrently_tasks: int = 128
        counter: int = 0

        while not frame_queue.empty() or thread_collector.is_alive():
            if len(tasks) > max_concurrently_tasks:
                await asyncio.gather(*tasks)
                tasks.clear()
            else:
                try:
                    frame = frame_queue.get()
                    _, img_encoded = cv2.imencode(".jpg", frame)
                    task = asyncio.create_task(self.send_request(img_encoded.tobytes()))
                    callback = partial(lambda c, t: results.update({c: t.result().tolist()}), counter)
                    task.add_done_callback(callback)
                    tasks.append(task)
                    counter += 1
                except Empty:
                    await asyncio.sleep(0.01)

        if len(tasks):
            await asyncio.gather(*tasks)
            tasks.clear()
        return results

    async def request_manager(self, content_type: str, data: bytes) -> Dict[int, List[np.ndarray]]:
        media_type, container = content_type.split("/")
        if media_type == Formats.VIDEO:
            frame_queue = Queue(maxsize=256)
            collector_thread = Thread(target=collect_frames, args=(data, container, frame_queue))
            collector_thread.start()
            results = await self.send_requests(frame_queue, collector_thread)
            collector_thread.join()
            return results
        elif media_type == Formats.IMAGE:
            result = await self.send_request(data)
            return {0: [result]}
        else:
            raise NotImplementedError("Pass unsupported data type of file.")


if __name__ == "__main__":
    args = parse_options()
    args.service_config = "./custom/yolo/yolo.yaml"
    config = ServiceConfig.from_file(args.service_config)
    service = Service(
        inference_server_cls=get_callable_from_string(config.server),
        fastapi=config.fastapi,
        triton=config.triton,
        protocol=config.protocol
    )
    while service.runner.thread.is_alive():
        time.sleep(0.1)
