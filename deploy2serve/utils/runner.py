from threading import Thread

import uvicorn


class ServerRunner(object):
    def __init__(self, server: uvicorn.Server) -> None:
        self.server: uvicorn.Server = server

        self.thread = Thread(target=server.run)
        self.thread.start()

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join()
