import asyncio
import threading

class AsyncTaskQueue:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.queue = asyncio.Queue()

        t = threading.Thread(target=self._start_loop, daemon=True)
        t.start()

        asyncio.run_coroutine_threadsafe(self.worker(), self.loop)

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def worker(self):
        while True:
            task = await self.queue.get()
            try:
                await task
            except Exception as e:
                print("Task error:", e)
            self.queue.task_done()

    def add_task(self, coro):
        asyncio.run_coroutine_threadsafe(
            self.queue.put(coro), self.loop
        )
