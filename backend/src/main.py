import json
import aiofiles
from fastapi import FastAPI, APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.responses import HTMLResponse, StreamingResponse
import asyncio
import os
import uuid
from typing import List, Optional
from fastapi.templating import Jinja2Templates
from faster_whisper import WhisperModel
import uvicorn
import time
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.environ["MODEL_NAME"]
NUM_WORKERS = int(os.environ["NUM_WORKERS"])
AUDIO_DIR = os.environ["AUDIO_DIR"]
HOST = os.environ["HOST"]
PORT = int(os.environ["PORT"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    for i in range(NUM_WORKERS):
        workers.append(asyncio.create_task(worker_loop(i)))
    print(f"Created {NUM_WORKERS} workers")
    yield
    for w in workers:
        w.cancel()


app = FastAPI(lifespan=lifespan)
api = APIRouter(prefix="/api")
templates = Jinja2Templates(directory="templates")\

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ["FRONTEND_CORS"].split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print(f"Loading model {MODEL_NAME}...")
model = WhisperModel(MODEL_NAME, device="cuda", compute_type="float16")
print("Model loaded and ready for transcription requests")

tasks: dict[str, "TranscriptionTask"] = {}
task_queue: asyncio.Queue["TranscriptionTask"] = asyncio.Queue()
workers: List[asyncio.Task] = []
lock: asyncio.Lock = asyncio.Lock()


class TranscriptionTask:
    def __init__(self, uuid: str, path: str, name: str, language: Optional[str] = None):
        self.uuid = uuid
        self.name = name
        self.path = path
        self.language = language
        self.progress = None
        self.segments: list[dict] = []
        self.finished = False
        self.cancelled = False
        self.error: Optional[str] = None
        self.subscribers: list[asyncio.Queue] = []
        self.worker_id = None

    def subscribe(self) -> asyncio.Queue:
        event_queue = asyncio.Queue()
        self.subscribers.append(event_queue)
        return event_queue
    
    def unsubscribe(self, event_queue: asyncio.Queue):
        try:
            self.subscribers.remove(event_queue)
        except ValueError:
            pass

    async def publish_update(self, **kwargs):
        for q in self.subscribers:
            await q.put(kwargs)
    
    async def run(self, worker_id: int):
        if self.cancelled:
            return

        self.worker_id = worker_id
        print(f"Worker {worker_id}: starting {self.uuid}")
        segments, info = await asyncio.to_thread(model.transcribe, self.path, self.language, multilingual=True)
        print(self.uuid, self.language, info.language, info.language_probability)
        self.language = info.language
        self.progress = 0.0
        await self.publish_update(progress=self.progress, language=self.language)

        def generate_segment():
            try:
                return next(segments)
            except StopIteration:
                return

        new_segments: list[dict] = []
        last_notification_time = 0
        while True:
            segment = await asyncio.to_thread(generate_segment)
            if segment is None:
                break
            seg_dict = {"start": round(segment.start, 2), "end": round(segment.end, 2), "text": segment.text.strip()}
            new_segments.append(seg_dict)

            now = time.time()
            if now - last_notification_time > 0.1:
                self.progress = round(segment.end / info.duration, 4)
                await self.publish_update(progress=self.progress, segments=new_segments)
                print(self.uuid, f"{(self.progress * 100):.0f}%")
                self.segments.extend(new_segments)
                new_segments = []
                last_notification_time = now

        self.progress = 1.0
        self.finished = True
        await self.publish_update(progress=self.progress, finished=self.finished, segments=new_segments)
        print(f"Worker {worker_id}: finished {self.uuid}")


async def worker_loop(worker_id: int):
    while True:
        async with lock:
            task: TranscriptionTask = await task_queue.get()
        print(f"Queue: {[i.uuid for i in list(task_queue._queue)]}")
        
        queue_list = list(task_queue._queue)
        for i, t in enumerate(queue_list):
            await t.publish_update(queue_pos=i + 1)

        try:
            await task.publish_update(queue_pos=None)
            await task.run(worker_id)
        except asyncio.CancelledError:
            task.cancelled = True
            task.finished = True
            print(task.uuid, "Task cancelled")
            await task.publish_update(finished=task.finished, cancelled=task.cancelled, message="Cancelled by user")
            raise
        except Exception as e:
            task.error = str(e)
            task.finished = True
            await task.publish_update(finished=task.finished, error=task.error)
            print(f"Worker {worker_id}: task {task.uuid} error {e}")
        finally:
            task_queue.task_done()


async def remove_from_queue(item) -> bool:
    async with lock:
        try:
            task_queue._queue.remove(item)
            print(f"Queue: {[i.uuid for i in list(task_queue._queue)]}")
            return True
        except ValueError:
            return False


@api.post("/transcribe")
async def start_transcription(request: Request, file: UploadFile = File(...), language: Optional[str] = Form(None)):
    os.makedirs(AUDIO_DIR, exist_ok=True)
    task_uuid = str(uuid.uuid4())
    file_path = f"{AUDIO_DIR}/{task_uuid}_{file.filename}"

    async with aiofiles.open(file_path, "wb") as out_f:
        while chunk := await file.read(1024 * 1024):
            if await request.is_disconnected():
                print("Client disconnected, aborting upload")
                return
            await out_f.write(chunk)

    task = TranscriptionTask(task_uuid, file_path, file.filename, language)
    tasks[task.uuid] = task
    await task_queue.put(task)
    print(f"Queue: {[i.uuid for i in list(task_queue._queue)]}")
    
    return {"task_id": task.uuid}


@api.get("/data/{task_id}")
async def get_transcription_data(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # calculate queue position
    queue_list = list(task_queue._queue)
    queue_pos = None
    if not task.finished and not task.cancelled and task in queue_list:
        queue_pos = queue_list.index(task) + 1

    return {k: v for k, v in {
        "task_id": task.uuid,
        "name": task.name,
        "language": task.language,
        "queue_pos": queue_pos,
        "progress": task.progress,
        "segments": task.segments,
        "finished": task.finished,
        "cancelled": task.cancelled,
        "error": task.error,
    }.items() if v is not None and v != []}


@api.get("/stream/{task_id}")
async def stream_transcription(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    event_queue = task.subscribe()

    async def event_generator():
        data = await get_transcription_data(task_id)
        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        try:
            while True:
                if data.get("finished") or data.get("error"):
                    break
                data = await event_queue.get()
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        finally:
            task.unsubscribe(event_queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@api.post("/cancel/{task_id}")
async def cancel_transcription(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if not task.finished:
        task.cancelled = True
        task.finished = True
        if task.worker_id is not None:
            workers[task.worker_id].cancel()
            workers[task.worker_id] = asyncio.create_task(worker_loop(task.worker_id))
        else:
            await task.publish_update(finished=task.finished, cancelled=task.cancelled, message="Cancelled by user")
            remove_from_queue(task)

            global task_queue
            queue_list = list(task_queue._queue)
            for i, task in enumerate(queue_list):
                await task.publish_update(queue_pos=i + 1)
        return {"task_id": task_id, "status": "cancelled"}
    else:
        return {"task_id": task_id, "status": "already finished"}


@api.get("/status")
async def get_status():
    return {
        "ready": True,
        "in_queue": task_queue.qsize()
    }


app.include_router(api)

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
