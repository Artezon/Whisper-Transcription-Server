import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
import asyncio
import os
import uuid
from typing import Optional
from faster_whisper import WhisperModel
import uvicorn
import time

MODEL_NAME = "large-v3"
MAX_SIMULTANEOUS_TASKS = 2
PRINT_SECOND_FRACTIONS = False

app = FastAPI()

print(f"Loading model {MODEL_NAME}...")
model = WhisperModel(MODEL_NAME, device="cuda", compute_type="float16")
print("Model loaded and ready for transcription requests")

semaphore = asyncio.Semaphore(MAX_SIMULTANEOUS_TASKS)
tasks: dict[str, "TranscriptionTask"] = {}  # store active tasks by UUID
task_queue: list["TranscriptionTask"] = []
task_queue_condition = asyncio.Condition()


class TranscriptionTask:
    def __init__(self, path: str, language: Optional[str] = None):
        self.uuid = str(uuid.uuid4())
        self.path = path
        self.language = language
        self.queue_position = None
        self.progress = 0.0
        self.segments: list[dict] = []
        self.finished = False
        self.cancelled = False
        self.error: Optional[str] = None
        self.subscribers: list[asyncio.Queue] = []

    def subscribe(self) -> asyncio.Queue:
        event_queue = asyncio.Queue()
        self.subscribers.append(event_queue)
        return event_queue
    
    def unsubscribe(self, event_queue: asyncio.Queue):
        try:
            self.subscribers.remove(event_queue)
        except ValueError:
            pass

    def publish_update(self, **kwargs):
        data = {k: v for k, v in kwargs.items() if v is not None}
        for q in self.subscribers:
            q.put_nowait(data)

    async def run(self):
        try:
            async with task_queue_condition:
                task_queue.append(self)
                self.queue_position = len(task_queue)
                self.publish_update(queue_pos=self.queue_position)

                while task_queue[0] != self:
                    await task_queue_condition.wait()
                    self.queue_position -= 1
                    self.publish_update(queue_pos=self.queue_position)
            
            async with semaphore:
                if task_queue[0] == self:
                    task_queue.pop(0)
                    self.queue_position = None
                    async with task_queue_condition:
                        task_queue_condition.notify_all()

                segments, info = await asyncio.to_thread(model.transcribe, self.path, self.language, multilingual=True)
                print(self.uuid, self.language, info.language, info.language_probability)
                self.language = info.language
                self.publish_update(language=self.language)

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
                    
                    current_time = time.time()
                    if current_time - last_notification_time > 0.1:
                        self.progress = round(segment.end / info.duration, 4)
                        self.publish_update(progress=self.progress, segments=new_segments)
                        print(self.uuid, f"{(self.progress * 100):.0f}%")
                        self.segments.extend(new_segments)
                        new_segments = []
                        last_notification_time = current_time
                
                self.progress = 1.0
                self.finished = True
                self.publish_update(progress=self.progress, finished=self.finished, segments=new_segments)
                print(self.uuid, "Finished")

        except asyncio.CancelledError:
            self.cancelled = True
            self.finished = True
            if self in task_queue:
                task_queue.pop(task_queue.index(self))
                async with task_queue_condition:
                    task_queue_condition.notify_all()
            print(self.uuid, "Task cancelled")
            self.publish_update(cancelled=self.cancelled, message="Cancelled by user")
            raise
        except Exception as e:
            self.error = str(e)
            self.finished = True
            if self in task_queue:
                task_queue.pop(task_queue.index(self))
                async with task_queue_condition:
                    task_queue_condition.notify_all()
            print(self.uuid, f"Transcription error: {self.error}")
            self.publish_update(error=self.error)


@app.post("/transcribe")
async def start_transcription(path: str, language: Optional[str] = None):
    path = path.strip(" '\"\n\t")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")

    task = TranscriptionTask(path, language)
    tasks[task.uuid] = task
    task.asyncio_task = asyncio.create_task(task.run())
    return {"task_id": task.uuid}


@app.get("/data/{task_id}")
async def get_transcription_data(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return {k: v for k, v in {
        "task_id": task.uuid,
        "language": task.language,
        "queue_pos": task.queue_position,
        "progress": task.progress,
        "segments": task.segments,
        "finished": task.finished,
        "cancelled": task.cancelled,
        "error": task.error,
    }.items() if v is not None and v != []}


@app.get("/stream/{task_id}")
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


@app.post("/cancel/{task_id}")
async def cancel_transcription(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Cancel the running asyncio task
    if not task.finished:
        task.asyncio_task.cancel()
        return {"task_id": task_id, "status": "cancelled"}
    else:
        return {"task_id": task_id, "status": "already finished"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
