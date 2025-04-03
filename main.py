from fastapi import FastAPI, Request
import subprocess
import whisperx
import uuid
import os

app = FastAPI()

@app.post("/transcribe")
async def transcribe(request: Request):
    data = await request.json()
    youtube_url = data["youtube_url"]

    uid = str(uuid.uuid4())
    mp3_file = f"{uid}.mp3"

    subprocess.run([
        "yt-dlp", "--extract-audio", "--audio-format", "mp3",
        "-o", mp3_file, youtube_url
    ])

    model = whisperx.load_model("large-v2", device="cpu")
    audio = whisperx.load_audio(mp3_file)
    result = model.transcribe(audio, batch_size=16)

    diarize_model = whisperx.DiarizationPipeline(use_auth_token=os.environ.get("HF_TOKEN"))
    diarize_segments = diarize_model(audio, result["segments"])
    result["segments"] = whisperx.assign_word_speakers(diarize_segments, result["segments"])

    os.remove(mp3_file)
    return result