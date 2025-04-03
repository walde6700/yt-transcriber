from fastapi import FastAPI, Request
import subprocess
import whisperx
import uuid
import os

app = FastAPI()


@app.get("/")
def root():
    return {"status": "API is running!"}


@app.post("/transcribe")
async def transcribe(request: Request):
    data = await request.json()
    youtube_url = data["youtube_url"]

    uid = str(uuid.uuid4())
    mp3_file = f"{uid}.mp3"

    # Download YouTube audio using yt-dlp
    subprocess.run([
        "yt-dlp", "--extract-audio", "--audio-format", "mp3",
        "-o", mp3_file, youtube_url
    ], check=True)

    # Load WhisperX model (CPU by default; use "cuda" if using GPU)
    model = whisperx.load_model("large-v2", device="cpu")
    audio = whisperx.load_audio(mp3_file)
    result = model.transcribe(audio, batch_size=16)

    # Diarization (speaker separation)
    hf_token = os.environ.get("HF_TOKEN")
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token)
    diarize_segments = diarize_model(audio, result["segments"])
    result["segments"] = whisperx.assign_word_speakers(diarize_segments, result["segments"])

    # Clean up
    os.remove(mp3_file)

    return result["segments"]
