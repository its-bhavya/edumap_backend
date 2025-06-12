import whisper
from pydub import AudioSegment
import os
import tempfile

model = whisper.load_model("tiny")  

def transcribe(audio_path):
    result = model.transcribe(audio_path)
    return result['text']

def save_to_data(file_name, text):
    os.makedirs("data/transcripts", exist_ok=True)
    save_path = os.path.join("data/transcripts", f"{file_name}.txt")
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(text)
