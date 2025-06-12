from fastapi import FastAPI
from pydantic import BaseModel
import dspy
import os
import json
import re
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

lm = dspy.LM("gemini/gemini-2.0-flash", api_key=api_key)
dspy.configure(lm=lm)

class StructureExtractionSignature(dspy.Signature):
    transcript = dspy.InputField(desc="The full transcript of a lecture or explanation.")
    central_topic = dspy.OutputField(desc="The central concept of the transcript.")
    subtopics = dspy.OutputField(
        desc="""A raw JSON list where each subtopic has:
        - 'title': string
        - 'description': string
        - optional 'children': a list of similar objects"""
    )

class MindmapExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.teleprompt = dspy.ChainOfThought(StructureExtractionSignature)

    def forward(self, transcript: str):
        return self.teleprompt(transcript=transcript)

# FastAPI
app = FastAPI()
extractor = MindmapExtractor()

class TranscriptRequest(BaseModel):
    transcript: str

@app.post("/extract")
def extract_json(req: TranscriptRequest):
    try:
        result = extractor.forward(transcript=req.transcript)

        def clean_json_field(field: str):
            field = re.sub(r"^```(?:json)?\n?", "", field.strip())
            field = re.sub(r"\n?```$", "", field)
            return json.loads(field)

        subtopics = clean_json_field(result.subtopics)
        data = {"central_topic": result.central_topic, "subtopics": subtopics}

        os.makedirs("data/json", exist_ok=True)
        json_name = result.central_topic
        json_name = json_name.replace(" ","-")
        path = f"data/json/{json_name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return {"success": True, "path": path, "data": data}
    except Exception as e:
        return {"success": False, "error": str(e)}
