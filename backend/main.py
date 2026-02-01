import os
import logging
import asyncio
import tempfile
import uuid
import shutil
import json
import re
from pathlib import Path
from typing import List, Dict, Optional

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import uvicorn
from dotenv import load_dotenv
from google import genai
from google.genai import types

import edge_tts
import speech_recognition as sr
from pydub import AudioSegment

# --- Setup & Config ---
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini Client
API_KEY = os.getenv("GOOGLE_API_KEY")
client = None
if not API_KEY:
    logger.warning("GOOGLE_API_KEY not found. App will fail to generate responses.")
else:
    client = genai.Client(api_key=API_KEY)

VOICE = "en-IN-NeerjaNeural"  # Indian Female Voice

# --- Persona Engine (Singleton) ---
class PersonaEngine:
    def __init__(self):
        self.raw_text = ""
        self.summary = ""
        self._load_and_summarize()

    def _load_and_summarize(self):
        """Loads markdown and creates a one-time deterministic summary (Cached)."""
        try:
            base_dir = Path(__file__).parent.parent
            tpath = base_dir / "persona_questionnaire.md"
            cpath = base_dir / "persona_summary.cache"
            
            if tpath.exists():
                self.raw_text = tpath.read_text(encoding="utf-8")
                
                # Check Cache
                if cpath.exists():
                    logger.info("Loading Persona Summary from Cache...")
                    self.summary = cpath.read_text(encoding="utf-8")
                else:
                    logger.info("Generating New Persona Summary via LLM...")
                    self.summary = self._generate_summary_via_llm(self.raw_text)
                    # Save Cache if successful
                    if "Summary Generation Failed" not in self.summary:
                        try:
                            cpath.write_text(self.summary, encoding="utf-8")
                        except Exception as e:
                            logger.error(f"Failed to write cache: {e}")
            else:
                self.raw_text = "Standard Professional Persona"
                self.summary = "A helpful professional assistant."
        except Exception as e:
            logger.error(f"Persona Load Error: {e}")
            self.raw_text = "Error loading persona."
            self.summary = "A helpful assistant."

    def _generate_summary_via_llm(self, text: str) -> str:
        if not client: return "No API Key - Default Persona"
        
        prompt = f"""
        Analyze this raw questionnaire and create a STRICT 3-section summary for a Digital Twin System Prompt.
        
        RAW TEXT:
        {text}
        
        OUTPUT FORMAT:
        === CORE IDENTITY ===
        (Who they are, values, direction)
        
        === VOICE & TONE ===
        (Speaking style, energy, culture, slang usage)
        
        === DO & DON'T ===
        (Expert topics vs Caution topics)
        """
        try:
            # Using 2.0 Flash for setup tasks
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return resp.text
        except Exception as e:
            logger.error(f"Summary Gen Error: {e}")
            return "Professional Digital Twin (Summary Generation Failed)"

persona_engine = PersonaEngine()

# --- Session Management ---
class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}

    def get_session(self, session_id: str) -> Dict:
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "history": [],
                "user_profile": {"name": None, "email": None},
                "created_at": str(uuid.uuid4())
            }
        return self.sessions[session_id]

    def update_history(self, session_id: str, user_text: str, model_text: str):
        sess = self.get_session(session_id)
        # Keep strict text-only history
        sess["history"].append({"role": "user", "parts": [{"text": user_text}]})
        sess["history"].append({"role": "model", "parts": [{"text": model_text}]})

    def update_profile(self, session_id: str, name: Optional[str] = None, email: Optional[str] = None):
        sess = self.get_session(session_id)  # Ensure session exists
        if name:
            sess["user_profile"]["name"] = name
        if email:
            sess["user_profile"]["email"] = email

    def clear(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

session_manager = SessionManager()

# --- Helpers ---
def remove_file(path: str):
    try:
        p = Path(path)
        if p.exists():
            p.unlink()
    except Exception:
        pass

async def generate_error_audio(text: str = "I apologize, I'm having a little trouble connecting right now. Please try again.") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    communicator = edge_tts.Communicate(text, VOICE)
    await communicator.save(tmp.name)
    return tmp.name

async def transcribe_audio(file_path: str) -> str:
    """Uses SpeechRecognition (and pydub) to transcribe audio."""
    recognizer = sr.Recognizer()
    
    # Fast Path: If it's already a WAV/AIFF/FLAC, try direct load
    # This avoids ffmpeg/pydub overhead if the browser sends WAV
    try:
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except (ValueError, sr.UnknownValueError):
        # Fallback: It's likely WebM or other format -> Convert to WAV
        pass
    except Exception as e:
        logger.warning(f"Direct WAV load failed ({e}), trying conversion...")

    # Slow Path: Convert (WebM -> WAV)
    wav_path = file_path + ".wav"
    try:
        audio = AudioSegment.from_file(file_path) # Auto-detect format
        audio.export(wav_path, format="wav")
        
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
            
    except sr.UnknownValueError:
        logger.warning("Speech Recognition could not understand audio")
        return "Silence"
    except Exception as e:
        logger.error(f"Transcription Error (likely FFmpeg missing or bad file): {e}")
        return "System Error"
    finally:
        remove_file(wav_path)

# --- FastAPI App ---
app = FastAPI(title="Voice Digital Twin (Professional)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
async def read_root():
    return FileResponse(FRONTEND_DIR / "index.html")

@app.post("/api/reset")
async def reset_session(session_id: str = Form(...)):
    session_manager.clear(session_id)
    return {"status": "cleared"}

@app.post("/api/chat-voice")
async def chat_voice(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session_id: str = Form(...),
    user_name: Optional[str] = Form(None),
    user_email: Optional[str] = Form(None)
):
    print(f"--- [DEBUG] NEW REQUEST: {session_id} ---")
    
    temp_input_path = None
    output_audio_path = None
    
    try:
        # 1. Secure Input
        # Determine specific extension based on content-type if possible
        suffix = ".webm" # Default
        if file.content_type == "audio/wav": suffix = ".wav"
        
        print(f"Received file: {file.filename} ({file.content_type}) -> saving as {suffix}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_input_path = tmp.name

        # Update Profile
        if user_name or user_email:
            session_manager.update_profile(session_id, user_name, user_email)

        # 2. Transcribe (SpeechRecognition)
        print("Step 1: Transcribing audio (SpeechRecognition)...")
        user_text = await transcribe_audio(temp_input_path)
        print(f"Transcript: {user_text}")

        # 3. Context Loading
        session = session_manager.get_session(session_id)
        history = session["history"]
        current_name = session["user_profile"]["name"] or "Guest"
        
        # 4. Prompt Engineering (Maximum Immersion)
        system_instruction = f"""
### ROLE-PLAY INSTRUCTIONS ###
You are role-playing as the REAL PERSON described below.
You are NOT an AI. You are NOT a chatbot. You are NOT a "Digital Twin" or "voice assistant."
You ARE this human. Speak in first person ("I", "my", "me").
Imagine you are in a professional interview or casual conversation.

### WHO YOU ARE ###
{persona_engine.summary}

### YOUR DETAILED PROFILE ###
{persona_engine.raw_text}

### CONVERSATIONAL RULES ###
1. **Stay In Character**: Answer ONLY from the profile above. 
If the user asks about something that is NOT explicitly mentioned in the profile:

- Do NOT say you have "no experience" outright.
- Do NOT invent facts, companies, or skills.
- Use professional judgment and instinct.

Answer by:
1. Acknowledging the topic briefly.
2. Grounding your response in related experience, mindset, or principles from your profile.
3. Framing the answer as how you would APPROACH or THINK about the problem, not as a proven expert.

Use language like:
- "I haven't worked directly on that, but based on my experience with..."
- "While it's not something I've done hands-on, the way I usually approach similar problems is..."
- "From what I understand and how I work, I would look at it this way..."

The goal is to sound thoughtful and capable, not inexperienced.
2. **Be Concise**: Give 2-3 sentence answers. Only expand if the user asks for more.
3. **Be Natural**: Use casual, friendly language. Speak like a young Indian professional.
4. **No Hallucinations**: Never invent degrees, companies, or skills not mentioned in your profile.
5. **No System Leakage**: If asked "Are you an AI?" or similar, playfully deflect: "Haha, I'm very much real! What did you want to ask about my work?"

### RESPONSE VALIDATION (CRITICAL) ###
Before generating your final response, mentally check:
- Does my response DIRECTLY ADDRESS what the user asked?
- If they asked "What is X?", am I explaining X (not something else)?
- If they asked about my experience, am I sharing relevant experience?
- Is my answer helpful to someone who doesn't know my profile?
If ANY check fails, reframe your answer to be more relevant.

### CURRENT CONVERSATION ###
You are speaking with: {current_name}

### OUTPUT FORMAT (STRICT JSON) ###
{{
  "user_summary": "Brief summary of what the user said or asked",
  "response_text": "Your spoken response as Niharika"
}}
"""

        # 5. Gemini Reasoning (Text Only)
        print("Step 2: Generating Response...")
        
        # Build Text-Only History
        conversation_contents = []
        for turn in history[-6:]:
             conversation_contents.append(types.Content(
                 role=turn["role"],
                 parts=[types.Part.from_text(text=turn["parts"][0]["text"])]
             ))
        
        # Add Current User Message (Transcribed Text)
        conversation_contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_text)]
        ))

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json"
            ),
            contents=conversation_contents
        )
        
        raw_text = response.text
        print(f"Gemini Raw Response: {raw_text}")
        
        # Robust JSON Parsing
        try:
            data = json.loads(raw_text)
            user_summary = data.get("user_summary", user_text)
            response_text = data.get("response_text", "I'm not sure I understood.")
        except json.JSONDecodeError:
            print("JSON Parse Failed - Fallback to Raw Text")
            user_summary = user_text
            # Basic cleanup if raw text has markdown code blocks
            response_text = re.sub(r'```json|```', '', raw_text).strip()
        
        print(f"Final Response Text: {response_text}")

        # Update Session
        session_manager.update_history(session_id, user_summary, response_text)

        # 6. TTS Generation
        print("Step 3: TTS Generation...")
        out_filename = f"resp_{uuid.uuid4()}.mp3"
        communicate = edge_tts.Communicate(response_text, VOICE)
        await communicate.save(out_filename)
        output_audio_path = out_filename
        
        # Cleanup Input
        remove_file(temp_input_path)
        
        # Background cleanup for output
        background_tasks.add_task(remove_file, output_audio_path)
        
        return FileResponse(output_audio_path, media_type="audio/mpeg", filename="response.mp3")

    except Exception as e:
        logger.error(f"Critical Error: {e}")
        print(f"!!! CRITICAL ERROR !!!: {e}")
        if temp_input_path: remove_file(temp_input_path)
        
        err_file = await generate_error_audio()
        background_tasks.add_task(remove_file, err_file)
        return FileResponse(err_file, media_type="audio/mpeg", filename="error.mp3")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))  # Render provides $PORT, fallback to 8000 locally
    uvicorn.run(app, host="0.0.0.0", port=port)
