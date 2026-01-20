import tkinter as tk
import sounddevice as sd
import numpy as np
import webrtcvad
import whisper
import asyncio
import edge_tts
import soundfile as sf
import io
import time
from openai import OpenAI
from dotenv import load_dotenv

# ================== SETUP ==================
load_dotenv()
client = OpenAI()

FS = 16000
FRAME_DURATION = 30  # ms
FRAME_SIZE = int(FS * FRAME_DURATION / 1000)

MAX_RECORD_TIME = 25
MAX_SILENCE_TIME = 1.5
QUESTION_LIMIT = 8

vad = webrtcvad.Vad(2)

system_prompt = """
Act like you are an interviewer and you are taking an interview in the tech stack of machine learning
for a company named Dynamic Multitech Solutions which develops inkjet printers and is based in Noida.

Rules:
1. Ask exactly 8 questions.
2. Start only after user says "hi".
3. Give company intro before first question.
4. Ask ONE question at a time.
5. Use english only.
6. No emojis.
7. After 8th answer say only: Thank you. Have a nice day.
"""

messages = [{"role": "system", "content": system_prompt}]
whisper_model = whisper.load_model("base")
question_count = 0

# ================== AI CALL ==================
def call_ai():
    res = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return res.choices[0].message.content

# ================== TEXT TO SPEECH ==================
async def speak(text):
    communicate = edge_tts.Communicate(
        text=text,
        voice="en-IN-PrabhatNeural",
        rate="+20%",
        pitch="+10Hz"
    )

    audio_bytes = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes += chunk["data"]

    audio_stream = io.BytesIO(audio_bytes)
    data, sr = sf.read(audio_stream, dtype="float32")
    sd.play(data, sr)
    sd.wait()

# ================== WEBRTCVAD RECORD ==================
def record_until_silence():
    status_label.config(text="🎤 Listening...")
    root.update()

    frames = []
    speech_detected = False
    silence_start = None
    start_time = time.time()

    with sd.InputStream(
        samplerate=FS,
        channels=1,
        dtype="int16",
        blocksize=FRAME_SIZE
    ) as stream:

        while True:
            frame, _ = stream.read(FRAME_SIZE)
            frame_bytes = frame.tobytes()
            is_speech = vad.is_speech(frame_bytes, FS)

            now = time.time()

            if now - start_time > MAX_RECORD_TIME:
                break

            if is_speech:
                speech_detected = True
                silence_start = None
                frames.append(frame.copy())
            else:
                if speech_detected:
                    if silence_start is None:
                        silence_start = now
                    elif now - silence_start > MAX_SILENCE_TIME:
                        break

    if not frames:
        return np.zeros((int(FS * 0.5), 1), dtype=np.int16)

    return np.concatenate(frames, axis=0)

# ================== INTERVIEW LOOP ==================
def interview_loop():
    global question_count

    audio = record_until_silence()

    # 🔥 CRITICAL FIX HERE
    audio_float = np.squeeze(audio).astype(np.float32) / 32768.0

    result = whisper_model.transcribe(audio_float, fp16=False)
    user_text = result["text"].strip()

    status_label.config(text=f"You said: {user_text}")
    root.update()

    messages.append({"role": "user", "content": user_text})

    ai_reply = call_ai()
    messages.append({"role": "assistant", "content": ai_reply})

    status_label.config(text="🔊 Interviewer speaking...")
    root.update()
    asyncio.run(speak(ai_reply))

    if "Thank you. Have a nice day." in ai_reply:
        status_label.config(text="Interview completed successfully.")
        return

    question_count += 1
    if question_count < QUESTION_LIMIT:
        time.sleep(0.8)
        interview_loop()

# ================== UI ==================
root = tk.Tk()
root.title("Dynamic Multitech Interview")
root.geometry("650x320")
root.resizable(False, False)

tk.Label(root, text="Dynamic Multitech Solutions",
         font=("Arial", 20, "bold")).pack(pady=10)

tk.Label(root, text="Machine Learning Interview (Voice Based)",
         font=("Arial", 12)).pack()

status_label = tk.Label(root, text="Interview starting...",
                        font=("Arial", 12), wraplength=560)
status_label.pack(pady=40)

# ================== START ==================
def start_interview():
    asyncio.run(
        speak(
            "Welcome to Dynamic Multitech Solutions. "
            "This inteview will be voice based. "
            "Please say hi to start the interview. "
        )
    )
    interview_loop()

root.after(1000, start_interview)
root.mainloop()
