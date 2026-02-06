import streamlit as st
from faster_whisper import WhisperModel
import google.generativeai as genai
import asyncio
from pathlib import Path
from pytubefix import YouTube
import torch

# --- Configuration ---
CACHE_DIR = Path("media_cache")
CACHE_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Ultra-Fast AI Summarizer", page_icon="⚡")

# Initialize Session State
if 'data' not in st.session_state:
    st.session_state.data = {"summary": None, "transcript": None, "title": None}

with st.sidebar:
    st.header("AI Settings")
    api_key = st.text_input("Gemini API Key", type="password")
    # faster-whisper handles model sizes similarly but much faster
    model_size = st.select_slider("Whisper Model", options=["tiny", "base", "small", "medium"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_faster_whisper(model_size, device):
    """Loads model once and caches it in RAM/VRAM."""
    # compute_type="int8" makes it even faster on CPU
    return WhisperModel(model_size, device=device, compute_type="int8")

async def generate_summary_async(model, text):
    prompt = f"Provide a detailed summary and bulleted key takeaways for this transcript:\n\n{text}"
    response = await model.generate_content_async(prompt)
    return response.text

# --- Main Application ---
st.title("⚡ Ultra-Fast Video Summarizer")

if api_key:
    genai.configure(api_key=api_key)
    ai_model = genai.GenerativeModel('gemini-2.5-flash')
    st.text_inputGEMINI_KEY = "AIzaSyCjPLBqTHChWk13M6-MtwuS-XJNNiFzfJ0"
    
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

    if st.button("Start Processing", use_container_width=True):
        if url:
            try:
                with st.status("🚀 Processing at Warp Speed...", expanded=True) as status:
                    # 1. Faster Download
                    yt = YouTube(url)
                    status.write(f"Downloading: {yt.title}")
                    audio_path = CACHE_DIR / f"{yt.video_id}.mp3"
                    if not audio_path.exists():
                        yt.streams.get_audio_only().download(output_path=str(CACHE_DIR), filename=f"{yt.video_id}.mp3")
                    
                    # 2. Faster-Whisper Transcription
                    status.write("Transcribing with CTranslate2 (4x Faster)...")
                    w_model = load_faster_whisper(model_size, device)
                    # segments is a generator, making it memory efficient
                    segments, info = w_model.transcribe(str(audio_path), beam_size=5)
                    transcript = " ".join([seg.text for seg in segments])
                    
                    # 3. Async LLM Summary
                    status.write("Generating AI Summary...")
                    summary = asyncio.run(generate_summary_async(ai_model, transcript))
                    
                    # Store in session state
                    st.session_state.data = {
                        "summary": summary,
                        "transcript": transcript,
                        "title": yt.title
                    }
                    status.update(label="Complete!", state="complete")

            except Exception as e:
                st.error(f"Runtime Error: {e}")

    # --- Persistent Results Display ---
    if st.session_state.data["summary"]:
        st.divider()
        tab1, tab2 = st.tabs(["✨ Summary", "📝 Full Transcript"])
        with tab1:
            st.markdown(f"### {st.session_state.data['title']}")
            st.write(st.session_state.data["summary"])
        with tab2:
            st.text_area("Raw Text", st.session_state.data["transcript"], height=400)
else:
    st.info("Enter your Gemini API Key to begin.")
    st.text_inputGEMINI_KEY = "AIzaSyCjPLBqTHChWk13M6-MtwuS-XJNNiFzfJ0"