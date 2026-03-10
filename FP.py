import streamlit as st
from faster_whisper import WhisperModel
import google.generativeai as genai
import asyncio
from pathlib import Path
from pytubefix import YouTube
import torch

# --- Configuration & UI Setup ---
CACHE_DIR = Path("media_cache")
CACHE_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="VeloScript AI | Ultra-Fast Summarizer",
    page_icon="⚡",
    layout="wide"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    /* Main Background and Font */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    /* Center Title */
    .main-title {
        font-size: 3rem !important;
        font-weight: 800;
        background: -webkit-linear-gradient(#00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Card-like containers */
    div[data-testid="stVerticalBlock"] > div:has(div.stMarkdown) {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Custom Button */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize Session State
if 'data' not in st.session_state:
    st.session_state.data = {"summary": None, "transcript": None, "title": None, "thumbnail": None}

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.header("Control Panel")
    api_key = st.text_input("Gemini API Key", type="password", help="Get your key from Google AI Studio")
    
    st.divider()
    st.subheader("Engine Settings")
    model_size = st.select_slider("Whisper Model", options=["tiny", "base", "small", "medium"], value="base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.caption(f"Running on: **{device.upper()}**")

@st.cache_resource
def load_faster_whisper(model_size, device):
    return WhisperModel(model_size, device=device, compute_type="int8")

async def generate_summary_async(model, text):
    prompt = f"""
    You are an expert content analyzer. Summarize the following transcript.
    Structure the response with:
    1. A concise overview (3 sentences max).
    2. Key Takeaways (bullet points with emojis).
    3. Sentiment/Tone of the video.
    
    Transcript: {text}
    """
    response = await model.generate_content_async(prompt)
    return response.text

# --- Main Interface ---
st.markdown('<h1 class="main-title">⚡ VeloScript AI</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8;'>Transform hours of video into minutes of reading with lightning-fast AI.</p>", unsafe_allow_html=True)

if api_key:
    genai.configure(api_key=api_key)
    # Using the latest Gemini 2.5 Flash for speed
    ai_model = genai.GenerativeModel('gemini-2.5-flash')
    
    # URL Input centered
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        url = st.text_input("", placeholder="Paste YouTube Link Here...", label_visibility="collapsed")
        proc_btn = st.button("🚀 Process Video", use_container_width=True)

    if proc_btn:
        if url:
            try:
                with st.status("🛠️ Working Magic...", expanded=True) as status:
                    # 1. Faster Download
                    yt = YouTube(url)
                    status.write(f"📥 Fetching: **{yt.title}**")
                    
                    audio_path = CACHE_DIR / f"{yt.video_id}.mp3"
                    if not audio_path.exists():
                        yt.streams.get_audio_only().download(output_path=str(CACHE_DIR), filename=f"{yt.video_id}.mp3")
                    
                    # 2. Faster-Whisper Transcription
                    status.write("✍️ Transcribing (CTranslate2 Optimized)...")
                    w_model = load_faster_whisper(model_size, device)
                    segments, _ = w_model.transcribe(str(audio_path), beam_size=5)
                    transcript = " ".join([seg.text for seg in segments])
                    
                    # 3. Async LLM Summary
                    status.write("🧠 Synthesizing Key Insights...")
                    summary = asyncio.run(generate_summary_async(ai_model, transcript))
                    
                    st.session_state.data = {
                        "summary": summary,
                        "transcript": transcript,
                        "title": yt.title,
                        "thumbnail": yt.thumbnail_url
                    }
                    status.update(label="✅ Analysis Complete!", state="complete")
            except Exception as e:
                st.error(f"Error: {e}")

    # --- Display Results ---
    if st.session_state.data["summary"]:
        st.divider()
        res_col1, res_col2 = st.columns([1, 2], gap="large")
        
        with res_col1:
            st.image(st.session_state.data["thumbnail"], use_container_width=True)
            st.subheader(st.session_state.data["title"])
            st.info("💡 Tip: Use the transcript tab to search for specific quotes.")

        with res_col2:
            tab1, tab2 = st.tabs(["✨ Summary", "📝 Full Transcript"])
            with tab1:
                st.markdown(st.session_state.data["summary"])
            with tab2:
                st.caption("Auto-generated transcript:")
                st.text_area("raw_text", st.session_state.data["transcript"], height=450, label_visibility="collapsed")
else:
    st.warning("⚠️ Please enter your Gemini API Key in the sidebar to unlock the AI.")

# --- Footer ---
st.markdown("<br><hr><center style='color: #64748b;'>Built by Gaurang Kodwani</center>", unsafe_allow_html=True)