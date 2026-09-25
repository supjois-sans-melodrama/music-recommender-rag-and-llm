import base64
import os
import time
import pandas as pd
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util

# Safely attempt to load local .env without failing on Streamlit Cloud
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ==========================================
# 0. API KEY CONFIGURATION
# ==========================================
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            API_KEY = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

# ==========================================
# 1. PAGE CONFIG & HIGH-CONTRAST STYLES
# ==========================================
st.set_page_config(
    page_title="Flow Music Studio Pro",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* TIGHT LAYOUT */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1.5rem !important;
        max-width: 1050px !important;
    }

    div[data-testid="stVerticalBlock"] > div:empty {
        display: none !important;
    }

    /* GLOBAL TYPOGRAPHY */
    html, body, p, h1, h2, h3, h4, h5, h6, label, input, textarea, button, li {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #0F172A;
        line-height: 1.5 !important;
    }

    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 50%, #e0f2fe 100%) !important;
    }

    /* HERO SECTION */
    .hero-container {
        text-align: center;
        padding: 0.5rem 1rem 1.2rem 1rem;
        max-width: 850px;
        margin: 0 auto;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(2, 177, 200, 0.08);
        color: #02B1C8 !important;
        border: 1px solid rgba(2, 177, 200, 0.25);
        border-radius: 30px;
        padding: 4px 12px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #02B1C8;
        border-radius: 50%;
    }

    .hero-title {
        font-size: clamp(2rem, 4vw, 2.8rem) !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
        color: #0F172A !important;
        margin-bottom: 0.5rem !important;
        line-height: 1.3 !important;
    }

    .highlight-word {
        background: linear-gradient(135deg, #02B1C8 0%, #19A0A8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        font-size: clamp(0.95rem, 1.5vw, 1.1rem) !important;
        color: #475569 !important;
        margin: 0 auto 0.8rem auto !important;
        font-weight: 600 !important;
        line-height: 1.4 !important;
    }

    .metrics-bar {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-bottom: 1rem;
        flex-wrap: wrap;
    }

    .metric-pill {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(203, 213, 225, 0.8);
        border-radius: 10px;
        padding: 5px 14px;
        font-size: 0.82rem;
        font-weight: 700;
        color: #0F172A !important;
    }

    /* GLASS CARDS & HIGH-CONTRAST CONTAINER OVERRIDES */
    .studio-card {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(16px) !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 14px !important;
        padding: 1.25rem 1.5rem !important;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05) !important;
        margin-bottom: 1rem !important;
    }

    .studio-card h1, .studio-card h2, .studio-card h3, 
    .studio-card h4, .studio-card h5, .studio-card h6 {
        color: #0F172A !important;
        font-weight: 800 !important;
        margin-top: 0.6rem !important;
        margin-bottom: 0.6rem !important;
        line-height: 1.35 !important;
    }

    .studio-card p, .studio-card li, .studio-card span {
        color: #334155 !important;
        font-weight: 600 !important;
        line-height: 1.55 !important;
        margin-bottom: 0.5rem !important;
    }

    .studio-card strong, .studio-card b {
        color: #0F172A !important;
        font-weight: 800 !important;
    }

    .card-header {
        font-size: 1.15rem;
        font-weight: 800;
        color: #0F172A !important;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
        line-height: 1.3 !important;
    }

    /* TAB HEADINGS NO-OVERLAP FIX */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(241, 245, 249, 0.9);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #CBD5E1;
    }

    .stTabs [data-baseweb="tab"] {
        height: auto !important;
        min-height: 42px !important;
        border-radius: 8px;
        padding: 8px 18px !important;
        font-weight: 800 !important;
        color: #334155 !important;
        background-color: transparent !important;
        white-space: nowrap !important;
        display: inline-flex !important;
        align-items: center !important;
    }

    .stTabs [data-baseweb="tab"] * {
        color: #334155 !important;
        font-weight: 800 !important;
        font-size: 0.95rem !important;
        line-height: 1.2 !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
        border: 1px solid #94A3B8 !important;
    }

    .stTabs [aria-selected="true"] * {
        color: #02B1C8 !important;
        font-weight: 800 !important;
    }

    /* BOLD HIGH-CONTRAST LABELS */
    label[data-testid="stWidgetLabel"], 
    .stSelectbox label, 
    .stTextInput label, 
    .stTextArea label {
        color: #0F172A !important;
        font-weight: 800 !important;
        font-size: 0.92rem !important;
        margin-bottom: 6px !important;
        letter-spacing: -0.01em !important;
    }

    /* DROPDOWNS (SELECTBOX) */
    div[data-testid="stSelectbox"] > div,
    div[data-baseweb="select"],
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] [role="combobox"] {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        border: 2px solid #0F172A !important;
        border-radius: 10px !important;
        color: #0F172A !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06) !important;
    }

    div[data-baseweb="select"] * {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
    }

    div[data-baseweb="select"] svg {
        fill: #0F172A !important;
        color: #0F172A !important;
    }

    div[data-baseweb="select"]:hover > div,
    div[data-testid="stSelectbox"]:hover > div {
        border-color: #02B1C8 !important;
        box-shadow: 0 0 0 3px rgba(2, 177, 200, 0.25) !important;
    }

    /* INPUT FIELDS & TEXTAREAS */
    .stTextInput input, .stTextArea textarea {
        background-color: #FFFFFF !important;
        border: 2px solid #0F172A !important;
        border-radius: 10px !important;
        color: #0F172A !important;
        font-weight: 700 !important;
    }

    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #02B1C8 !important;
        box-shadow: 0 0 0 3px rgba(2, 177, 200, 0.25) !important;
    }

    div[data-testid="stInputInstructions"] {
        display: none !important;
    }

    .stTextInput input::placeholder, .stTextArea textarea::placeholder {
        color: #64748B !important;
        font-weight: 500 !important;
        opacity: 0.8 !important;
    }

    /* BUTTONS */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #19A0A8 0%, #02B1C8 100%) !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
        font-weight: 800 !important;
        font-size: 0.95rem !important;
        padding: 0.65rem 1rem !important;
        box-shadow: 0 4px 12px rgba(25, 160, 168, 0.3) !important;
        width: 100%;
    }

    .stButton > button[kind="primary"] * {
        color: white !important;
    }

    .stButton > button[kind="secondary"] {
        background: #FFFFFF !important;
        color: #0F172A !important;
        border: 2px solid #0F172A !important;
        border-radius: 10px !important;
        font-weight: 800 !important;
        font-size: 0.95rem !important;
        padding: 0.65rem 1rem !important;
        width: 100%;
    }

    /* RESULT ANCHOR CARDS */
    .result-anchor-card {
        background: #F8FAFC;
        border: 1.5px solid #CBD5E1;
        border-left: 5px solid #02B1C8;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
    }

    .result-anchor-title {
        font-size: 1rem;
        font-weight: 800;
        color: #0F172A !important;
        margin-bottom: 0.4rem;
        line-height: 1.3 !important;
    }

    .result-anchor-sub {
        font-size: 0.88rem;
        color: #334155 !important;
        font-weight: 600;
        line-height: 1.5 !important;
    }

    /* PROMPT CONTAINER BAR */
    .prompt-bar-container {
        background: #FFFFFF;
        border: 1.5px solid #CBD5E1;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 0.4rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
    }

    .prompt-bar-title {
        color: #02B1C8 !important;
        font-weight: 800 !important;
        font-size: 0.95rem !important;
        letter-spacing: -0.01em;
    }

    /* BULLETPROOF CODE BLOCK & SELECTION CONTRAST FIX */
    div[data-testid="stCode"] {
        background-color: #0F172A !important;
        background: #0F172A !important;
        border: 1.5px solid #334155 !important;
        border-radius: 10px !important;
    }

    div[data-testid="stCode"] pre,
    div[data-testid="stCode"] code,
    div[data-testid="stCode"] span {
        background-color: transparent !important;
        color: #F8FAFC !important;
    }

    div[data-testid="stCode"] ::selection,
    div[data-testid="stCode"] code::selection,
    div[data-testid="stCode"] pre::selection,
    div[data-testid="stCode"] span::selection {
        background: #02B1C8 !important;
        color: #FFFFFF !important;
    }

    /* DATAFRAME & TABLE STYLING */
    div[data-testid="stDataFrame"] {
        border: 2px solid #CBD5E1 !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03) !important;
    }

    div[data-testid="stDataFrame"] th {
        background-color: #0F172A !important;
        color: #FFFFFF !important;
        font-weight: 800 !important;
        font-size: 0.95rem !important;
    }

    div[data-testid="stDataFrame"] td {
        font-weight: 700 !important;
        color: #0F172A !important;
    }

    /* NOTIFICATION / ALERT BOX FIX */
    div[data-testid="stNotification"],
    div[data-testid="stAlert"] {
        background-color: #FEF3C7 !important;
        border: 1.5px solid #F59E0B !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
    }

    div[data-testid="stNotification"] *,
    div[data-testid="stAlert"] * {
        color: #78350F !important;
        font-weight: 700 !important;
        line-height: 1.4 !important;
    }
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-container">
    <div class="status-badge">
        <span class="status-dot"></span> PRODUCER ENGINE V3.6 ACTIVE
    </div>
    <h1 class="hero-title">Create the <span class="highlight-word">song</span> you imagine.</h1>
    <p class="hero-subtitle">Get music prompts and song ideas to generate your best songs.</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA INGESTION ENGINE
# ==========================================
@st.cache_data(ttl=60)
def load_data():
    try:
        import gspread
        from oauth2client.service_account import ServiceAccountCredentials
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        
        if "gcp_service_account" in st.secrets:
            creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(st.secrets["gcp_service_account"]), scope)
        else:
            creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
            
        client = gspread.authorize(creds)
        sheet = client.open("flowmusic").sheet1
        df = pd.DataFrame(sheet.get_all_records())
        df.fillna("", inplace=True)
        return df
    except Exception:
        try:
            return pd.read_csv("flowmusic.csv").fillna("")
        except Exception:
            return None

df = load_data()
if df is None or df.empty:
    st.error("No dataset found. Ensure 'flowmusic.csv' or your Google Sheet is accessible.")
    st.stop()

# Dynamic Metrics Bar
st.markdown(f"""
<div class="metrics-bar">
    <div class="metric-pill">🎵 <b>{len(df)}</b> Library Style Anchors</div>
    <div class="metric-pill">⚡ <b>Sub-Second</b> RAG Indexing</div>
    <div class="metric-pill">🤖 <b>Multi-Tier Fallback</b> Engine</div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 3. VECTOR INDEXING ENGINE
# ==========================================
@st.cache_resource
def load_encoder():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

with st.spinner("Initializing neural studio engine..."):
    bi_encoder = load_encoder()

@st.cache_resource
def get_embeddings(_dataframe):
    passages = [
        f"Title: {r['Song']} | Genre: {r['Genre']} | Creature: {r['Main Creature']} | Summary: {r['Summary']}"
        for _, r in _dataframe.iterrows()
    ]
    candidate_embeddings = bi_encoder.encode(passages, convert_to_tensor=True, normalize_embeddings=True)
    return passages, candidate_embeddings

passages, candidate_embeddings = get_embeddings(df)

# ==========================================
# 4. TABBED APPLICATION INTERFACE
# ==========================================
tab_generator, tab_library, tab_architecture = st.tabs([
    "🎛️ Studio Generator", 
    "📚 Library Explorer", 
    "📖 Architecture & Tech Stack"
])

# -----------------------------------------------------------------------------
# TAB 1: STUDIO GENERATOR
# -----------------------------------------------------------------------------
with tab_generator:
    st.markdown('<div class="studio-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">🎼 Track Controls & Specifications</div>', unsafe_allow_html=True)
    
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        genre_option = st.selectbox(
            "Genre Fusion:",
            [
                "Epic Cinematic Folk-Bass Fusion",
                "Chamber Pop / Baroque Art-Pop",
                "City Pop / Lo-fi R&B",
                "Synthwave / Cyberpunk",
                "Indie Folk / Acoustic Narrative",
                "Bollywood-EDM Fusion",
                "Post-Punk / Surf Rock",
                "Instrumental / Ambient Soundscape",
                "Other (Specify in concept)"
            ]
        )

    with col_b:
        mood_option = st.selectbox(
            "Emotional Vibe:",
            [
                "Resilient & Triumphant",
                "Defiant & Energetic",
                "Melancholic & Reflective",
                "Ethereal & Mystical",
                "Humorous & Parody",
                "Calm & Atmospheric",
                "Dark & Suspenseful"
            ]
        )

    with col_c:
        language_option = st.selectbox(
            "Lyrics Language:",
            [
                "English",
                "Hindi",
                "Spanish",
                "Japanese",
                "Elvish / Fantasy Language",
                "Instrumental (No Lyrics)",
                "Dual Language / Mixed"
            ]
        )

    col_d, col_e = st.columns([1, 2])

    with col_d:
        creature_input = st.text_input(
            "Character / Focal Creature:",
            value="",
            placeholder="e.g., jovial elf, centaur"
        )

    with col_e:
        idea_input = st.text_area(
            "Story Narrative & Concept:",
            value="",
            placeholder="Describe what happens in the track...",
            height=85
        )

    btn_col1, btn_col2 = st.columns([1, 1])

    combined_query = (
        f"Genre: {genre_option}. Mood: {mood_option}. Language: {language_option}. "
        f"Character/Creature: {creature_input}. Concept: {idea_input}"
    )

    search_clicked = btn_col1.button("🔍 Search Library Matches", type="secondary")
    compose_clicked = btn_col2.button("✨ Get song prompt and lyrics", type="primary")

    st.markdown('</div>', unsafe_allow_html=True)

    # SEARCH RESULTS DISPLAY
    if search_clicked:
        with st.spinner("Retrieving matching anchors..."):
            query_embedding = bi_encoder.encode(
                f"Represent this sentence for searching relevant passages: {combined_query}", 
                convert_to_tensor=True, 
                normalize_embeddings=True
            )
            cosine_scores = util.cos_sim(query_embedding, candidate_embeddings)[0]
            top_results = torch.topk(cosine_scores, k=3)

            st.markdown('<div class="studio-card">', unsafe_allow_html=True)
            st.markdown("### 🎯 Top Relevant Style Anchors")
            
            for idx in top_results.indices:
                row = df.iloc[int(idx)]
                st.markdown(f"""
                <div class="result-anchor-card">
                    <div class="result-anchor-title">🎵 {row['Song']} <span style="color:#02B1C8; font-weight:800;">• {row['Genre']}</span></div>
                    <div class="result-anchor-sub"><b>Character/Creature:</b> {row['Main Creature']} <br><b>Summary:</b> {row['Summary']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="prompt-bar-container">
                    <div class="prompt-bar-title">💡 Try this prompt</div>
                </div>
                """, unsafe_allow_html=True)
                st.code(row['Prompt'], language="text")

            st.markdown('</div>', unsafe_allow_html=True)

    # GENERATION RESPONSE DISPLAY
    if compose_clicked:
        if not API_KEY:
            st.error("API Key missing! Set your GEMINI_API_KEY in your local .env or Streamlit Secrets.")
            st.stop()

        with st.spinner("AI Producer is composing your track prompt..."):
            query_embedding = bi_encoder.encode(
                f"Represent this sentence for searching relevant passages: {combined_query}", 
                convert_to_tensor=True, 
                normalize_embeddings=True
            )
            cosine_scores = util.cos_sim(query_embedding, candidate_embeddings)[0]
            top_results = torch.topk(cosine_scores, k=3)
            
            top_match_row = df.iloc[int(top_results.indices[0])]
            few_shot_prompt_text = ""
            for idx in top_results.indices:
                row = df.iloc[int(idx)]
                few_shot_prompt_text += f"\nExample Style (Genre: {row['Genre']}):\n{row['Prompt']}\n"

            from google import genai
            from google.genai import types
            from google.genai.errors import APIError

            client = genai.Client(api_key=API_KEY)
            system_instruction = (
                "You are a master AI music producer and prompt engineer. Create highly detailed, "
                "production-ready music generation prompts matching specifications."
            )
            
            rag_prompt = (
                f"USER SPECIFICATIONS:\n"
                f"- Primary Genre/Fusion: {genre_option}\n"
                f"- Mood/Vibe: {mood_option}\n"
                f"- Lyrics Language: {language_option}\n"
                f"- Character/Creature Focus: {creature_input}\n"
                f"- Core Story/Concept: {idea_input}\n\n"
                f"STYLE INSPIRATION EXAMPLES FROM LIBRARY:\n"
                f"{few_shot_prompt_text}\n\n"
                f"TASK:\n"
                f"Write a brand-new, production-ready music generation prompt incorporating all user specifications above.\n\n"
                f"FORMAT YOUR RESPONSE EXACTLY AS:\n"
                f"- **New Song Title**: [Title]\n"
                f"- **Genre & Style**: [Genre details]\n"
                f"- **Language**: {language_option}\n"
                f"- **Main Character / Creature**: {creature_input}\n"
                f"- **Summary**: [2-3 sentence overview]\n"
                f"- **Production Prompt**:\n"
                f"```text\n"
                f"[A detailed prompt including structural layout, instruments, tempo/BPM, vocal characteristics, and sample lyrics/phrases in the specified language]\n"
                f"```\n"
            )

            models_to_try = [
                "gemini-3.6-flash",
                "gemini-3.8-flash",
                "gemini-3.1-pro-preview",
                "gemini-3.5-flash-lite"
            ]
            
            final_response = None
            used_model = None

            for model_name in models_to_try:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=rag_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            temperature=0.7,
                        ),
                    )
                    if response and response.text:
                        final_response = response.text
                        used_model = model_name
                        break
                except APIError:
                    time.sleep(0.5)
                    continue
                except Exception:
                    continue

            if final_response:
                st.markdown('<div class="studio-card">', unsafe_allow_html=True)
                st.markdown(f"### ✨ Generated Track Concept *(via {used_model})*")
                st.markdown(final_response)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ Cloud LLM endpoints are currently undergoing peak load. Displaying direct RAG library match.")
                
                st.markdown('<div class="studio-card">', unsafe_allow_html=True)
                st.markdown("### 🎯 RAG Library Fallback Match")
                st.markdown(f"* **Closest Match Title**: {top_match_row['Song']}")
                st.markdown(f"* **Genre & Style**: {top_match_row['Genre']}")
                st.markdown(f"* **Language**: {language_option}")
                st.markdown(f"* **Main Character / Creature**: {creature_input}")
                st.markdown(f"* **Summary**: {top_match_row['Summary']}")
                st.markdown("""
                <div class="prompt-bar-container">
                    <div class="prompt-bar-title">💡 Try this prompt</div>
                </div>
                """, unsafe_allow_html=True)
                st.code(top_match_row['Prompt'], language="text")
                st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 2: LIBRARY EXPLORER
# -----------------------------------------------------------------------------
with tab_library:
    st.markdown('<div class="studio-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">📊 Library Catalog & Search</div>', unsafe_allow_html=True)
    
    search_term = st.text_input("Filter library by song title, creature, or genre:", "", placeholder="Type to filter...")
    
    if search_term:
        filtered_df = df[
            df['Song'].str.contains(search_term, case=False) |
            df['Genre'].str.contains(search_term, case=False) |
            df['Main Creature'].str.contains(search_term, case=False)
        ]
    else:
        filtered_df = df

    st.dataframe(
        filtered_df[['Song', 'Genre', 'Main Creature', 'Summary']], 
        use_container_width=True, 
        height=400,
        column_config={
            "Song": st.column_config.TextColumn("🎵 Track Title", width="medium", help="Official title of the song"),
            "Genre": st.column_config.TextColumn("🎸 Genre & Style", width="medium"),
            "Main Creature": st.column_config.TextColumn("🐉 Character/Creature", width="medium"),
            "Summary": st.column_config.TextColumn("📝 Summary Narrative", width="large"),
        },
        hide_index=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 3: ARCHITECTURE & TECH STACK
# -----------------------------------------------------------------------------
with tab_architecture:
    st.markdown("""
    <div class="studio-card">
        <h2 style="margin-top:0; color:#0F172A !important; font-weight:800;">🏗️ System Architecture & Tech Stack</h2>
        <p style="color:#475569 !important; font-size:0.95rem; font-weight:500;">
            <b>Flow Music Playground Pro</b> is an experimental Retrieval-Augmented Generation (RAG) web application that transforms structured song concepts into production-ready AI music generation prompts. It features a real-time data layer, vector similarity search, and a <b>near zero-downtime multi-tier model fallback system</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="studio-card">', unsafe_allow_html=True)
    st.markdown("### 📐 System Architecture Diagram")
    
    def render_svg(svg_file_path):
        with open(svg_file_path, "r") as f:
            svg_content = f.read()
        b64_encoded = base64.b64encode(svg_content.encode("utf-8")).decode("utf-8")
        st.markdown(
            f'<div style="text-align: center; padding: 10px;">'
            f'<img src="data:image/svg+xml;base64,{b64_encoded}" style="max-width:100%; height:auto; border-radius:12px;" />'
            f'</div>', 
            unsafe_allow_html=True
        )

    try:
        render_svg("rag_architecture.svg")
    except Exception:
        try:
            st.image("rag_architecture.svg", use_column_width=True)
        except Exception:
            st.info("Place 'rag_architecture.svg' in your project root directory to render the SVG architecture diagram.")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="studio-card">', unsafe_allow_html=True)
    st.markdown("### 1. Tech Stack Breakdown")
    col_tech1, col_tech2 = st.columns(2)
    
    with col_tech1:
        st.markdown("""
        **Frontend & Application Layer**
        * **Streamlit (Python):** Reactive UI framework managing user interaction state.
        * **Custom CSS Engine:** Injects custom glassmorphism cards, responsive buttons, and modern Google typography (*Plus Jakarta Sans*).

        **Data Ingestion & Live Sync**
        * **Google Sheets API (`gspread`):** Connects directly to the cloud spreadsheet database.
        * **Pandas:** Data manipulation, cleaning, and local CSV fallback parsing from `flowmusic.csv`.
        * **Caching Strategy (`@st.cache_data`):** TTL 60-second cache prevents hitting Google Sheets API quotas while maintaining near-instant live synchronization.
        """)

    with col_tech2:
        st.markdown("""
        **Vector Search Engine**
        * **`sentence-transformers` (`BAAI/bge-small-en-v1.5`):** High-efficiency embedding model mapping track metadata to dense vector space.
        * **PyTorch (`torch`):** Computes normalized tensor operations and ranks matches via **Cosine Similarity**.

        **Generative AI & Resilient Fallback Engine**
        * **Google GenAI SDK (`google.genai`):** Communicates with Gemini model endpoints.
        * **4-Tier Model Fallback Sequence:** Programmatically failovers through valid production models (`gemini-3.6-flash` $\\rightarrow$ `gemini-3.8-flash` $\\rightarrow$ `gemini-3.1-pro-preview` $\\rightarrow$ `gemini-3.5-flash-lite`) on API errors with immediate short-circuit return.
        * **Zero-Downtime Local RAG Fallback:** Direct local RAG rendering if all cloud LLM endpoints are unavailable.
        """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="studio-card">', unsafe_allow_html=True)
    st.markdown("### 2. How Retrieval-Augmented Generation (RAG) & Fallbacks Work")
    
    st.markdown("""
    #### Step 1: Semantic Vector Indexing
    Metadata columns are concatenated into text passages:
    $$\\text{Passage} = \\text{Song Title} + \\text{Genre} + \\text{Main Creature} + \\text{Summary}$$

    The model transforms passages into 384-dimensional vector embeddings stored in memory as a tensor matrix.

    #### Step 2: Query Vectorization & Similarity Matching
    User inputs are vectorized and compared against dataset embeddings using Cosine Similarity:
    """)

    st.latex(r"\text{Similarity Score} = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}")

    st.markdown("""
    #### Step 3: Multi-Tier Generation & Zero-Downtime Fallback Pipeline
    1. **Few-Shot Context Assembly:** The top 3 dataset matches are injected into the LLM system prompt as reference style anchors.
    2. **Sequential Model Retry Chain (Short-Circuit):**
       $$\\text{gemini-3.6-flash} \\rightarrow \\text{gemini-3.8-flash} \\rightarrow \\text{gemini-3.1-pro-preview} \\rightarrow \\text{gemini-3.5-flash-lite}$$
       Execution returns immediately upon the first successful model response, preventing unnecessary downstream API calls.
    3. **Local RAG Direct Output (Fail-Safe):** If all cloud LLM endpoints fail or experience server outages, the pipeline bypasses LLM generation and directly formats the **top-ranked RAG library match** into a structured production prompt.
    """)
    st.markdown('</div>', unsafe_allow_html=True)