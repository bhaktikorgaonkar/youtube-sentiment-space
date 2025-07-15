import os, re, requests, pandas as pd, streamlit as st
import google.generativeai as genai

# ── 1.  CONFIG  ──────────────────────────────────────────────────────────────
WORKER_URL   = "https://yt-comments.bhakti-korgaonkar.workers.dev"
GEMINI_API   = st.secrets["GEMINI_API_KEY"]    # add in Space secrets
genai.configure(api_key=GEMINI_API)
model = genai.GenerativeModel("gemini-pro")

# ── 2.  HELPER  ──────────────────────────────────────────────────────────────
YOUTUBE_RE = re.compile(r"v=([a-zA-Z0-9_-]{11})")

def extract_id(url: str) -> str | None:
    m = YOUTUBE_RE.search(url)
    return m.group(1) if m else None

def fetch_comments(video_id: str) -> list[str]:
    """Call the Cloudflare Worker and return a list of comment strings."""
    r = requests.get(f"{WORKER_URL}?videoId={video_id}", timeout=15)
    r.raise_for_status()
    return r.json().get("comments", [])

def classify(comments: list[str]) -> list[str]:
    """Batch‑classify comments with Gemini into P/N/Neg."""
    joined = "\n".join(f"- {c}" for c in comments)
    prompt = (
        "Classify each YouTube comment below as Positive, Neutral, or Negative. "
        "Return the labels in the same order, one per line.\n\n" + joined
    )
    resp = model.generate_content(prompt)
    labels = [l.strip() for l in resp.text.splitlines() if l.strip()]
    if len(labels) != len(comments):
        labels += ["Neutral"] * (len(comments) - len(labels))  # fallback
    return labels

# ── 3.  UI  ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="YouTube Sentiment (Gemini free)", layout="wide")
st.title("📊 YouTube Comment Sentiment Analyzer — Hugging Face Spaces")

url = st.text_input("Paste a YouTube video URL", placeholder="https://youtu.be/…")

if url:
    video_id = extract_id(url)
    if not video_id:
        st.error("Could not extract video ID — double‑check the link.")
        st.stop()

    with st.spinner("Fetching comments…"):
        try:
            comments = fetch_comments(video_id)
        except Exception as e:
            st.error(f"Error fetching comments: {e}")
            st.stop()

    if not comments:
        st.warning("No comments found or Worker returned empty list.")
        st.stop()

    with st.spinner("Scoring sentiment with Gemini…"):
        labels = classify(comments)

    df = pd.DataFrame({"Comment": comments, "Sentiment": labels})
    st.metric("Comments analysed", len(df))

    # Pie chart
    counts = df["Sentiment"].value_counts()
    st.pyplot(counts.plot.pie(autopct="%1.0f%%", ylabel="").figure)

    # Data table
    st.dataframe(df, use_container_width=True)
