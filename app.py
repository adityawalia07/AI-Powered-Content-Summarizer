import os
import re
import requests
import streamlit as st
import certifi
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from youtube_transcript_api import YouTubeTranscriptApi

# --- SSL FIX ---
os.environ["SSL_CERT_FILE"] = certifi.where()

# --- Load API Key ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Streamlit UI ---
st.set_page_config(page_title="AI Summarizer", page_icon="🧠")
st.title("🧠 Groq-Powered YouTube & Web Summarizer")
url = st.text_input("🔗 Enter YouTube or Website URL")

# --- Utility Functions ---
def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None

def get_youtube_transcript(url):
    video_id = extract_video_id(url)
    if not video_id:
        return None
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry["text"] for entry in transcript])
    except:
        return None

def scrape_website_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        html = requests.get(url, headers=headers, timeout=10, verify=True).text
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(["script", "style"]): tag.decompose()
        return " ".join(soup.stripped_strings)[:10000]
    except Exception as e:
        st.error(f"Website loading failed: {e}")
        return None

def summarize_with_groq(text, model="llama3-8b-8192"):
    llm = ChatGroq(model=model, groq_api_key=GROQ_API_KEY)
    prompt = PromptTemplate(
        input_variables=["text"],
        template="""
Summarize this content in 300 words. Focus on clarity, key points, and logical flow.

Content:
{text}
        """
    )
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
    docs = [Document(page_content=text)]
    result = chain.invoke({"input_documents": docs})
    return result.get("output_text", "") if isinstance(result, dict) else str(result)

# --- Main Action ---
if st.button("📝 Generate Summary"):
    if not GROQ_API_KEY:
        st.error("⚠️ GROQ API key not found. Set it in your `.env` file.")
    elif not url:
        st.warning("📌 Please enter a valid URL.")
    else:
        with st.spinner("🔄 Extracting and Summarizing..."):
            # --- Get content ---
            if "youtube.com" in url or "youtu.be" in url:
                content = get_youtube_transcript(url)
                if not content:
                    st.warning("⚠️ No transcript found or video is restricted.")
            else:
                content = scrape_website_text(url)

            # --- Summarize ---
            if content:
                summary = summarize_with_groq(content)
                st.success("✅ Summary:")
                st.write(summary)
                with st.expander("📄 Source Content"):
                    st.write(content)
            else:
                st.error("❌ Failed to extract content.")
