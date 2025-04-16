import streamlit as st
import traceback
import logging
import re
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set page configuration as the first Streamlit command
st.set_page_config(
    page_title="AI Content Summarizer", 
    page_icon="📝",
    layout="wide"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Comprehensive import handling
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from pytube import YouTube, exceptions as pytube_exceptions
    from langchain_groq import ChatGroq
    from langchain.prompts import PromptTemplate
    from langchain.docstore.document import Document
    from langchain.chains.summarize import load_summarize_chain
    from bs4 import BeautifulSoup
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.error("Please install required libraries:")
    st.code("""
    pip install youtube-transcript-api pytube langchain-groq requests beautifulsoup4 python-readability python-dotenv
    """)
    st.stop()

def extract_youtube_video_id(url: str) -> str:
    """
    Extract YouTube video ID from various URL formats
    """
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([^&]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([^?&]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([^?&]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return ""

def fetch_video_metadata(video_id: str):
    """
    Alternative method to fetch video metadata using YouTube's public data
    """
    try:
        # Try pytube first
        try:
            yt = YouTube(f"https://youtube.com/watch?v={video_id}")
            return {
                "title": yt.title,
                "author": yt.author,
                "length": yt.length,
                "thumbnail": yt.thumbnail_url
            }
        except Exception as pytube_error:
            logger.warning(f"Pytube failed: {pytube_error}")

        # Fallback to requests method
        try:
            url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                return {
                    "title": data.get('title', 'Untitled Video'),
                    "author": data.get('author_name', 'Unknown Channel'),
                    "length": 0,
                    "thumbnail": data.get('thumbnail_url')
                }
        except Exception as requests_error:
            logger.warning(f"Requests method failed: {requests_error}")

        # Ultimate fallback
        return {
            "title": "Video Title Unavailable",
            "author": "Channel Unavailable",
            "length": 0,
            "thumbnail": None
        }

    except Exception as e:
        logger.error(f"Metadata retrieval error: {e}")
        return {
            "title": "Video Title Unavailable",
            "author": "Channel Unavailable",
            "length": 0,
            "thumbnail": None
        }

def extract_youtube_transcript(video_id: str) -> str:
    """
    Extract transcript from a YouTube video with detailed error handling
    """
    try:
        # Fetch transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Combine transcript texts
        full_transcript = ' '.join([entry['text'] for entry in transcript])
        
        # Truncate to first 10000 characters
        return full_transcript[:10000]
    
    except Exception as transcript_error:
        st.error(f"Transcript Extraction Error: {transcript_error}")
        st.info("Possible reasons:")
        st.info("- No closed captions available")
        st.info("- Video might be age-restricted")
        st.info("- Transcript might be disabled")
        return ""

def extract_website_content(url: str) -> str:
    """
    Extract main content from a website using multiple methods
    """
    try:
        # Add headers to mimic browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Fetch webpage
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Use BeautifulSoup for initial parsing
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text from body
        body_text = soup.get_text(separator=' ', strip=True)
        
        # Clean up and truncate text
        cleaned_text = ' '.join(body_text.split())
        
        # Limit to 10000 characters
        return cleaned_text[:10000]
    
    except requests.RequestException as e:
        st.error(f"Website Extraction Error: {e}")
        st.info("Possible reasons:")
        st.info("- Invalid URL")
        st.info("- Website is not accessible")
        st.info("- Network connectivity issues")
        return ""
    except Exception as e:
        st.error(f"Unexpected error extracting website content: {e}")
        return ""

def langchain_summarize(
    text: str, 
    model: str = "llama3-8b-8192",
    additional_instructions: str = ""
) -> str:
    """
    Use LangChain to generate a summary with comprehensive error handling
    """
    try:
        # Validate inputs
        if not text:
            st.error("No text provided for summarization")
            return ""

        # Get API key from environment variable
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error("No GROQ API key found in environment variables")
            return ""

        # Initialize LLM
        llm = ChatGroq(
            model=model, 
            temperature=0.3, 
            groq_api_key=api_key
        )

        # Create a document
        docs = [Document(page_content=text)]

        # Prompt template with optional additional instructions
        prompt_template = f"""
        Provide a comprehensive summary of the following content:
        - Capture the main ideas, key points, and most important information
        - Organize the summary in a clear, logical manner
        - Aim for a concise summary that captures the essence of the content
        - Length should be around 300-400 words
        {additional_instructions}

        Content: {{text}}

        DETAILED SUMMARY:
        """
        prompt = PromptTemplate(
            template=prompt_template, 
            input_variables=["text"]
        )

        # Summarization chain
        chain = load_summarize_chain(
            llm, 
            chain_type="stuff", 
            prompt=prompt
        )

        # Generate summary
        output_summary = chain.invoke({"input_documents": docs})
        summary_text = output_summary.get('output_text', '') if isinstance(output_summary, dict) else str(output_summary)

        return summary_text

    except Exception as e:
        st.error(f"Summarization Error: {e}")
        logger.error(traceback.format_exc())
        return ""

def main():
    # Custom CSS for enhanced UI
    st.markdown("""
    <style>
    .stTextInput > div > div > input {
        color: white !important;
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    .stSelectbox > div > div > select {
        color: white !important;
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    .stRadio > div {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px;
        padding: 10px;
    }
    .stButton > button {
        background-color: #4CAF50 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        background-color: #45a049 !important;
        transform: scale(1.05) !important;
    }
    .stAlert {
        border-radius: 10px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    
    # Title with gradient
    st.markdown("""
    <h1 style="
        background: linear-gradient(90deg, #00DBDE 0%, #FC00FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 20px;
    ">🤖 AI-Powered Content Summarizer</h1>
    """, unsafe_allow_html=True)
    
    # Content URL Input with icon
    content_url = st.text_input(
        "📎 Enter YouTube Video or Website URL", 
        placeholder="https://www.youtube.com/watch?v=... or https://example.com"
    )
    
    # Content Type Selection
    content_type = st.radio(
        "📊 Select Content Type", 
        ["YouTube Video", "Website"], 
        horizontal=True
    )
    
    # Check for API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("⚠️ No GROQ API Key found in .env file. Please make sure to set the GROQ_API_KEY environment variable.")
    
    # Advanced Options Expander
    with st.expander("🚀 Advanced Summarization Options"):
        # Model selection dropdown
        available_models = [
            "llama3-8b-8192", 
            "gemma2-9b-it", 
            "qwen-qwq-32b"
        ]
        selected_model = st.selectbox(
            "💡 Choose LLM Model", 
            available_models, 
            index=0
        )
        
        additional_context = st.text_area(
            "Additional Context or Specific Instructions", 
            placeholder="Example: Focus on technical details or summarize for a specific audience..."
        )
    
    # Summarize Button
    if st.button("Generate Summary", use_container_width=True):
        # Validate inputs
        if not api_key:
            st.warning("📢 Please set your Groq API Key in the .env file")
            return
        
        if not content_url:
            st.warning("📢 Please enter a URL")
            return
        
        # Show loading spinner
        with st.spinner("🔍 Extracting content and generating summary..."):
            try:
                # Extract content based on type
                if content_type == "YouTube Video":
                    # Extract video ID
                    video_id = extract_youtube_video_id(content_url)
                    if not video_id:
                        st.error("❌ Invalid YouTube URL")
                        return

                    # Extract transcript from YouTube video
                    extracted_content = extract_youtube_transcript(video_id)
                    
                    # Retrieve video details safely
                    video_details = fetch_video_metadata(video_id)
                    
                    # Display metadata if available
                    if extracted_content:
                        # Display video details as a card
                        st.markdown(f"""
                        <div style="
                            background-color: rgba(255, 255, 255, 0.1);
                            border-radius: 10px;
                            padding: 15px;
                            margin-bottom: 15px;
                        ">
                        <h3 style="color: #4CAF50;">📺 {video_details['title']}</h3>
                        <p>📢 Channel: {video_details['author']}</p>
                        {'<p>⏱️ Length: ' + str(video_details['length']) + ' seconds</p>' if video_details['length'] > 0 else ''}
                        </div>
                        """, unsafe_allow_html=True)
                
                else:  # Website
                    # Extract website content
                    extracted_content = extract_website_content(content_url)
                    
                    # Display URL as a card
                    st.markdown(f"""
                    <div style="
                        background-color: rgba(255, 255, 255, 0.1);
                        border-radius: 10px;
                        padding: 15px;
                        margin-bottom: 15px;
                    ">
                    <h3 style="color: #4CAF50;">🌐 Website Content</h3>
                    <p>📎 URL: {content_url}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Check if content was extracted
                if not extracted_content:
                    st.error("❌ Could not extract content")
                    return
                
                # Generate AI summary
                summary = langchain_summarize(
                    text=extracted_content, 
                    model=selected_model,
                    additional_instructions=additional_context
                )
                
                # Display summary in a styled card
                if summary:
                    st.markdown("""
                    <div style="
                        background-color: rgba(255, 255, 255, 0.1);
                        border-radius: 10px;
                        padding: 20px;
                        border-left: 5px solid #4CAF50;
                    ">
                    <h3 style="color: #4CAF50;">✨ AI-Generated Summary</h3>
                    """, unsafe_allow_html=True)
                    st.write(summary)
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.error("❌ Failed to generate summary")
            
            except Exception as e:
                st.error(f"❌ An unexpected error occurred: {e}")
                logger.error(traceback.format_exc())

    # Footer
    st.markdown("""
    <div style="
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: rgba(0, 0, 0, 0.7);
        color: white;
        text-align: center;
        padding: 10px;
    ">
    Made with ❤️ by an AI Assistant | Powered by Groq & LangChain
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()