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

# Import handling
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api.formatters import TextFormatter
    from langchain_groq import ChatGroq
    from langchain.prompts import PromptTemplate
    from langchain.docstore.document import Document
    from langchain.chains.summarize import load_summarize_chain
    from bs4 import BeautifulSoup
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.error("Please install required libraries:")
    st.code("""
    pip install youtube-transcript-api langchain-groq 
            requests beautifulsoup4 python-dotenv
    """)
    st.stop()

def extract_youtube_video_id(url: str) -> str:
    """
    Extract YouTube video ID from various URL formats
    """
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/v\/([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?m\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            if len(video_id) == 11:
                return video_id
    
    return ""

def fetch_video_metadata(video_id: str):
    """
    Fetch video metadata using YouTube's oEmbed API
    """
    try:
        url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        if response.status_code == 200:
            data = response.json()
            return {
                "title": data.get('title', 'Untitled Video'),
                "author": data.get('author_name', 'Unknown Channel'),
                "status": "success"
            }
        else:
            raise requests.RequestException(f"HTTP {response.status_code}")

    except Exception as e:
        logger.warning(f"oEmbed API failed: {e}")
        return {
            "title": "Video Title Unavailable",
            "author": "Channel Unavailable", 
            "status": "failed",
            "error": str(e)
        }

def extract_youtube_transcript(video_id: str) -> tuple:
    """
    Extract transcript from a YouTube video
    Returns tuple: (transcript_text, success_status, error_message)
    """
    try:
        if not video_id or len(video_id) != 11:
            return "", False, "Invalid video ID format"
        
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            transcript = None
            
            # Try to find a manual English transcript first
            try:
                transcript = transcript_list.find_manually_created_transcript(['en'])
            except:
                try:
                    transcript = transcript_list.find_generated_transcript(['en'])
                except:
                    try:
                        transcript = next(iter(transcript_list))
                    except:
                        return "", False, "No transcripts available for this video"
            
            transcript_data = transcript.fetch()
            
            formatter = TextFormatter()
            formatted_transcript = formatter.format_transcript(transcript_data)
            
            cleaned_transcript = ' '.join(formatted_transcript.split())
            
            # Increased character limit for better content retention
            # Most Groq models can handle 8k tokens, which is roughly 32k characters
            if len(cleaned_transcript) > 30000:
                cleaned_transcript = cleaned_transcript[:30000] + "..."
                logger.info(f"Transcript truncated to 30000 characters")
            
            return cleaned_transcript, True, "Transcript extracted successfully"
        
        except Exception as transcript_error:
            error_msg = str(transcript_error)
            
            if "Could not retrieve a transcript" in error_msg:
                return "", False, "No transcript available - video may not have captions enabled"
            elif "Too Many Requests" in error_msg:
                return "", False, "Rate limit exceeded - please try again later"
            elif "Video unavailable" in error_msg:
                return "", False, "Video is unavailable or private"
            elif "Transcript disabled" in error_msg:
                return "", False, "Transcripts are disabled for this video"
            else:
                return "", False, f"Transcript extraction failed: {error_msg}"
    
    except Exception as e:
        logger.error(f"Unexpected error in transcript extraction: {e}")
        return "", False, f"Unexpected error: {str(e)}"

def extract_website_content(url: str) -> tuple:
    """
    Extract main content from a website
    Returns tuple: (content_text, success_status, error_message)
    """
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' not in content_type:
            return "", False, f"Unsupported content type: {content_type}"
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'noscript']):
            element.decompose()
        
        # Try to find main content areas
        main_content = ""
        content_selectors = [
            'main', 'article', '[role="main"]', '.content', '#content', 
            '.post-content', '.entry-content', '.article-content'
        ]
        
        for selector in content_selectors:
            content_elements = soup.select(selector)
            if content_elements:
                main_content = ' '.join([elem.get_text(separator=' ', strip=True) for elem in content_elements])
                break
        
        # If no main content found, extract from body
        if not main_content:
            body = soup.find('body')
            if body:
                main_content = body.get_text(separator=' ', strip=True)
            else:
                main_content = soup.get_text(separator=' ', strip=True)
        
        cleaned_text = ' '.join(main_content.split())
        
        if len(cleaned_text) < 100:
            return "", False, "Insufficient content extracted from website"
        
        # Increased character limit for better content retention
        if len(cleaned_text) > 30000:
            cleaned_text = cleaned_text[:30000] + "..."
            logger.info(f"Website content truncated to 30000 characters")
        
        return cleaned_text, True, "Website content extracted successfully"
    
    except requests.exceptions.Timeout:
        return "", False, "Request timeout - website took too long to respond"
    except requests.exceptions.ConnectionError:
        return "", False, "Connection error - unable to reach website"
    except requests.exceptions.HTTPError as e:
        return "", False, f"HTTP error {e.response.status_code}: {e.response.reason}"
    except requests.exceptions.RequestException as e:
        return "", False, f"Request error: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error extracting website content: {e}")
        return "", False, f"Unexpected error: {str(e)}"

def get_model_limits(model: str) -> dict:
    """
    Get model-specific limits for optimal performance
    """
    model_configs = {
        "llama3-8b-8192": {
            "max_tokens": 8192,
            "max_input_chars": 32000,
            "recommended_summary_tokens": 1500
        },
        "gemma2-9b-it": {
            "max_tokens": 8192,
            "max_input_chars": 32000,
            "recommended_summary_tokens": 1500
        },
        "qwen-qwq-32b": {
            "max_tokens": 32768,
            "max_input_chars": 120000,
            "recommended_summary_tokens": 2000
        }
    }
    
    return model_configs.get(model, model_configs["llama3-8b-8192"])

def langchain_summarize(
    text: str, 
    model: str = "llama3-8b-8192",
    additional_instructions: str = ""
) -> tuple:
    """
    Use LangChain to generate a summary with model-specific optimizations
    Returns tuple: (summary_text, success_status, error_message)
    """
    try:
        if not text or len(text.strip()) < 50:
            return "", False, "Text too short for meaningful summarization"

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return "", False, "GROQ API key not found in environment variables"

        # Get model-specific configuration
        model_config = get_model_limits(model)
        
        # Truncate input if it exceeds model limits
        if len(text) > model_config["max_input_chars"]:
            text = text[:model_config["max_input_chars"]] + "..."
            logger.info(f"Input truncated to {model_config['max_input_chars']} characters for {model}")

        try:
            llm = ChatGroq(
                model=model, 
                temperature=0.3, 
                groq_api_key=api_key,
                max_tokens=model_config["recommended_summary_tokens"],
                timeout=90  # Increased timeout for longer content
            )
        except Exception as llm_error:
            return "", False, f"Failed to initialize LLM: {str(llm_error)}"

        docs = [Document(page_content=text)]

        base_prompt = """
        Provide a comprehensive and well-structured summary of the following content:

        **Instructions:**
        - Identify and highlight the main themes and key points
        - Organize information in a logical, easy-to-follow structure
        - Include important details, statistics, or examples when relevant
        - Maintain the original tone and context
        - Aim for a detailed summary that captures the essence of the content
        - Use clear, concise language
        - Structure the summary with appropriate headings or sections if the content is complex
        """
        
        if additional_instructions:
            base_prompt += f"\n\n**Additional Requirements:**\n{additional_instructions}"
        
        base_prompt += "\n\n**Content to Summarize:**\n{text}\n\n**SUMMARY:**"

        prompt = PromptTemplate(
            template=base_prompt, 
            input_variables=["text"]
        )

        try:
            chain = load_summarize_chain(
                llm, 
                chain_type="stuff", 
                prompt=prompt,
                verbose=False
            )
        except Exception as chain_error:
            return "", False, f"Failed to create summarization chain: {str(chain_error)}"

        try:
            result = chain.invoke({"input_documents": docs})
            
            if isinstance(result, dict):
                summary_text = result.get('output_text', '')
            else:
                summary_text = str(result)
            
            if not summary_text or len(summary_text.strip()) < 20:
                return "", False, "Generated summary is too short or empty"
            
            return summary_text.strip(), True, "Summary generated successfully"
            
        except Exception as generation_error:
            error_msg = str(generation_error)
            if "rate limit" in error_msg.lower():
                return "", False, "API rate limit exceeded - please try again later"
            elif "authentication" in error_msg.lower():
                return "", False, "API authentication failed - check your API key"
            elif "timeout" in error_msg.lower():
                return "", False, "Request timeout - please try again"
            else:
                return "", False, f"Summary generation failed: {error_msg}"

    except Exception as e:
        logger.error(f"Unexpected summarization error: {e}")
        return "", False, f"Unexpected error during summarization: {str(e)}"

def main():
    # Custom CSS for enhanced UI
    st.markdown("""
    <style>
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
    .error-card {
        background-color: rgba(255, 0, 0, 0.1);
        border-left: 5px solid #ff4444;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success-card {
        background-color: rgba(0, 255, 0, 0.1);
        border-left: 5px solid #4CAF50;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Title
    st.markdown("""
    <h1 style="
        background: linear-gradient(90deg, #00DBDE 0%, #FC00FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 20px;
    ">🤖 AI-Powered Content Summarizer</h1>
    """, unsafe_allow_html=True)
    
    # Content URL Input
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
    
    # Advanced Options
    with st.expander("🚀 Advanced Summarization Options"):
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
        
        # Display model capabilities
        model_info = get_model_limits(selected_model)
        st.info(f"📊 Model: {selected_model} | Max Input: ~{model_info['max_input_chars']:,} chars | Max Output: {model_info['recommended_summary_tokens']} tokens")
        
        additional_context = st.text_area(
            "Additional Context or Specific Instructions", 
            placeholder="Example: Focus on technical details, summarize for a specific audience, highlight key statistics..."
        )
    
    # Summarize Button
    if st.button("Generate Summary", use_container_width=True):
        if not api_key:
            st.error("📢 Please set your Groq API Key in the .env file")
            return
        
        if not content_url:
            st.warning("📢 Please enter a URL")
            return
        
        # Show loading spinner
        with st.spinner("🔍 Extracting content and generating summary..."):
            try:
                extracted_content = ""
                extraction_success = False
                extraction_error = ""
                
                if content_type == "YouTube Video":
                    video_id = extract_youtube_video_id(content_url)
                    if not video_id:
                        st.error("❌ Invalid YouTube URL format")
                        return

                    extracted_content, extraction_success, extraction_error = extract_youtube_transcript(video_id)
                    video_details = fetch_video_metadata(video_id)
                    
                    # Display video metadata
                    st.markdown(f"""
                    <div class="{'success-card' if video_details['status'] == 'success' else 'error-card'}">
                    <h3>📺 {video_details['title']}</h3>
                    <p>📢 Channel: {video_details['author']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if extraction_success:
                        st.success(f"✅ {extraction_error}")
                        st.info(f"📝 Transcript length: {len(extracted_content):,} characters")
                    else:
                        st.error(f"❌ Transcript extraction failed: {extraction_error}")
                        return
                
                else:  # Website
                    extracted_content, extraction_success, extraction_error = extract_website_content(content_url)
                    
                    st.markdown(f"""
                    <div class="{'success-card' if extraction_success else 'error-card'}">
                    <h3>🌐 Website Content</h3>
                    <p>📎 URL: {content_url}</p>
                    <p>Status: {'✅ Content extracted successfully' if extraction_success else '❌ ' + extraction_error}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if extraction_success:
                        st.info(f"📝 Content length: {len(extracted_content):,} characters")
                    else:
                        return
                
                # Generate AI summary
                summary, summary_success, summary_error = langchain_summarize(
                    text=extracted_content, 
                    model=selected_model,
                    additional_instructions=additional_context
                )
                
                if summary_success:
                    st.markdown("""
                    <div style="
                        background-color: rgba(255, 255, 255, 0.1);
                        border-radius: 10px;
                        padding: 20px;
                        border-left: 5px solid #4CAF50;
                        margin-top: 20px;
                    ">
                    <h3 style="color: #4CAF50;">✨ AI-Generated Summary</h3>
                    """, unsafe_allow_html=True)
                    st.write(summary)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.info(f"📊 Summary length: {len(summary):,} characters | Model used: {selected_model}")
                else:
                    st.error(f"❌ Summary generation failed: {summary_error}")
            
            except Exception as e:
                st.error(f"❌ An unexpected error occurred: {e}")
                logger.error(traceback.format_exc())
                with st.expander("🔍 Error Details (for debugging)"):
                    st.code(traceback.format_exc())

    # Footer
    st.markdown("""
    <div style="
        margin-top: 50px;
        text-align: center;
        padding: 20px;
        color: rgba(255, 255, 255, 0.7);
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    ">
    Made with ❤️ using Streamlit | Powered by Groq & LangChain
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()