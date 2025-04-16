# 🤖 AI-Powered Content Summarizer
A smart and stylish Streamlit app that lets you summarize YouTube videos and website articles using powerful LLMs hosted on Groq, integrated with LangChain. Get concise, structured summaries in seconds — perfect for research, content review, or quick learning!


## ✨ Features
- 🔗 Accepts YouTube video or website URLs
- 🎙️ Extracts transcripts from YouTube videos (if captions available)
- 🧠 Summarizes content using Groq-hosted LLMs via LangChain
- 💡 Supports model selection (LLaMA 3, Gemma, Qwen, etc.)
- 📝 Add custom summarization instructions
- ⚙️ Clean, responsive, and beautiful UI
- ✅ Built-in error handling and fallback mechanisms

## 🛠️ Setup Instructions

pip install streamlit youtube-transcript-api pytube langchain-groq requests beautifulsoup4 python-dotenv
### Set Up Environment Variables
Create a .env file in the root directory:
Add your Groq API Key inside:

### Running the Application

```bash
streamlit run app.py
```

## 🧠 Supported LLMs
Choose from any of the following models via dropdown in the UI:

- llama3-8b-8192
- gemma2-9b-it
- qwen-qwq-32b

You can easily extend the list in the code if Groq adds more models.

## ⚠️ Limitations
- YouTube transcript extraction relies on publicly available captions.

- Website scraping may not work for JavaScript-heavy sites.

- Summarization length capped around 10,000 characters.

## 🧩 Technologies Used
- Streamlit
- LangChain
- Groq API
- Pytube
- YouTube Transcript API
- BeautifulSoup
- Python Dotenv

## 🙌 Credits
Made with ❤️ by Aditya
Powered by Groq + LangChain + Open Source Libraries
