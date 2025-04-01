import streamlit as st
import os
import asyncio
import nest_asyncio
import sys
import google.generativeai as genai
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from streamlit_chat import message as st_message
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from langchain_core.documents import Document
from langchain.document_loaders import TextLoader
from io import StringIO
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings



# Set Windows event loop policy
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

load_dotenv()
nest_asyncio.apply()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ---------------------------
# Initialize Session State Variables
# ---------------------------
if "url_submitted" not in st.session_state:
    st.session_state.url_submitted = False
if "extraction_done" not in st.session_state:
    st.session_state.extraction_done = False
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""
if "embedding_done" not in st.session_state:
    st.session_state.embedding_done = False
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "summary" not in st.session_state:
    st.session_state.summary = ""

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(layout="wide", page_title="WebSage")
st.title("Project WebSage")

page = st.sidebar.selectbox("Navigation", ["Home", "AI Engine", "Contact"])

if page == "Home":
    st.markdown("""
    ## Welcome to WEB-AGE
    **WEB-AGE** is a cutting-edge RAG Chatbot application that extracts content from any URL, generates detailed summaries, and enables AI-powered conversations.  
    This version uses **Google Generative AI** for summarization and retrieval-based question answering.

    **Features:**
    - **Website Extraction:** Crawl and extract web page content.
    - **Summarization:** Generate detailed summaries of the extracted content.
    - **Embeddings & Retrieval:** Create embeddings with FAISS for intelligent document retrieval.
    - **Chatbot Interface:** Interact with your content via a conversational agent.

    Get started by selecting **AI Engine** from the sidebar.
    """)

elif page == "AI Engine":
    with st.form("url_form"):
        url_input = st.text_input("Enter a URL to crawl:")
        submit_url = st.form_submit_button("Submit URL")
        if submit_url and url_input:
            st.session_state.url_submitted = True
            st.session_state.extraction_done = False
            st.session_state.embedding_done = False
            st.session_state.chat_history = []
            st.session_state.summary = ""

    if st.session_state.url_submitted:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.header("1. Website Extraction")
            if not st.session_state.extraction_done:
                with st.spinner("Extracting website..."):
                    async def simple_crawl(url):
                        crawler_run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
                        async with AsyncWebCrawler() as crawler:
                            result = await crawler.arun(url=url, config=crawler_run_config)
                            return result.markdown

                    extracted = asyncio.run(simple_crawl(url_input))
                    st.session_state.extracted_text = extracted
                    st.session_state.extraction_done = True
                st.success("Extraction complete!")

            preview = "\n".join([line for line in st.session_state.extracted_text.splitlines() if line.strip()][:5])
            st.text_area("Extracted Text Preview", preview, height=150)

            st.download_button(
                label="Download Extracted Text",
                data=st.session_state.extracted_text,
                file_name="extracted_text.txt",
                mime="text/plain",
            )

            st.markdown("---")
            st.subheader("Summarize Web Page")
            if st.button("Summarize Web Page"):
                with st.spinner("Summarizing..."):
                    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
                    response = model.generate_content(st.session_state.extracted_text)
                    st.session_state.summary = response.text
                st.success("Summarization complete!")

            if st.session_state.summary:
                st.subheader("Summarized Output")
                st.markdown(st.session_state.summary, unsafe_allow_html=False)

        with col2:
            st.header("2. Create Embeddings")
            if st.session_state.extraction_done and not st.session_state.embedding_done:
                if st.button("Create Embeddings"):
                    with st.spinner("Creating embeddings..."):
                        # Load extracted text as documents
                        documents = [Document(page_content=st.session_state.extracted_text)]
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                        texts = text_splitter.split_documents(documents)

                        model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                        vectorstore = FAISS.from_texts(texts=[chunk.page_content for chunk in texts], embedding=model)

                        vectorstore.save_local("faiss_index")

                        st.session_state.vectorstore = vectorstore
                        st.session_state.embedding_done = True
                    st.success("Vectors are created!")

        with col3:
            st.header("3. Chat with the Bot")
            if st.session_state.embedding_done:
                vectorstore = st.session_state.vectorstore
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

                user_input = st.text_input("Your Message:")
                if st.button("Send") and user_input:
                    context = "\n".join([doc.page_content for doc in retriever.get_relevant_documents(user_input)])
                    prompt = f"Context:\n{context}\n\nQuestion: {user_input}\n\nAnswer:"
                    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
                    response = model.generate_content(prompt)
                    bot_answer = response.text
                    st.session_state.chat_history.append({"user": user_input, "bot": bot_answer})

                if st.session_state.chat_history:
                    for chat in st.session_state.chat_history:
                        st_message(chat["user"], is_user=True)
                        st_message(chat["bot"], is_user=False)
            else:
                st.info("Please create embeddings to activate the chat.")
