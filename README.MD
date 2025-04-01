# QnA ESG Reporter & Web-Age

## QnA ESG Reporter

### Q-n-A App to Explore ESG Reports using RAG-LLM

#### About
Q-n-A with ESG report using LLM is an end-to-end project based on the Retrieval-Augmented Generation (RAG) concept, where external information, in this case, Environmental-Social-Governance (ESG) reports, are provided to large language models (LLM).

#### Motivation
Global companies must produce a report detailing how they fulfill their responsibilities toward ESG, complying with regulations. The **ESG report** is crucial for gaining the trust of investors and stakeholders. However, these reports are often extensive, complex, and carefully worded, making manual extraction of relevant information tedious and time-consuming. This project addresses this challenge by providing users with concise responses along with referenced page numbers from the report against their queries.

#### How it Works
1. **User Input:** The front-end collects the year, company name, and user query.
2. **Data Retrieval:** The report is fetched from a GCP bucket/repository.
3. **Processing:** The report is processed to obtain text embeddings.
4. **Query Handling:** The LLM retrieves relevant text chunks and generates a concise response.
5. **Source Reference:** Corresponding page numbers from the metadata are included in the response.

The project is developed on **Google Cloud Platform (GCP)**. Reports are collected via web scraping and stored in a GCP Bucket. To render the Streamlit app from GCP, local tunnel and IPv4 utilities are used.

#### Screenshots
_(Add frontend screenshots here)_

---

## Web-Age

### AI-Powered Web Content Extraction and Summarization

#### Welcome to WEB-AGE
**WEB-AGE** is an advanced RAG Chatbot application designed to extract content from any URL, generate detailed summaries, and facilitate AI-powered interactions. Users can choose between **Closed Source (OpenAI)** and **Open Source (Ollama)** for flexible summarization and conversation.

#### Features
- **Website Extraction:** Crawls and extracts content from web pages.
- **Summarization:** Generates detailed summaries of extracted content.
- **Embeddings & Retrieval:** Uses FAISS for intelligent document retrieval.
- **Chatbot Interface:** Enables interaction with the extracted content through a conversational AI agent.
- **AI Model Flexibility:** Supports multiple AI models, including OpenAI and Ollama.

#### Screenshots
_(Add screenshots here)_

---

## File Structure
```
├── README.MD
├── config.py
├── conversational_ai_agent
│   ├── chat_history.pdf
│   ├── faiss_index-1
│   └── news_esg_agent.py
├── dashboard
│   └── esg_reporter.py
├── data
│   ├── SP50List.csv
│   └── SnP50_list_json
├── doc
├── gcs-bucket
│   ├── ESG_Report_files.csv
│   ├── SP50List.csv
│   └── reports
│       ├── 2024_AAPL.pdf
│       ├── 2024_AMZN.pdf
│       ├── 2024_GOOG.pdf
│       ├── 2024_GOOGL.pdf
│       ├── 2024_JPM.pdf
│       ├── 2024_MSFT.pdf
│       ├── 2024_NVDA.pdf
│       ├── 2024_UNH.pdf
│       ├── 2025_AAPL.pdf
│       ├── 2025_AMZN.pdf
│       ├── 2025_BAC.pdf
│       ├── 2025_CRM.pdf
│       ├── 2025_CVX.pdf
│       ├── 2025_GOOG.pdf
│       ├── 2025_GOOGL.pdf
│       ├── 2025_JNJ.pdf
│       ├── 2025_JPM.pdf
│       ├── 2025_MSFT.pdf
│       ├── 2025_NFLX.pdf
│       ├── 2025_NVDA.pdf
│       ├── 2025_ORCL.pdf
│       ├── 2025_PG.pdf
│       ├── 2025_UNH.pdf
│       ├── 2025_WFC.pdf
│       └── 2025_XOM.pdf
├── requirements.txt
├── utils
│   ├── companySymbol2Name.py
│   ├── scrap_SP50_name_symbol.py
│   └── scrap_pdf_report.py
└── vectordb
    ├── 863d8002-821b-4894-98aa-4d3b13a29234
    │   ├── data_level0.bin
    │   ├── header.bin
    │   ├── index_metadata.pickle
    │   ├── length.bin
    │   └── link_lists.bin
    └── chroma.sqlite3 # This main documentation file
```

---

## Code Explanation

### QnA ESG Reporter
- **app.py**: Implements a Streamlit-based front-end for user queries, processes reports, generates embeddings, and retrieves relevant content.
- **utils.py**: Handles interactions with the GCP bucket and data retrieval.

### Web-Age
- **web_scraper.py**: Extracts web content from given URLs.
- **summarizer.py**: Uses AI models to generate content summaries.
- **chatbot.py**: Implements the conversational interface for user interaction.
- **config.py**: Allows switching between OpenAI and Ollama models.



## Conclusion
Both **esg_reporter** and **news_esg_agent** leverage **RAG-based AI** to enhance information retrieval and summarization. These tools enable users to efficiently analyze ESG reports and interact with web content dynamically using AI-powered chatbots.

