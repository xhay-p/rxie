import os
from decouple import AutoConfig
config = AutoConfig(search_path='/Users/akshay/Work/github/genAIPG/.env')

os.environ["TAVILY_API_KEY"] = config("TAVILY_API_KEY")
os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")
os.environ["GOOGLE_API_KEY"] = config("GOOGLE_API_KEY")


import streamlit as st
from streamlit_autorefresh import st_autorefresh
import time
from datetime import datetime

st.set_page_config(
    page_title="arXiv Trend Analyser",
    page_icon="📑",
    layout="wide",
)

from pydantic import BaseModel, Field
from langchain_community.document_loaders import WebBaseLoader
import bs4
from typing import List, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Configuration
temp = 0.0
top_k = 10
max_tokens = 20000

MODEL_OPTIONS = {
    "gemini-3.1-flash-lite-preview": "google",
    "gemini-3-flash-preview": "google",
    "gpt-5-mini": "openai",
    "gpt-5.4-nano": "openai",
}

def load_doc_from_urls(urls: List[str], tags: List[str], tag_classes: List[str]):
    from bs4 import BeautifulSoup
    import requests

    docs = []
    for url in urls:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            dt_tags = soup.find_all("dt")  # Find all <dt> tags
            dd_tags = soup.find_all("dd")  # Find all <dd> tags

            for dt, dd in zip(dt_tags, dd_tags):
                # Extract the arXiv URL from the <dt> tag
                link = dt.find("a", href=True)
                if link and "/abs/" in link["href"]:
                    arxiv_url = f"https://arxiv.org{link['href']}"  # Construct the full URL

                    # Extract the title, authors, and subjects from the <dd> tag
                    title = dd.find("div", class_="list-title")
                    authors = dd.find("div", class_="list-authors")
                    subjects = dd.find("div", class_="list-subjects")

                    # Build a structured document
                    doc = {
                        "arxiv_url": arxiv_url,
                        "title": title.get_text(strip=True).replace("Title:", "") if title else "No title",
                        "authors": authors.get_text(strip=True) if authors else "No authors",
                        "subjects": subjects.get_text(strip=True).replace("Subjects:", "") if subjects else "No subjects",
                    }
                    docs.append(doc)
    return docs

# Run the autorefresh about every 6000 milliseconds (6 seconds)
st_autorefresh(interval=60000*60, key="arxiv_dataframerefresh")


def create_chain(model_name):
    trend_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI research analyst specializing in identifying **research trends and emerging themes in scientific literature**.
Your task is to **analyze the provided list of recently uploaded arXiv papers and extract meaningful research themes and trends**, particularly those relevant to my research interests.

**Non-negotiable behavior:** Deliver the full analysis in one reply. Do not comment on how large the input is, how long the answer might be, or that a full per-paper breakdown is "unwieldy." Do not offer to process only a sample or the first N papers. Do not ask the user to pick options (e.g. fast vs batched vs interactive). Start immediately with the structured topic sections and cover **every** paper from the input exactly once as specified."""),
        ("human", """
### My Research Interests

I am an AI researcher focused on the following areas:

* Generative AI, 
    - Large Language Models and Small Language Models
    - Multimodal Models
    - Retrieval-Augmented Generation (RAG) and Retrieval-Augmented Models
*Deep Learning and Machine Learning
    - Model Architectures, Optimizations and Training Techniques
* Reinforcement Learning
* Foundation Models
* Reasoning Models and AI Reasoning
* Agentic AI, Agentic Reasoning, and Agent Orchestration
* Single-cell and Genomics foundation models
* AI and Agentic AI for drug discovery

### Objectives

Analyze the provided arXiv papers and:

1. **Identify the major research themes and emerging trends** present in the papers.
2. **Group papers into meaningful topic clusters** based on their research focus.
3. **Highlight themes that align with my research interests**, but also include other significant emerging trends.
4. **Ensure that every paper from the input is included in exactly one topic cluster.**
5. Extract **important keywords** that represent each theme.

### Response rules (must follow)

* **No meta-preamble:** Do not thank the user, do not note that the list is "extremely large" or "hundreds–thousands of entries," and do not explain that you will only show a subset or a "representative sample."
* **No deferral:** Do not propose follow-up options for "full coverage" or ask which processing mode the user prefers. This single response **is** the full analysis.
* **Start with output:** Your first heading must be a topic section (e.g. `### Topic 1: ...`), not an introduction about how you will proceed.
* **If space is tight:** Keep per-paper lines concise (short titles, abbreviated author lists, one line of keywords) so every paper still appears—never drop papers or replace them with "sample only" messaging.

### Required Output Structure

Present the analysis in the following structured format:

---

### Topic 1: [Theme / Research Trend Name]

**1. Theme Summary**
Provide a concise explanation of the research theme and why it is emerging or important.

**2. Key Trends**
List 2–4 specific trends or innovations observed within this theme.

**3. Important Keywords**
List the most relevant technical keywords and concepts associated with this topic.

**4. Relevant Papers**

For each paper include:

* **Title:**
* **Authors:**
* **arXiv URL:**
* **Key Keywords from the Paper:**

---

### Topic 2: [Theme / Research Trend Name]

(Same structure as above)
         
and so on for all identified themes...

---

### Coverage Requirement

* **All papers from the input must be included in the analysis.**
* **Each paper must appear under one topic.**
* Topics should represent **coherent research themes rather than individual papers.**

### Additional Insights (Optional but encouraged)

At the end, include a section:

**Overall Research Trends**

* Identify **2–4 high-level trends across all papers**.
* Highlight **which themes are most aligned with my research interests**.

---

### Input Papers
Below are the scraped arXiv paper details:

{input}

---

### Context
Today's date: **{date}**""")
    ])

    provider = MODEL_OPTIONS[model_name]
    if provider == "openai":
        llm = ChatOpenAI(
            model=model_name,
        )
    else:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temp,
            max_tokens=max_tokens,
            timeout=None,
            max_retries=2,
        )

    trend_chain = trend_prompt | llm

    return trend_chain

def arxiv_daily_trend_analysis(model_name):
    docs = load_doc_from_urls(
        urls=['https://arxiv.org/list/cs.AI/recent?skip=0&show=2000',
            'https://arxiv.org/list/cs.CL/recent?skip=0&show=2000',
            'https://arxiv.org/list/cs.CV/recent?skip=0&show=2000',
            'https://arxiv.org/list/cs.IR/recent?skip=0&show=2000',
            'https://arxiv.org/list/q-bio/recent?skip=0&show=2000',],
        tags=['dt', 'a', 'dd', 'div'],
        tag_classes=['meta'] 
    )

    print(f"Number of documents: {len(docs)}")  

    trend_chain = create_chain(model_name)

    result = trend_chain.invoke({"input": docs, "date": time.strftime("%Y-%m-%d")})
    print(result.content)

    return result.content

def run():
    with st.sidebar:
        st.markdown("### Settings")
        selected_model = st.selectbox(
            "Model",
            options=list(MODEL_OPTIONS.keys()),
            index=0,
        )
        st.caption(f"**Provider:** `{MODEL_OPTIONS[selected_model]}`")
        st.caption(f"**Temperature:** `{temp}`")
        st.caption(f"**Max tokens:** `{max_tokens}`")
        st.divider()
        analyse = st.button("Analyse Trends", type="primary", use_container_width=True)
        st.divider()

    st.markdown(
        "<h1 style='text-align:center;'>📑 arXiv Trend Analyser</h1>"
        f"<p style='text-align:center;color:grey;'>{datetime.now().strftime('%A, %B %d, %Y')}</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    if analyse:
        with st.spinner("Fetching papers & analysing trends — this may take a minute..."):
            result = arxiv_daily_trend_analysis(selected_model)
        st.markdown(result)
    else:
        st.info("Select a model and click **Analyse Trends** in the sidebar to start.")

run()