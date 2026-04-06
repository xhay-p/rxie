# rxie

**arXiv Trend Analyser** — a [Streamlit](https://streamlit.io/) app that pulls recent papers from selected arXiv category listings, then uses **Google Gemini** (via LangChain) to produce a structured analysis of research themes, topic clusters, and how they relate to configurable research interests.

## Requirements

- Python 3.11+ (matches the included [Dev Container](.devcontainer/devcontainer.json))
- A [Google AI Studio](https://aistudio.google.com/) API key with Gemini access

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Set your API key in the environment (the LangChain Google integration reads this by default):

```bash
export GOOGLE_API_KEY="your-key-here"
```

## Run

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`). In the sidebar, pick a Gemini model and click **Analyse Trends**.

## What it does

1. Fetches recent listing pages for **cs.AI**, **cs.CL**, **cs.CV**, **cs.IR**, and **q-bio** (up to thousands of entries per category, deduplicated by paper URL).
2. Sends the paper metadata (title, authors, subjects, URL) to the model with a fixed prompt that asks for themed clusters and coverage of every paper.
3. Renders the model’s markdown response in the app.

Analysis can take a while and uses the Gemini API; monitor usage in Google AI Studio.

## Dev Container

Opening the repo in a Dev Container installs dependencies and can start Streamlit automatically (see `postAttachCommand` in `.devcontainer/devcontainer.json`). Port **8501** is forwarded for the app.

## Configuration

Model IDs live in `MODEL_OPTIONS` in `app.py`. When Google renames or deprecates preview models, update those strings to match the current [Gemini API model names](https://ai.google.dev/gemini-api/docs/models).
