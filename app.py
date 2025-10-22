
import streamlit as st
import os
import pandas as pd
import pytesseract
from PIL import Image
import re
import io
import fitz
from groq import Groq
import json
import plotly.express as px
from wordcloud import WordCloud
from textblob import TextBlob
from fpdf import FPDF
import spacy
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import datetime
import functools
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import hashlib
import pickle
import feedparser
import base64


# -----------------------
# NLTK Setup
# -----------------------

# Download corpora only if missing (avoids re-downloading every run)
for corpus in ["wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{corpus}")
    except LookupError:
        nltk.download(corpus)

lemmatizer = WordNetLemmatizer()

@functools.lru_cache(maxsize=512)     #caches 512 calls, if out of spcace, removes LRU least recently used one
def get_synonyms_antonyms(word):
    """
    Returns synonyms and antonyms of a given word using NLTK's WordNet.
    Includes lemmatization and removes duplicates.
    """
    try:
        word = lemmatizer.lemmatize(word.lower())
        synonyms, antonyms = set(), set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                lemma_clean = lemma.name().replace("_", " ")
                synonyms.add(lemma_clean)
                if lemma.antonyms():
                    antonyms.add(lemma.antonyms()[0].name().replace("_", " "))
        return {"synonyms": sorted(list(synonyms)), "antonyms": sorted(list(antonyms))}
    except Exception:
        return {"synonyms": [], "antonyms": []}



# -----------------------
# Tesseract Config
# -----------------------

import shutil
tesseract_path = shutil.which("tesseract") or "/usr/bin/tesseract"    #usable on windows/mac/linux, fall back to default path
pytesseract.pytesseract.tesseract_cmd = tesseract_path


# -----------------------
# Groq API Configuration
# -----------------------
# Notes:
# 1. API key is loaded securely from an environment file (.env)
# 2. JSON is used instead of Pickle for safer caching
# 3. Cache folder is automatically created and ignored from Git
# 4. Errors are gracefully handled and logged

# -----------------------
# Groq API Configuration (Colab Secrets Compatible)
# -----------------------
import os
import json
import hashlib
import traceback
import streamlit as st
from groq import Groq
import asyncio
# -----------------------
# Load API Key from Environment (Colab Secrets or local env)
# -----------------------


# -----------------------
# Hardcoded API Key
# -----------------------
GROQ_API_KEY = "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"

if not GROQ_API_KEY:
    st.error("🚨 Missing GROQ_API_KEY.")
else:
    client = Groq(api_key=GROQ_API_KEY)

# -----------------------
# Cache Setup
# -----------------------
import os
import asyncio
import hashlib
import json
import base64
import traceback
import re
import streamlit as st
from redis.asyncio import Redis

# -----------------------
# Redis Connection (passwordless)
# -----------------------
redis = Redis(
    host="redis-12037.c1.us-central1-2.gce.redns.redis-cloud.com",
    port=12037,
    password="Llx6RhHJNY0dGabrTwgX2W3ogAffS5Ej",
    decode_responses=True
)

# -----------------------
# Cached Groq API Call Function (Redis)
# -----------------------
async def cached_groq_call_async(model_name, prompt_text, max_output_tokens=1000, ttl=86400):
    """
    Async Groq API call with Redis caching.
    """
    key = hashlib.sha256((model_name + prompt_text).encode()).hexdigest()

    # Check Redis cache
    cached_response = await redis.get(key)
    if cached_response:
        st.info("🧠 Loaded response from Redis cache.")
        return json.loads(cached_response)

    # API call wrapped in asyncio.to_thread to avoid blocking
    try:
        response = await asyncio.to_thread(
            lambda: client.chat.completions.create(
                messages=[{"role": "user", "content": prompt_text}],
                model=model_name,
                temperature=0.2,
                max_tokens=max_output_tokens
            )
        )

        # Extract response safely
        if response.choices and hasattr(response.choices[0].message, "content"):
            result = response.choices[0].message.content
        else:
            st.warning("⚠ Empty response from Groq API.")
            return None

        # Save to Redis with TTL
        await redis.set(key, json.dumps(result), ex=ttl)
        return result

    except Exception as e:
        st.error("🚨 Groq API call failed.")
        print(traceback.format_exc())
        return None

# -----------------------
# Async Whisper Transcription
# -----------------------
async def transcribe_audio_async(file_content, file_name):
    import asyncio, os, openai

    tmp_saved = False
    try:
        # Save file
        await asyncio.to_thread(lambda: open(file_name, "wb").write(file_content))
        tmp_saved = True

        # Transcribe
        response = await asyncio.to_thread(
            lambda: client.audio.transcriptions.create(
                file=open(file_name, "rb"),
                model="whisper-large-v3",
                prompt="Transcribe this audio clearly in English."
            )
        )
        return response.text

    except Exception as e:
        st.error(f"Audio transcription failed: {e}")
        return None

    finally:
        # Cleanup
        if tmp_saved and await asyncio.to_thread(os.path.exists, file_name):
            await asyncio.to_thread(os.remove, file_name)

# Llama Guard Safety Filter (Groq)
# -----------------------
import json
import re
import streamlit as st

async def moderate_content(text):
    """
    Uses meta-llama/llama-guard-4-12b to check if text is safe to process.

    Parameters:
        text (str): Text to be analyzed for safety.

    Returns:
        tuple: (is_safe: bool, reason: str)
    """
    try:
        # -----------------------------
        # Step 1: Truncate long input
        # -----------------------------
        truncated_text = text[:3000]

        # -----------------------------
        # Step 2: Strict prompt for JSON-only output
        # -----------------------------
        prompt = f"""
        You are a JSON-only moderation classifier.
        Analyze the following text for unsafe, harmful, or sensitive content.

        Respond with ONLY valid JSON in this exact format — no extra text:
        {{
          "is_safe": true or false,
          "reason": "short explanation"
        }}

        Text to analyze:
        {truncated_text}
        """

        # -----------------------------
        # Step 3: Call cached Groq API
        # -----------------------------
        response = await cached_groq_call_async(
            model_name="meta-llama/llama-guard-4-12b",
            prompt_text=prompt,
            max_output_tokens=200
        )

        # -----------------------------
        # Step 4: Handle empty response
        # -----------------------------
        if not response:
            return True, "Moderation unavailable: no response."

        # Optional debug (can remove later)
        # st.text_area("🧠 Raw moderation output", response, height=120)

        # -----------------------------
        # Step 5: Extract JSON object
        # -----------------------------
        match = re.search(r'(\{.*?\})', response, re.DOTALL)
        if not match:
            return True, "Fallback: no JSON returned."

        raw_json = match.group(1)

        # -----------------------------
        # Step 6: Clean malformed JSON
        # -----------------------------
        cleaned_json = (
            raw_json
            .replace("'", '"')           # Single → double quotes
            .replace("True", "true")     # Python → JSON
            .replace("False", "false")
            .replace("None", "null")
            .strip()
        )

        # -----------------------------
        # Step 7: Parse JSON safely
        # -----------------------------
        try:
            data = json.loads(cleaned_json)
            return data.get("is_safe", True), data.get("reason", "No reason given.")
        except json.JSONDecodeError:
            st.warning(f"⚠ Failed to parse moderation JSON even after cleaning: {cleaned_json}")
            return True, "Moderation fallback: invalid JSON."

    except Exception as e:
        st.warning(f"Moderation failed: {e}")
        return True, "Moderation unavailable."


# -----------------------
# spaCy model
# -----------------------
nlp = spacy.load("en_core_web_sm")

# -----------------------
# Domain Keywords
# -----------------------

DOMAIN_KEYWORDS = {
    "real_estate": {
        "property", "real estate", "housing", "apartment", "commercial",
        "residential", "land", "broker", "developer", "builder", "market",
        "sales", "rent", "lease", "mortgage", "valuation", "sq ft",
        "investment", "reit", "construction", "project", "flats"
    },
    "finance": {
        "stock", "investment", "bank", "loan", "equity", "market",
        "portfolio", "risk", "finance", "fund", "mutual fund",
        "trading", "bond", "dividend", "shares"
    },
    "technology": {
        "ai", "machine learning", "software", "hardware", "cloud",
        "innovation", "blockchain", "startups", "it", "programming",
        "gadgets", "tech", "algorithm"
    },
    "healthcare": {
        "hospital", "clinic", "medicine", "drug", "healthcare",
        "vaccine", "research", "disease", "pharma", "treatment",
        "diagnosis"
    },
    "sports": {
        "football", "soccer", "cricket", "tournament", "league",
        "athlete", "match", "score", "medal", "championship"
    }
}



# -----------------------
# Helper Functions
# -----------------------

# -----------------------
# Domain Classification
# -----------------------
def classify_text_domain(text):
    """
    Classifies text into a domain based on keyword frequency.
    Returns:
        domain_key (str) - e.g., 'real_estate', 'finance'
        score (int) - number of matching keywords
    """
    text_lower = text.lower()
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        count = sum(text_lower.count(keyword) for keyword in keywords)
        scores[domain] = count
    # Pick the domain with highest count
    domain_key = max(scores, key=scores.get)      #if tie, python returns 1st one with max
    return domain_key, scores[domain_key]

# -----------------------
# Text Cleaning
# -----------------------
def clean_entity_text(text):
    """Strips punctuation from the start/end and normalizes whitespace."""
    text = text.strip()
    text = re.sub(r'^[.,!?;:]+', '', text)
    text = re.sub(r'[.,!?;:]+$', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# -----------------------
# Document Text Extraction
# -----------------------
def extract_text_from_document(file_content, file_ext):       #content in bytes, and file extension
    """Extract text from images, PDFs, or plain text files."""
    all_pages = []       #store text per page, each el is a dictionary with page_no:content
    ext = file_ext.lower()
    try:
        if ext in [".jpg", ".jpeg", ".png"]:
            img = Image.open(io.BytesIO(file_content)) #io.BytesIO(file_content) creates a file-like object from raw bytes, so PIL can read it. Image.open(...) loads the image into memory.
            text = pytesseract.image_to_string(img, config='--oem 3 --psm 3')     #--oem 3: OCR Engine Mode → default + LSTM.--psm 3: Page Segmentation Mode → fully automatic page segmentation.
            all_pages.append({'page_num': 1, 'text': text.strip()})
        elif ext == ".pdf":
            doc = fitz.open(stream=file_content, filetype="pdf")
            try:
                for i in range(len(doc)):
                    page = doc.load_page(i)
                    text = page.get_text()
                    if not text.strip():
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                        #If no text was found (e.g., scanned PDF), fallback to OCR:page.get_pixmap(matrix=fitz.Matrix(2,2)) → render page as an image, 2x resolution.
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        #Image.frombytes(...) → convert pixmap to PIL Image.
                        text = pytesseract.image_to_string(img, config='--oem 3 --psm 3')
                        #pytesseract.image_to_string(...) → extract text from the image.
                    all_pages.append({'page_num': i+1, 'text': text.strip()})
            finally:
                doc.close()
        elif ext == ".txt":
            text = file_content.decode("utf-8")
            all_pages.append({'page_num': 1, 'text': text.strip()})
    except Exception:
        try:
            all_pages.append({'page_num': 1, 'text': file_content.decode("utf-8", errors="ignore")})   #try to decode errors
        except Exception:
            all_pages.append({'page_num': 1, 'text': ""})
    return all_pages

# -----------------------
# Text Chunking
# -----------------------
import re
import json
import asyncio
import streamlit as st
from typing import List, Tuple

# Assume cached_groq_call_async is already defined with Redis caching
# from previous code
# from your_module import cached_groq_call_async

# -----------------------
# Split text into articles
# -----------------------
def split_into_articles(text: str, min_length: int = 80) -> List[str]:
    """
    Split text into chunks based on double newlines.
    If no double newline exists but text is long enough, returns as single chunk.
    Ignores chunks smaller than min_length.
    """
    chunks = [chunk.strip() for chunk in re.split(r'\n{2,}', text) if len(chunk.strip()) > min_length]
    if not chunks and len(text.strip()) > min_length:
        return [text.strip()]
    return chunks

# -----------------------
# Model Selection
# -----------------------
def select_model(article_text: str, domain: str = "General") -> str:
    """
    Select LLM model based on article length.
    """
    length = len(article_text)
    if length > 1200:
        return "openai/gpt-oss-120b"
    elif length > 600:
        return "openai/gpt-oss-20b"
    else:
        return "llama-3.1-8b-instant"

# -----------------------
# Analyze Article (async-friendly)
# -----------------------
async def analyze_article_async(article_text: str, domain: str = None) -> dict:
    """
    Analyze article and return structured JSON including:
    - relevance
    - summary
    - headline
    - companies, publishers, date
    - categories
    - domain classification
    """
    # Step 1: Determine domain
    if domain:
        domain_display = domain.replace("_", " ").title()
        domain_key = domain
        score = 1.0
    else:
        # Assume classify_text_domain is defined elsewhere
        domain_key, score = classify_text_domain(article_text)
        domain_display = domain_key.replace("_", " ").title()

    # Step 2: Select model
    model_name = select_model(article_text, domain_display)

    # Step 3: Construct prompt
    prompt = f"""
You are a content intelligence assistant.
Analyze the following text in the context of the domain: '{domain_display}'.
Return JSON with:
- "is_relevant": true/false
- "relevance_reason": "...",
- "summary": "...",
- "headline": "...",
- "companies": ["..."],
- "publishers": ["..."],
- "publication_date": "...",
- "categories": ["..."]

---ARTICLE---
{article_text[:3000]}
---END---"""

    # Step 4: Send to Groq API with caching
    response = await cached_groq_call_async(model_name, prompt, max_output_tokens=400)

    # Step 5: Default result
    result = {
        "is_relevant": False,
        "relevance_reason": "Too little text or empty.",
        "summary_of_headline": "N/A",
        "headline": "N/A",
        "company_or_account": "N/A",
        "publishers": "N/A",
        "date_of_publishment": "N/A",
        "categories": ["N/A"],
        "domain": domain_display,
        "domain_score": score
    }

    # Step 6: Parse response JSON
    if response:
        match = re.search(r'(\{.*?\})', response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                result.update({
                    "is_relevant": data.get("is_relevant", False),
                    "relevance_reason": data.get("relevance_reason", "N/A"),
                    "summary_of_headline": data.get("summary", "N/A"),
                    "headline": data.get("headline", "N/A"),
                    "company_or_account": data.get("companies", "N/A"),
                    "publishers": data.get("publishers", "N/A"),
                    "date_of_publishment": data.get("publication_date", "N/A"),
                    "categories": data.get("categories", ["N/A"])
                })
            except json.JSONDecodeError:
                st.warning("⚠ Failed to parse JSON from Groq response.")

    return result

# -----------------------
# Helper: Synchronous wrapper (optional for non-async code)
# -----------------------
def analyze_article(article_text: str, domain: str = None) -> dict:
    """
    Wrapper to call async analyze_article_async from sync code.
    Uses nest_asyncio for notebooks or Streamlit.
    """
    import nest_asyncio
    nest_asyncio.apply()
    return asyncio.get_event_loop().run_until_complete(analyze_article_async(article_text, domain))

# -----------------------
# Sentiment Analysis
# -----------------------
def analyze_sentiment(text):
    """Return 'Positive', 'Negative', or 'Neutral' using TextBlob polarity."""
    try:
        polarity = TextBlob(text).sentiment.polarity    #-1 for negative, 0 for neutral, +1 for positive
        if polarity > 0.1:
            return "Positive"
        elif polarity < -0.1:
            return "Negative"
        else:
            return "Neutral"
    except Exception:
        return "Neutral"

# -----------------------
# Named Entity Extraction
# -----------------------
def extract_entities(text):   #chunk wise
    """Return companies, people, and locations from text using spaCy."""
    try:
        doc = nlp(text)
        return {
            "companies": list(set([ent.text for ent in doc.ents if ent.label_ == "ORG"])),
            # Iterates over all entities detected (doc.ents).
            # Checks if ent.label_ is "ORG" (organization).
            # Collects the text of the entity (e.g., "Google", "Microsoft").
            # Wraps in set() → removes duplicates.
            # Converts back to list() → easier to use in JSON.
            "people": list(set([ent.text for ent in doc.ents if ent.label_ == "PERSON"])),
            # Same as above but for "PERSON" entities.
            # Detects names like "Elon Musk", "Alice".
            "locations": list(set([ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]))  #GPE is geopolitical entities
            #Looks for geographical entities:
            #  "GPE" → Geopolitical entities (countries, cities, states).
            #  "LOC" → Other locations (mountains, rivers, landmarks).
        }
    except Exception:
        return {"companies": [], "people": [], "locations": []}  #if none found, spaCy doesn't load, empty text, bad characters- fallback so pipeline doesn't fail



# -----------------------
# Unicode-safe PDF Generation
# -----------------------
from fpdf import FPDF
import io
import os

def generate_pdf(df):
    """
    Generate a PDF report from a DataFrame of articles.
    Returns a BytesIO object for download or further processing.
    """
    pdf = FPDF()
    pdf.add_page()

    # -----------------------
    # Font Setup
    # -----------------------
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    if os.path.exists(font_path):
        pdf.add_font("DejaVu", "", font_path, uni=True)     #unicode friendly
        pdf.set_font("DejaVu", '', 16)   #the empty '' would hold B, I ,U if we wanted bold, italic, or underline
    else:
        pdf.set_font("Arial", '', 16)  # fallback if DejaVu not found

    # Title
    pdf.multi_cell(0, 10, "Content Intelligence Report", align="C") #0 cell width- fill page width, height of cell is 10(more the value, more the vertical space)
    pdf.ln(10) #Adds vertical space after the title (10 units).

    pdf.set_font(pdf.font_family, '', 12)  #use font defined before- deja vu, with size 12

    # -----------------------
    # Article Rows
    # -----------------------
    for idx, row in df.iterrows():
        pdf.multi_cell(0, 8, f"Headline: {row.get('headline','N/A')}")
        pdf.multi_cell(0, 8, f"Summary: {row.get('summary_of_headline','N/A')}")
        pdf.multi_cell(0, 8, f"Publisher: {row.get('publishers','N/A')}")
        pdf.multi_cell(0, 8, f"Date: {row.get('date_of_publishment','N/A')}")
        #not a table, just rows

        # Categories
        cats = row.get('categories', ["N/A"])   #for that article, is category just one, or a list, or N/A (not present)
        if isinstance(cats, list):   #if it is a list of categories
            cats_str = ', '.join([str(x) for x in cats])  #cconvert each to a string and join by ,
        else:
            cats_str = str(cats)   #if only 1 category, convert to string
        pdf.multi_cell(0, 8, f"Categories: {cats_str}")  #print in pdf

        # Sentiment
        pdf.multi_cell(0, 8, f"Sentiment: {row.get('sentiment','N/A')}")

        # Entities
        entities = row.get('entities', {})
        companies = ', '.join(entities.get('companies', []))   #covert dictionaries to strings
        people = ', '.join(entities.get('people', []))
        locations = ', '.join(entities.get('locations', []))
        pdf.multi_cell(0, 8, f"Entities: Companies-{companies}; People-{people}; Locations-{locations}")

        pdf.ln(5)  # spacing between articles results

    # -----------------------
    # Output as BytesIO
    # -----------------------
    pdf_bytes = pdf.output(dest='S').encode('latin1', 'ignore')
    #dest='S':'S' stands for “string” (not saving to a file). Returns the PDF as a string in memory, not as a physical file.
    #.encode('latin1', 'ignore')- Converts the PDF string into bytes using the Latin-1 character encoding. 'ignore' → any characters that cannot be encoded in Latin-1 are silently skipped.Result: pdf_bytes is a bytes object representing the PDF content.
    return io.BytesIO(pdf_bytes)  #Creates an in-memory binary stream (like a file, but in memory).
#Why use it:
#Stream can be passed to other libraries or returned in web apps without saving to disk.
#Example: You can directly provide it to Streamlit’s st.download_button to let users download the PDF.



# -----------------------
# Web Cleaning & Scraping
# -----------------------
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
import datetime
import streamlit as st

def clean_web(html_text: str) -> str:
    """
    Cleans raw HTML text to remove scripts, styles, and extra whitespace.
    Returns clean plain text.
    """
    try:
        soup = BeautifulSoup(html_text, "html.parser")
        # Remove script, style, noscript tags
        for s in soup(["script", "style", "noscript"]):
            s.extract()   #removes tags and everything inside it
        # Get text with spaces, collapse multiple spaces into one
        return re.sub(r'\s+', ' ', soup.get_text(separator=" ", strip=True)).strip()
        #produces a clean, human-readable text string without code, scripts, or extra spaces.
    except Exception:
        # Fallback: just remove extra whitespace
        return re.sub(r'\s+', ' ', html_text).strip()


def scrape_website(url: str, max_articles: int = 10) -> list[dict]:   #only 10 articles, each as a dict with the 4 fields, and finally a list of dicts
    """
    Scrape headlines, article text, and publication dates from a website.

    Args:
        url: Website URL to scrape.
        max_articles: Max number of articles to scrape.

    Returns:
        List of dictionaries with keys: 'headline', 'article_text', 'date_of_publishment'.
    """
    data = []     #empty list
    try:
        # Setup headless Chrome   for Selenium
        chrome_options = Options()
        chrome_options.add_argument("--headless")     #Runs Chrome invisibly (no GUI). Perfect for scripts and servers.
        chrome_options.add_argument("--no-sandbox")   #Avoids permission issues inside restricted environments.
        chrome_options.add_argument("--disable-dev-shm-usage")  #Prevents memory sharing problems in Docker/VMs.

        # Use 'with' context to ensure proper cleanup
        with webdriver.Chrome(options=chrome_options) as driver:  #webdriver.Chrome() starts a Chrome browser controlled by Selenium.Using a with block ensures that Chrome automatically closes when done — even if something crashes
            driver.get(url)  #driver.get(url) opens the webpage.
            # Wait for page to load basic content. This line waits up to 5 seconds for the <body> tag to appear — meaning the page has finished basic loading. If not loaded, Selenium raises exception
            WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

            # Get html of current loaded webpage and parse page with BeautifulSoup into a searchable structure(You can now easily do things like soup.find_all("p") or soup.find("article").)
            soup = BeautifulSoup(driver.page_source, "html.parser")
            # Find main article-like tags
            tags = soup.find_all(["article", "div", "section", "li"])

            # Limit number of articles
            for t in tags[:max_articles]:
                # Headline: first h1/h2/h3/a tag
                headline_tag = t.find(["h1", "h2", "h3", "a"])
                headline = headline_tag.get_text(strip=True) if headline_tag else "N/A"

                # Text: combine all paragraph tags
                raw_text = " ".join([p.get_text(strip=True) for p in t.find_all("p")])
                cleaned = clean_web(raw_text if raw_text else headline)  #call the clean function

                # Publication date: time or span tag with class containing date/time
                date_tag = t.find(["time", "span"], {"class": re.compile(r"date|time", re.I)})
                #Searches for <time> or <span> tags whose class names include "date" or "time".
                #Example: <span class="post-date">Oct 13, 2025</span>
                pub_date = date_tag.get_text(strip=True) if date_tag else "N/A"
                scraped_on = str(datetime.date.today())

                # Append structured result
                data.append({
                    "headline": headline,
                    "article_text": cleaned,
                    "date_of_publishment": pub_date,
                    "scraped_on": scraped_on
                })

    except Exception as e:
        st.error(f"Failed to scrape website: {e}")

    return data



# -----------------------
# RSS Feed Fetching
# -----------------------
import feedparser
import datetime
import streamlit as st

def fetch_rss_articles(rss_url, max_articles=10):
    """
    Fetches and cleans articles from an RSS feed.

    Args:
        rss_url (str): The RSS feed URL.
        max_articles (int): Maximum number of articles to fetch (default 10).

    Returns:
        list[dict]: A list of dictionaries, each containing:
            - headline
            - article_text
            - date_of_publishment
            - scraped_on
    """
    data = []
    try:
        # Parse the RSS feed
        feed = feedparser.parse(rss_url)
        #Parses rss feed fro the given url and returns an object (feed) that contains:
        #  feed.entries: list of articles/items.
        #  feed.bozo: boolean — True if feed has parsing errors.
        #  feed.bozo_exception: the actual parsing error (if any).


        # Check for malformed feeds, cuz even if malformed, feedparser still gives partial info
        if feed.bozo:
            st.warning(f"⚠ Warning: The RSS feed might be malformed: {feed.bozo_exception}")

        # Iterate through up to 'max_articles' entries
        for entry in feed.entries[:max_articles]:
            # Extract headline/title safely
            headline = getattr(entry, "title", "N/A")

            # Extract publication date if available, else mark as 'N/A'
            pub_date = getattr(entry, "published", "N/A")

            # Extract summary or content, clean it for HTML tags etc.
            summary = getattr(entry, "summary", "")
            cleaned = clean_web(summary) if summary else ""

            # Record the scrape date
            scraped_on = str(datetime.date.today())

            # Append to the collected data
            data.append({
                "headline": headline,
                "article_text": cleaned,
                "date_of_publishment": pub_date,
                "scraped_on": scraped_on
            })

    except Exception as e:
        st.error(f"❌ Failed to fetch RSS feed from {rss_url}: {e}")

    return data


# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Content Intelligence Platform 2.0", layout="wide")
st.markdown("""<style>html, body, [class*="stApp"] {background: linear-gradient(135deg, #1C1C2B, #3B0A45, #6A0572, #000000); color: #E7A6C0;} .stButton>button, .stDownloadButton>button {background: linear-gradient(135deg, #6A0572, #BB0A21, #3B0A45, #1C1C2B); color:#E7A6C0; border:none;}</style>""", unsafe_allow_html=True)
st.title("📰 Content Intelligence Platform 2.0")

# Sidebar
st.sidebar.header("Analysis Settings")
domain = st.sidebar.selectbox("Choose domain:", list(DOMAIN_KEYWORDS.keys())+["Custom"])
if domain=="Custom": domain = st.sidebar.text_input("Enter custom topic:", "Climate Change")
uploaded_files = st.file_uploader("Upload files:", type=["pdf","jpg","jpeg","png","txt","mp3","wav","m4a"], accept_multiple_files=True)
website_url = st.text_input("Or enter website URL:")
rss_url = st.text_input("Or enter RSS feed URL:")

all_data = []

import nest_asyncio
import asyncio
import pandas as pd
import io
from wordcloud import WordCloud
import plotly.express as px

nest_asyncio.apply()  # fixes "Event loop is closed" in Streamlit/Colab

if st.button("Analyze"):
    total_tasks = len(uploaded_files) + (1 if website_url else 0) + (1 if rss_url else 0)
    progress = st.progress(0)
    counter = 0
    all_data = []

    # --- Local Files ---
    for f in uploaded_files:
        name = f.name
        ext = os.path.splitext(name)[1].lower()
        content = f.getvalue()

        # --- Moderation Check ---
        is_safe, reason = asyncio.get_event_loop().run_until_complete(
            moderate_content(content)  # Pass file content for moderation
        )
        if not is_safe:
            st.warning(f"⚠ Skipped {f.name} due to unsafe content ({reason}).")
            continue

        # --- Audio Transcription ---
        if ext in [".mp3", ".wav", ".m4a"]:
            st.info(f"🎧 Transcribing audio: {f.name}")
            text_from_audio = asyncio.get_event_loop().run_until_complete(
                transcribe_audio_async(content, f.name)
            )
            if text_from_audio:
                pages = [{'page_num': 1, 'text': text_from_audio}]
            else:
                st.error(f"❌ Failed to transcribe {f.name}")
                continue
        else:
            # --- Extract text from documents (PDF, DOCX, etc.) ---
            pages = extract_text_from_document(content, ext)

        # --- Analyze each page ---
        for page in pages:
            for art in split_into_articles(page['text']):
                cleaned = re.sub(r'\s+', ' ', art).strip()
                if not cleaned:
                    continue

                sentiment = analyze_sentiment(cleaned)
                analysis_result = asyncio.get_event_loop().run_until_complete(
                    analyze_article_async(cleaned, domain)
                )
                entities = extract_entities(cleaned)

                all_data.append({
                    "file_name": name,
                    "headline": analysis_result.get("headline", "N/A"),
                    "summary_of_headline": analysis_result.get("summary_of_headline", "N/A"),
                    "publishers": analysis_result.get("publishers", "N/A"),
                    "date_of_publishment": analysis_result.get("date_of_publishment", "N/A"),
                    "categories": analysis_result.get("categories", ["N/A"]),
                    "sentiment": sentiment,
                    "entities": entities,
                    "is_relevant": analysis_result.get("is_relevant", False),
                    "relevance_reason": analysis_result.get("relevance_reason", "N/A"),
                    "article_text": cleaned[:500] + "..." if len(cleaned) > 500 else cleaned
                })

        counter += 1
        progress.progress(min(1.0, counter / total_tasks))

    # --- Website scraping ---
    if website_url:
        st.info(f"Fetching articles from: {website_url}")
        for art in scrape_website(website_url):
            cleaned = art["article_text"]
            sentiment = analyze_sentiment(cleaned)
            analysis_result = asyncio.get_event_loop().run_until_complete(
                analyze_article_async(cleaned, domain)
            )
            entities = extract_entities(cleaned)
            all_data.append({
                "file_name": "Live_Website",
                "headline": analysis_result.get("headline") or art.get("headline", "N/A"),
                "summary_of_headline": analysis_result.get("summary_of_headline", "N/A"),
                "publishers": analysis_result.get("publishers", "N/A"),
                "date_of_publishment": analysis_result.get("date_of_publishment") or art.get("date_of_publishment", "N/A"),
                "categories": analysis_result.get("categories", ["N/A"]),
                "sentiment": sentiment,
                "entities": entities,
                "is_relevant": analysis_result.get("is_relevant", False),
                "relevance_reason": analysis_result.get("relevance_reason", "N/A"),
                "article_text": cleaned[:500] + "..." if len(cleaned) > 500 else cleaned
            })
        counter += 1
        progress.progress(min(1.0, counter / total_tasks))

    # --- RSS feed ---
    if rss_url:
        st.info(f"Fetching articles from RSS feed: {rss_url}")
        for art in fetch_rss_articles(rss_url):
            cleaned = art["article_text"]
            sentiment = analyze_sentiment(cleaned)
            analysis_result = asyncio.get_event_loop().run_until_complete(
                analyze_article_async(cleaned, domain)
            )
            entities = extract_entities(cleaned)
            all_data.append({
                "file_name": "RSS_Feed",
                "headline": analysis_result.get("headline") or art.get("headline", "N/A"),
                "summary_of_headline": analysis_result.get("summary_of_headline", "N/A"),
                "publishers": analysis_result.get("publishers", "N/A"),
                "date_of_publishment": analysis_result.get("date_of_publishment") or art.get("date_of_publishment", "N/A"),
                "categories": analysis_result.get("categories", ["N/A"]),
                "sentiment": sentiment,
                "entities": entities,
                "is_relevant": analysis_result.get("is_relevant", False),
                "relevance_reason": analysis_result.get("relevance_reason", "N/A"),
                "article_text": cleaned[:500] + "..." if len(cleaned) > 500 else cleaned
            })
        counter += 1
        progress.progress(min(1.0, counter / total_tasks))

    progress.empty()

    # --- Display results ---
    if all_data:
        df = pd.DataFrame(all_data)
        relevant_df = df[df['is_relevant'] == True] if 'is_relevant' in df.columns else df
        if not relevant_df.empty:
            st.success(f"Found {len(relevant_df)} relevant articles.")
            st.dataframe(relevant_df)

            # WordCloud
            combined_text = " ".join(relevant_df['article_text'].tolist())
            if combined_text.strip():
                wc = WordCloud(width=800, height=400, background_color="white").generate(combined_text)
                st.subheader("📊 Word Cloud")
                st.image(wc.to_array())

            # Dashboard
            st.subheader("📈 Dashboard")
            tab1, tab2, tab3, tab4 = st.tabs(["Categories", "Publishers", "Timeline", "Sentiment"])
            with tab1:
                exploded = relevant_df.explode("categories")
                st.plotly_chart(px.histogram(exploded, x="categories", title="Articles by Category"), use_container_width=True)
            with tab2:
                st.plotly_chart(px.bar(relevant_df, x="publishers", title="Articles by Publisher"), use_container_width=True)
            with tab3:
                try:
                    relevant_df['__date_plot'] = pd.to_datetime(relevant_df['date_of_publishment'], errors='coerce')
                    st.plotly_chart(px.line(relevant_df.sort_values('__date_plot'), x="__date_plot", y=relevant_df.index, title="Timeline"), use_container_width=True)
                except Exception:
                    st.plotly_chart(px.line(relevant_df, x="date_of_publishment", y=relevant_df.index, title="Timeline"), use_container_width=True)
            with tab4:
                st.plotly_chart(px.histogram(relevant_df, x="sentiment", title="Sentiment Analysis"), use_container_width=True)

            # PDF/Excel/CSV/JSON export
            pdf_bytes = generate_pdf(relevant_df)
            st.download_button("📥 Download PDF Report", pdf_bytes, file_name="CIP_Report.pdf", mime="application/pdf")

            excel_buffer = io.BytesIO()
            relevant_df.to_excel(excel_buffer, index=False)
            excel_buffer.seek(0)
            st.download_button("📥 Download Excel Report", excel_buffer, file_name="CIP_Report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            csv_buffer = io.StringIO()
            relevant_df.to_csv(csv_buffer, index=False)
            st.download_button("📥 Download CSV Report", csv_buffer.getvalue(), file_name="CIP_Report.csv", mime="text/csv")

            json_buffer = io.StringIO()
            json_buffer.write(relevant_df.to_json(orient="records"))
            st.download_button("📥 Download JSON Report", json_buffer.getvalue(), file_name="CIP_Report.json", mime="application/json")
        else:
            st.warning("No relevant articles found.")
    else:
        st.warning("No data processed.")
