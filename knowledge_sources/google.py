from googleapiclient.discovery import build
from selenium import webdriver 
from selenium.webdriver.chrome.options import Options 
from bs4 import BeautifulSoup
import re
import requests
import os
import json

env = json.loads(open('.env.json').read())
# apply env to os.environ
for key in env:
    os.environ[key] = env[key]

api_key = os.environ['GOOGLE_CLOUD_API_KEY']
cse_id = os.environ['GOOGLE_CLOUD_CSE_ID']

def search_google(search_phrase, top_k):
    # Build a service object for interacting with the API
    service = build("customsearch", "v1", developerKey=api_key)

    # Execute the search
    res = service.cse().list(q=search_phrase, cx=cse_id, num=top_k).execute()

    # Check if 'items' is in the results and return results if present
    if 'items' in res:
        return res['items'][:top_k]
    else:
        return []

def fetch_page(url):
    try:
        # instance of Options class allows 
        # us to configure Headless Chrome 
        options = Options() 

        # this parameter tells Chrome that 
        # it should be run without UI (Headless) 
        options.headless = True
        options.add_argument("--headless")
        # initializing webdriver for Chrome with our options 
        driver = webdriver.Chrome(options=options) 
        driver.set_page_load_timeout(10)

        # getting GeekForGeeks webpage 
        driver.get(url) 
        page_content = driver.page_source
        # close browser after our manipulations 
        driver.close()
        return page_content
    except Exception as e:
        return ""

def fetch_page_fast(url):
    try:
        # Send a GET request to the URL
        response = requests.get(url, timeout=5)

        # Return the raw HTML
        return response.text
    except Exception as e:
        return ""
    
def is_readable(text, threshold=0.7):
    # Count alphanumeric characters and whitespace
    alnum_count = sum(c.isalnum() or c.isspace() for c in text)
    
    # Calculate the ratio of alphanumeric characters to the total length of the string
    alnum_ratio = alnum_count / max(len(text), 1)
    
    # Return True only if the ratio exceeds the threshold
    return alnum_ratio > threshold

def extract_human_readable_text(raw_html):
    # Initialize BeautifulSoup
    soup = BeautifulSoup(raw_html, 'html.parser')

    # Remove script and style elements
    for script_or_style in soup(["script", "style"]):
        script_or_style.extract()

    # Get text
    text = soup.get_text()

    # Break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())

    # Use a list to collect the clean text chunks
    clean_chunks = []

    for line in lines:
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for phrase in line.split("  "))
        
        for chunk in chunks:
            # If chunk passes the readability test, append it to the clean_chunks list
            if is_readable(chunk):
                clean_chunks.append(chunk)

    # Combine clean chunks into the full text, separating with a space
    text = ' '.join(clean_chunks)

    # Further cleaning to remove any additional non-human-readable text
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]+', '', text)    # remove non-printable characters

    return text

def get_text_from_url(url):
    text = fetch_page(url)
    return extract_human_readable_text(text)

def search(search_phrase, top_k):
    results = search_google(search_phrase, top_k)
    urls = [result['link'] for result in results]
    texts = [get_text_from_url(url) for url in urls]
    return dict(zip(urls, texts))
