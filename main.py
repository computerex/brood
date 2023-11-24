import openai
import requests
import os
import re
import time
import numpy as np
import json
from bs4 import BeautifulSoup, NavigableString
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher
from thefuzz import process
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
# cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
from flask import Flask, request, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llm import llm_chat, message as llm_message

#cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
#mpnet = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
#minilm = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
nltk.download('punkt')
deepinfra_key  = json.loads(open('.env.json').read())['DEEP_INFRA_API_KEY']
deepinfra_base = "https://api.deepinfra.com/v1/openai"

openai_key = json.loads(open('.env.json').read())['OPEN_AI_API_KEY']
openai_base = openai.api_base

app = Flask(__name__)

def extract_json_from_text(text):
    # Define a regular expression pattern to match JSON
    pattern = r'{[^}]*}|\[[^\]]*\]'

    # Use regex to find all JSON parts in the text
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        # Extract the last matched JSON string
        json_str = matches[-1]

        try:
            # Attempt to parse the JSON
            json_data = json.loads(json_str)
            return json_data
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
    
    # Return None if no valid JSON was found
    return None

def rewrite_as_probe(text, api_key, api_base,  model="meta-llama/Llama-2-70b-chat-hf"):
    import openai
    openai.api_key = api_key
    openai.api_base = api_base

    probe_system = open('prompts/probe-voice-system.txt').read()
    probe_rewrite = open('prompts/probe-voice-rewrite.txt').read()

    PROMPT = probe_rewrite.format(text=text)

    tries = 0
    while tries < 3:
        try:
            messages=[
                {"role": "system", "content": f'{probe_system}'},
                {"role": "user", "content": PROMPT}
            ]
            chat_completion = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                stream=False,
                max_tokens=2500,
                temperature=0.3,
            )

            output = chat_completion.choices[0].message.content
            return output
        except Exception as e:
            print(e)
            tries += 1

def split_documents(documents, file_name_list, max_chunk_size=2000):
    """
    Split a list of documents into chunks of sentences.
    Each chunk comprises of sentences and the total character count is close to 'max_chunk_size'.

    Args:
    documents (list): List of input documents (strings).
    max_chunk_size (int): Maximum size (in characters) for each chunk.

    Returns:
    list: A flattened list of document chunks respecting sentence boundaries.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap  = 200,
        length_function = len,
    )
    docs = text_splitter.create_documents(documents)
    docs = [d.page_content for d in docs]
    return docs

def main():
    messages=[
        llm_message("system", 'You are an Assistant.'),
    ]
    airboros = 'jondurbin/airoboros-l2-70b-gpt4-1.4.1'
    llama = 'meta-llama/Llama-2-70b-chat-hf'
    mistral = 'mistralai/Mistral-7B-Instruct-v0.1'
    gpt35turbo = 'gpt-3.5-turbo'
    gpt4turbo = 'gpt-4-1106-preview'

    while True:
        messages.append({"role": "user", "content": input("User: ")})
        output = llm_chat(messages, model=gpt4turbo, max_tokens=2000)
        print('AI: ' + output)
        messages.append({"role": "assistant", "content": output})
    pass

if __name__ == "__main__":
    main()

