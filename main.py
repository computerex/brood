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
from knowledge_sources.google import search, search_google
from utils.compression import compress_against_query

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

def get_api_creds_from_model(model):
    if model.startswith('gpt'):
        return openai_key, openai_base
    return deepinfra_key, deepinfra_base

def synthesize_answer_from_doc(query, doc, model, max_tokens=300, temperature=0, system_message='You are an AI Assistant.'):
    api_key, api_base = get_api_creds_from_model(model)
    PROMPT = \
    f"""Answer the question below using the provided information. If you are completely unsure of the answer, be as helpful as possible by informing the user of what you know, but don't make up unsupported facts.

BEGIN CONTEXT
{doc}
END CONTEXT

{query}

Answer:"""

    import openai
    openai.api_key = api_key
    openai.api_base = api_base
    tries = 0
    while tries < 3:
        try:
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": PROMPT}
            ]
            chat_completion = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                stream=False,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            output = chat_completion.choices[0].message.content
            return output
        except Exception as e:
            print(e)
            tries += 1

    return "I don't know. There is something wrong with my brain. Try again, and if I continue to be dumbfounded, try again later."

def rewrite_query(conversation, model, max_tokens=300, temperature=0):
    api_key, api_base = get_api_creds_from_model(model)
    PROMPT = \
    f"""Construct a search engine query from the conversation below. Ensure the final query is fully self contained. Ensure
    the final query is a suitable search engine query, which is no longer than 5-10 words. Use good search engine
    practices and phrasing.

BEGIN CONVERSATION
{conversation}
END CONVERSATION

self-contained query:"""

    import openai
    openai.api_key = api_key
    openai.api_base = api_base
    try:
        messages=[
            {"role": "system", "content": 'You are an AI Assistant. You are helping the user research by helping them google things.'},
            {"role": "user", "content": PROMPT}
        ]
        chat_completion = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            stream=False,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        output = chat_completion.choices[0].message.content
        return output
    except Exception as e:
        print(e)
        return conversation

def messages_to_string(messages, num_messages=10):
    input_str = ''
    for msg in messages[-num_messages:]:
        input_str += f"""{msg['role']}: {msg['content']}\n\n"""
    return input_str

def main():
    # query = "what is the easiest way of installing orbiter mods?"
    # answers = search(query, 3)
    # for link in answers:
    #     print(link)
    #     doc = answers[link]
    #     com = compress_against_query(query, doc, 3)
    #     print(com)
    # return
    airoboros = 'jondurbin/airoboros-l2-70b-gpt4-1.4.1'
    mistral = 'mistralai/Mistral-7B-Instruct-v0.1'
    gpt4turbo = 'gpt-4-1106-preview'
    gpt35turbo = 'gpt-3.5-turbo'
    yi34bchat = '01-ai/Yi-34B-Chat'

    messages=[
        llm_message("system", 'You are an Assistant.'),
    ]
    while True:
        query = input("User: ")
        messages.append(llm_message("user", query))
        inp = messages_to_string(messages[1:], num_messages=3)
        q_rewrite = rewrite_query(inp, model=yi34bchat, max_tokens=1000)
        if q_rewrite.startswith('"') and q_rewrite.endswith('"'):
            q_rewrite = q_rewrite[1:-1]
        print('Rewritten query: ' + q_rewrite)
        docs = search(q_rewrite.strip(), 5)
        print('docs: ')
        print(docs)
        filtered_docs = []
        for link in docs:
            doc = docs[link]
            com = compress_against_query(q_rewrite, doc, 1)
            filtered_docs.append(com)
        
        output = synthesize_answer_from_doc(inp, filtered_docs, model=yi34bchat, max_tokens=1000)
        # output = llm_chat(messages, model=gpt35turbo, max_tokens=3000, temperature=1)
        print('AI: ' + output)
        messages.append(llm_message("assistant", output))
    pass

if __name__ == "__main__":
    main()

