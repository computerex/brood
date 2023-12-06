from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def compress_against_query(query, document, top_k=3):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap  = 200,
        length_function = len,
    )
    docs = text_splitter.create_documents([document])
    chunks = [d.page_content for d in docs]

    tuples = [(query, doc) for doc in chunks]
    if len(tuples) == 0:
        return chunks
    scores = cross_encoder.predict(tuples)
    # sort relevant array based on scores
    relevant = [x for _,x in sorted(zip(scores, chunks), reverse=True)]
    return relevant[:top_k]
