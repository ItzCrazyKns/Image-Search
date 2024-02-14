import streamlit as st
import json
from multimodalClient import MultimodalEmbeddingsClient
from google.cloud import storage
from utils import load_config
import faiss
import json
import numpy as np

config = load_config()

def load_data(filepath):
    ids = []
    embeddings = []
    with open(filepath, 'r') as file:
        for line in file:
            item = json.loads(line)
            ids.append(item['id'])
            embeddings.append(item['embedding'])
    embeddings = np.array(embeddings).astype('float32')
    return ids, embeddings

def create_index(embeddings):
    index = faiss.IndexFlatL2(config["index_dimensions"])
    index.add(embeddings)
    return index

def search_similar_items(index, ids, query_embedding, k=5):
    distances, indices = index.search(query_embedding, k+1)
    similar_ids = [ids[i] for i in indices[0]]
    return similar_ids

filepath = config["filepath"]

ids, embeddings = load_data(filepath)
index = create_index(embeddings)

st.set_page_config(
    layout="wide",
    page_title="Image Search",
)

st.title("Image Search using Faiss")
st.subheader("Search for similar images using multimodal embeddings & Faiss")

client = MultimodalEmbeddingsClient(project=config["project"], location=config["location"], api_endpoint=config["api_endpoint"])

storage_client = storage.Client()
bucket = storage_client.bucket(config["storage_bucket"])

file = st.file_uploader("Pick a file", type=["jpg", "jpeg", "png"])

allResults=[]

def handle_search():
    if file is not None:
        image_content = file.read()
        embedding = client.generate_embeddings(image_content)

        embedding = np.array(embedding, dtype=np.float32)

        embedding = embedding.reshape(1, -1)

        similar_ids = search_similar_items(index, ids, embedding, k=config["search_k"])

        for id in similar_ids:
            allResults.append(bucket.blob(id).download_as_bytes())

        if len(allResults)>=1:
            st.write("Your query:")
            st.image(image_content, width=200)
            st.write("")
            st.write("Matching images:")
            st.image(allResults, width=200)
        else:
            st.write("No matching images found.")

st.button("Search", on_click=handle_search())
