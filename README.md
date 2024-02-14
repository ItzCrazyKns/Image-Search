# Image Similarity Search
Search for similar images using [Vertex AI](https://cloud.google.com/vertex-ai)'s multimodal embeddings and [Faiss](https://faiss.ai/).

## How it works
1. It starts by getting all the images from the bucket specified in the `config.storage_bucket` variable.
2. It then loops through all of the files and generate embeddings for them using Vertex AI's multimodal embeddings.
3. The embeddings are then saved in a file which can be modified through the `config.filepath` variable.
4. The UI is made using Streamlit, when you upload an image in the Streamlit app the script converts the image into embeddings.
5. Then a Faiss index is created to perform similarity search over the previously saved embeddings and the embeddings created from the image
6. The result is then displayed in the Streamlit app

**Note**: The `config.index_dimensions` variable is set by default to 1408 because the Embeddings for Multimodal `multimodalembedding` model generates 1408 dimensions vectors.

## Installation
1. Clone the repository:
```bash
git clone https://github.com/ItzCrazyKns/Image-Search.git
```
2. Modify the `config.json` file.
3. Install required packages:
```bash
pip install -r requirements.txt
```
4. Generate embeddings for the images saved in the bucket:
```bash
python generateEmbeddings.py
```
5. Run the Streamlit app:
```bash
streamlit run main.py
```

**Note**: Make sure you're authenticated in the gcloud CLI. 

