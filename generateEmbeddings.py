from multimodalClient import MultimodalEmbeddingsClient
from google.cloud import storage
from utils import load_config
import json

config = load_config()

client = MultimodalEmbeddingsClient(project=config["project"], location=config["location"], api_endpoint=config["api_endpoint"])

storage_client = storage.Client()

bucket = storage_client.bucket(config["storage_bucket"])

filepath = config["filepath"]

files = bucket.list_blobs()
for file in files:
    with file.open('rb') as image_file:
        image_content = image_file.read()
        embeddings = client.generate_embeddings(image_content)

        with open(filepath, "a") as f:
            name = file.name
            json_record = json.dumps({"id": name, "embedding": embeddings})
            f.write(json_record)
            f.write("\n")