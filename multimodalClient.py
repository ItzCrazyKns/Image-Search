from google.cloud import aiplatform
from google.protobuf import struct_pb2
import base64

class MultimodalEmbeddingsClient:
    def __init__ (self, project: str, location: str, api_endpoint: str):
        self.client = aiplatform.gapic.PredictionServiceClient(client_options={"api_endpoint": api_endpoint})
        self.project = project
        self.location = location
        self.api_endpoint = api_endpoint

    def generate_embeddings(self, image_bytes: bytes):
        instance = struct_pb2.Struct()
        encoded_content = base64.b64encode(image_bytes).decode('utf-8')
        image_struct = instance.fields['image'].struct_value
        image_struct.fields['bytesBase64Encoded'].string_value = encoded_content
    
        instances = [instance]
        
        endpoint = (f"projects/{self.project}/locations/{self.location}"
                    "/publishers/google/models/multimodalembedding@001")
        
        response = self.client.predict(endpoint=endpoint, instances=instances)

        image_embedding = None
    
        image_emb_value = response.predictions[0]['imageEmbedding']
        image_embedding = [v for v in image_emb_value]

        return image_embedding
