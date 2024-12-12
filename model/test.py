from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests
import torch
import torch.nn.functional as F

processor = ViTImageProcessor.from_pretrained('vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('vit-base-patch16-224-in21k')

def get_image_embedding_url(url):
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    feature_vector = outputs.last_hidden_state.mean(dim=1)
    return feature_vector

def get_image_embedding(path):
    image = Image.open(path)
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    feature_vector = outputs.last_hidden_state.mean(dim=1)
    return feature_vector

url1 = 'http://images.cocodataset.org/val2017/000000039769.jpg'
url2 = 'http://images.cocodataset.org/val2017/000000039769.jpg'
path1 = 'model/photos/photo1.jpg'
path2 = 'model/photos/photo2.jpg'
path3 = 'model/photos/photo3.jpg'

embedding1 = get_image_embedding_url(url1)
embedding2 = get_image_embedding_url(url2)
embedding3 = get_image_embedding(path1)
embedding4 = get_image_embedding(path2)
embedding5 = get_image_embedding(path3)

cosine_sim1 = F.cosine_similarity(embedding1, embedding2)
cosine_sim2 = F.cosine_similarity(embedding3, embedding4)
cosine_sim3 = F.cosine_similarity(embedding4, embedding5)

print("Cosine Similarity:", cosine_sim1.item())
print("Cosine Similarity:", cosine_sim2.item())
print("Cosine Similarity:", cosine_sim3.item())