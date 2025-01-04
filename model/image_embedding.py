from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import torch.nn.functional as F
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

processor = ViTImageProcessor.from_pretrained('vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('vit-base-patch16-224-in21k')

def get_image_embedding(path):
    image = Image.open(path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    feature_vector = outputs.last_hidden_state.mean(dim=1)
    return feature_vector

if __name__ == '__main__':
    #model accuracy testing
    path1 = 'model/test_photos/photo1.jpg'
    path2 = 'model/test_photos/photo2.jpg'
    path3 = 'model/test_photos/photo3.jpg'

    embedding3 = get_image_embedding(path1)
    embedding4 = get_image_embedding(path2)
    embedding5 = get_image_embedding(path3)

    cosine_sim2 = F.cosine_similarity(embedding3, embedding4)
    cosine_sim3 = F.cosine_similarity(embedding4, embedding5)

    print("Cosine similarity same person:", cosine_sim2.item())
    print("Cosine similarity different people:", cosine_sim3.item())