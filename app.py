import requests 
import tensorflow as tf
import json
import os

def rest_request(img_batch, url=None):
    url = 'http://localhost:8501/v1/models/saved_model:predict'

    payload = json.dumps({'instances': img_batch.numpy().tolist()})
    response = requests.post(url, payload)
    return response


dir = input("Enter image's directory path: ")
print(os.listdir(dir))
dataset = tf.keras.utils.image_dataset_from_directory(dir,
                                                      label_mode=None,
                                                      image_size=(256, 256))

for img_batch in dataset:
    result = rest_request(img_batch)
    print(result.text)
