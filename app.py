import os
# preventing tensorflow verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import requests 
import tensorflow as tf
import json
import pandas as pd
import subprocess
import re

# get class names
def get_classes():
    url = 'https://raw.githubusercontent.com/m-bashari-m/vehicle-color-recognition/main/logs/dataset-info.csv'
    df = pd.read_csv(url, index_col=0)

    return df['color']

# img_batch: 4 dimentional tensor
# returns request.Response
def rest_request(img_batch):
    url = 'http://localhost:8501/v1/models/saved_model:predict'
    payload = json.dumps({'instances': img_batch.numpy().tolist()})
    response = requests.post(url, payload)
    return response

def print_predictions_info(preds_per_class, classes):
    n_sharps = 20
    print(n_sharps*'#', 'Total Predictions Info', n_sharps*'#')
    for i in range(len(classes)):
        print(classes[i],'   \t=>', preds_per_class[i], '   \t|', end='')
        
        if (i+1) % 2 == 0:
            print()

    print()
    

classes = get_classes().to_list()


dataset = tf.keras.utils.image_dataset_from_directory('/data',
                                                      label_mode=None,
                                                      image_size=(256, 256),
                                                      shuffle=False)

files = [file.split('/')[-1] for file in dataset.file_paths]
prediction_per_class = tf.zeros(shape=len(classes), dtype=tf.int32).numpy()
files_index = 0

for img_batch in dataset:
    result = rest_request(img_batch)
    result = result.json()
    print(result)
    indexes = tf.argmax(result['predictions'], axis=1)
    for index in indexes:
        print(files[files_index], '\tpredicted as', classes[index])
        prediction_per_class[index] += 1
        files_index += 1

print_predictions_info(prediction_per_class, classes)
