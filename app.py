import requests 
import tensorflow as tf
import json
import pandas as pd

def get_classes():
    url = 'https://raw.githubusercontent.com/m-bashari-m/vehicle-color-recognition/main/logs/dataset-info.csv'
    df = pd.read_csv(url, index_col=0)

    return df['color']


def rest_request(img_batch, url=None):
    url = 'http://localhost:8501/v1/models/saved_model:predict'

    payload = json.dumps({'instances': img_batch.numpy().tolist()})
    response = requests.post(url, payload)
    return response


classes = get_classes().to_list()
dir = input("Enter image's directory path: ")
dataset = tf.keras.utils.image_dataset_from_directory(dir,
                                                      label_mode=None,
                                                      image_size=(256, 256),
                                                      shuffle=False)

files = [file.split('/')[-1] for file in dataset.file_paths]
prediction_per_class = tf.zeros(shape=len(classes), dtype=tf.int32).numpy()
files_index = 0

for img_batch in dataset:
    result = rest_request(img_batch)
    result = result.json()
    indexes = tf.argmax(result['predictions'], axis=1)
    for index in indexes:
        print(files[files_index], '\tpredicted as', classes[index])
        prediction_per_class[index] += 1
        files_index += 1

print('\n###### Total Predictions Info ######')
for i in range(len(classes)):
    print(prediction_per_class[i], 'image(s) predicted as', classes[i])
