import os
import requests 
import json
import numpy as np
from PIL import Image
from flask import Flask, render_template, jsonify

app = Flask(__name__)

CLASSES = np.array(["beige", "black", "blue",
                    "brown", "cream", "crimson",
                    "gold", "green", "grey",
                    "navy-blue", "ornage", "red",
                    "silver", "titanium", "white",
                    "yellow"])

@app.route("/")
def home():
    return render_template("index.html")

'''
img_batch: 4 dimentional array
returns request.Response
'''
def rest_request(img_batch: np.ndarray, url: str):
    payload = json.dumps({'instances': img_batch.tolist()})
    response = requests.post(url, payload)
    
    return response

'''
returns:
   images: All images in directory in 4 dimentions which are normalized
   files: All file names corresponding to images. It uses for response
'''
def get_dataset(img_size: tuple, dir='/data'): 
    try:
        files = os.listdir(dir)
        paths = [os.path.join(dir, file) for file in files]
    except Exception:
        raise Exception('Data not found. Make sure to specify accurate data path and re-run the container.')
    else:
        if len(files) == 0:
            raise Exception('There is no image in this directory')

    images = np.array([]).reshape(-1, *img_size, 3)
    
    for i, path in enumerate(paths) :
        # Omits files with wrong format
        try:
            image = Image.open(path).resize(img_size)
            image_arr = np.array(image, dtype=np.float64) / 255.
            image_arr = np.expand_dims(image_arr, 0)
            images = np.concatenate([images, image_arr], axis=0)
        except:     
            files.pop(i)

    return images, files

'''
params:
   dataset: 4 dimentional array which includes normalized images
   fiels: List of file names
'''
def get_prediction(dataset: np.ndarray, files: list):
    url = 'http://tf-serving:8501/v1/models/vcor:predict'
    print('Request has been sent to', url)
    print('Waiting for response...')

    result = rest_request(dataset, url)
    result = result.json()
    
    try:
        indexes = np.argmax(result['predictions'], axis=1)
    except:
        raise Exception(result['error'])

    result_dict = dict(zip(files, CLASSES[indexes]))
    return jsonify(result_dict)
    

@app.route('/predict', methods=['GET'])
def predict():
    IMG_SIZE = (256, 256)
    try:
        dataset, files = get_dataset(IMG_SIZE)
        response = get_prediction(dataset, files)
        print("Prediction results are available at localhost:8080/predict")
        return response, 200

    except Exception as ex:
        error = {'error': str(ex)}
        print("Operation failed. You can find the error message in localhost:8080/predict")
        return jsonify(error), 400

if __name__ == '__main__':
    app.run(port=8080, host="0.0.0.0")
