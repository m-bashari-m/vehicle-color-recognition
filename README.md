# Vehicle Color Recognition
A deep learning model for car color classification with 16 classes.


## Data
This model is trianed by [Shenasa.ia](https://shenasa-ai.ir/) cars dataset (which is private) and [VCoR](https://www.kaggle.com/datasets/landrykezebou/vcor-vehicle-color-recognition-dataset) (Vehicle Color Recognition) Dataset from kaggle.

Images are resized to 256x256 and some classes are omitted due to appropriate combination.

## Classes
There are 16 classes as below:

|      |color    |train|validation|total|
|------|---------|-----|----------|-----|
|1     |beige    |950  |124       |1074 |
|2     |black    |1235 |298       |1533 |
|3     |blue     |1102 |275       |1377 |
|4     |brown    |951  |237       |1188 |
|5     |cream    |300  |33        |333  |
|6     |crimson  |223  |9         |232  |
|7     |gold     |300  |25        |325  |
|8     |green    |804  |46        |850  |
|9     |grey     |1373 |343       |1716 |
|10    |navy-blue|290  |45        |335  |
|11    |orange   |762  |47        |809  |
|12    |red      |1040 |259       |1299 |
|13    |silver   |845  |211       |1056 |
|14    |titanium |300  |54        |354  |
|15    |white    |5744 |1435      |7179 |
|16    |yellow   |824  |54        |878  |


## Architecture
This is the model architecture which uses [R50x1](https://tfhub.dev/google/bit/s-r50x1/1) tensorflow hub layer for feature extraction.

<img src="images/architecture.png"
     alt="Markdown Monster icon"/>

# How To Use This Model
### 1. Install Docker
This model is going to serve with [tensorflow serving](https://www.tensorflow.org/tfx/guide/serving), thus you need to install docker.

These are docker installation instruction for:
* [Docker for Windows ](https://docs.docker.com/desktop/windows/install/)
* [Docker for Linux](https://docs.docker.com/desktop/linux/install/)
* [Docker for macOS](https://docs.docker.com/desktop/mac/install/)

### 2. Clone The Project
```bash
git clone https://github.com/m-bashari-m/vehicle-color-recognition.git
```
After cloning the project, change current directory to vehicle-color-recognition directory.
```bash
cd vehicle-color-recognition
```

### 3. Install TensorFlow Serving

```bash
docker pull tensorflow/serving
```
### 4. Create A Docker Network
```bash
docker network create model-net
```
### 5. Run TensorFlow Serving
```bash
docker run -it --rm  \
           -v /path/to/vehicle-color-recognition/saved_model:/models/saved_model \
           -e MODEL_NAME=saved_model \
           --name tf-serving \
           --net model-net  \
           tensorflow/serving
```

### 6. Build App Image
```bash
docker build -t vcor .
```
### 7. Run App
Followig command will run the app.
```bash
docker run -it --rm -v /path/to/data:/data --net model-net vcor
```

### 8. Get TF-Serving Container IP Address
Run following command to get tf-serving conatiner IP address. This IP is going to be used to create url to send request.
```bash
docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' tf-serving
```
Copy and paste the result in the running app.