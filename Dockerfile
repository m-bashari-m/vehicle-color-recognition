FROM python:3.8-buster

ADD ./requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt

RUN mkdir /templates

COPY ./templates /templates

RUN mkdir /data

COPY app.py ./app.py

CMD [ "python", "app.py" ]
