FROM python:3.8-buster

ADD ./requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt

COPY app.py ./app.py

CMD [ "python3", "app.py" ]
