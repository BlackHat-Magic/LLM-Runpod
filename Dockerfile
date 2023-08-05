FROM python:3.8-slim-buster

WORKDIR /app

ADD . /app

RUN pip install --no-cache-dir -r requirements.txt

RUN python DownloadModel.py

CMD [ "python", "-u", "Manticore-Pygmalion-Guanaco-SuperHOT.py" ]
