FROM python:3.8-slim-buster

ARG CACHE_DIR

ARG MODEL_ID

ARG DEFAULT_TEMPERATURE

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists

WORKDIR /app

ADD ./.env .

ADD ./requirements.txt .

ADD ./src/RunLLM.py .

RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "-u", "RunLLM.py" ]
