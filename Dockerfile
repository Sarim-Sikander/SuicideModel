FROM tiangolo/uvicorn-gunicorn:python3.8-slim

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN python3 -m spacy download en_core_web_lg

COPY ./app /app