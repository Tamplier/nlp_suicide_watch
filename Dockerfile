FROM python:3.12-slim as builder

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y python3-dev libhunspell-dev hunspell-en-us g++
RUN python -m pip install --upgrade pip

WORKDIR /nlp_suicide_watch

COPY requirements.txt .

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y libhunspell-dev hunspell-en-us
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /nlp_suicide_watch

COPY --from=builder /opt/venv /opt/venv

RUN python -m spacy download en_core_web_sm

COPY . .
