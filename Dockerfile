# Base image
FROM runpod/pytorch:3.10-2.0.0-117

SHELL ["/bin/bash", "-c"]

WORKDIR /

# Environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONBUFFERED 1
ENV DEBIAN_FRONTEND noninteractive

# Install packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /requirements.txt && \
    rm /requirements.txt

# Fetch model
COPY builder/model_fetcher.py /model_fetcher.py
RUN python /model_fetcher.py
RUN rm /model_fetcher.py

ADD src .

CMD [ "python", "-u", "/rp_handler.py" ]