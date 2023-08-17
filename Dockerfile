# Base image
FROM runpod/pytorch:3.10-2.0.1-117-devel

SHELL ["/bin/bash", "-c"]

WORKDIR /

# Environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONBUFFERED 1
ENV DEBIAN_FRONTEND noninteractive

# Install packages
COPY builder /
RUN /bin/bash /packages.sh && rm /packages.sh

# Install Python dependencies (Worker Template)
RUN pip install --upgrade pip --no-cache-dir && \
    pip install -r /requirements.txt && \
    rm /requirements.txt && \
    rm -r /root/.cache/pip

# Fetch model
RUN python /cache_models.py && rm /cache_models.py

ADD src .

CMD [ "python", "-u", "/rp_handler.py" ]
