# Base image
#FROM runpod/pytorch:3.10-2.0.0-117
FROM runpod/pytorch:3.10-2.0.1-117-devel

SHELL ["/bin/bash", "-c"]

WORKDIR /

# Environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONBUFFERED 1
ENV DEBIAN_FRONTEND noninteractive

# Install packages
# COPY builder/packages.sh /packages.sh
COPY builder /
RUN /bin/bash /packages.sh && rm /packages.sh

# Install Python dependencies (Worker Template)
# COPY builder/requirements.txt /requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt && \
    rm -r /root/.cache/pip

# Fetch model
# COPY builder/cache_models.py /cache_models.py
RUN python /cache_models.py && rm /cache_models.py

ADD src .

CMD [ "python", "-u", "/rp_handler.py" ]