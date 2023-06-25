#! /bin/bash

set -e

apt-get update -y

apt-get install -y --no-install-recommends \
git \
wget \
libgl1-mesa-glx \
libglib2.0-0

apt-get clean && \
apt-get autoremove && \
rm -rf /var/lib/apt/lists/*
