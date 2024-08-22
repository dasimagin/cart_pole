FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL /bin/bash
SHELL ["/bin/bash", "-c"]

WORKDIR /tmp

### COMMON

RUN apt update -q \
    && apt install -yq --no-install-recommends \
        build-essential \
        coinor-libipopt-dev \
        git \
        python3 \
        python3-dev \
        python3-pip \
    && pip3 install --no-cache-dir -U pip \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

### POETRY

RUN pip3 install --no-cache-dir -U poetry \
    && poetry completions bash >> ~/.bash_completion

## PYTHON DEPENDENCIES
COPY pyproject.toml /tmp/pyproject.toml

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root
