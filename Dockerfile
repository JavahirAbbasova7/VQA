FROM nvidia/cuda:11.0.3-base-ubuntu20.04

WORKDIR /home

RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
    libsm6 libxext6 libxrender-dev curl \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y git vim curl wget build-essential


RUN echo "**** Installing Python ****" && \
    add-apt-repository ppa:deadsnakes/ppa &&  \
    apt-get install -y build-essential python3.8 python3.8-dev python3-pip && \
    curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3.8 get-pip.py && \
    rm -rf /var/lib/apt/lists/*

COPY . .

RUN python3.8 -m pip install -r requirements.txt

ENTRYPOINT ["/bin/bash"]