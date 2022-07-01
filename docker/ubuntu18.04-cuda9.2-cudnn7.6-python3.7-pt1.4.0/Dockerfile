# nvcc installed base-docker for apexp &
FROM nvidia/cuda:9.2-devel-ubuntu18.04

ENV CUDNN_VERSION 7.6.5.32
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
            libcudnn7=$CUDNN_VERSION-1+cuda9.2 && \
                rm -rf /var/lib/apt/lists/*

# for python3
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN add-apt-repository ppa:deadsnakes/ppa

ARG DEBIAN_FRONTEND=noninteractive

# install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-utils \
        python3 \
        wget \
        less \
        locales \
        htop \
        make \
        build-essential \
        git \
        ssh \
        sudo \
        tar \
        screen \
        curl \
        libcurl3-dev \
        libfreetype6-dev \
        libpng-dev \
        pkg-config \
        python3.7-dev \
        tmux \
        unzip \
        zip \
        vim \
        rsync \
        bash \
        bash-completion \
        python3-apt \
        zlib1g-dev \
        zsh \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# support zplug
RUN apt-get update && \
    apt-get install -y --no-install-recommends gawk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install openjpeg and openslide
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        python3-tk \
        libjpeg-dev \
        libxml2-dev \
        libtiff5-dev \
        libcairo2-dev \
        libsqlite3-dev \
        libgdk-pixbuf2.0-dev \
        libglib2.0-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir /openlibs && \
    cd /openlibs && \
    wget https://github.com/uclouvain/openjpeg/archive/v2.2.0.tar.gz && \
    tar -xvf v2.2.0.tar.gz && \
    cd openjpeg-2.2.0 && \
    cmake . && \
    make && \
    sudo make install && \
    sudo ldconfig && \
    cd .. && \
    wget https://github.com/openslide/openslide/releases/download/v3.4.1/openslide-3.4.1.tar.gz && \
    tar -xvf openslide-3.4.1.tar.gz && \
    cd openslide-3.4.1 && \
    ./configure && \
    make && \
    sudo make install && \
    cd / && \
    rm -rf /openlibs

# locale set-up
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# set up TCMalloc
RUN cd /usr/lib && ln -s libtcmalloc.so.4 libtcmalloc.so && ldconfig

# Set python3 as default
# RUN rm /usr/bin/python
RUN ln -s /usr/bin/python3.7 /usr/bin/python

# install python packages via pip
RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py
RUN pip --version

# The recent setuptools has bug (should be fixed <46)
RUN pip --no-cache-dir install setuptools==45

RUN echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf && sudo ldconfig
RUN pip --no-cache-dir install \
        pillow \
        cycler \
        ipykernel==5.1.4 \
        ipython==7.12.0 \
        streamlit==0.56.0 \
        numpy==1.16.1 \
        virtualenv==20.0.4 \
        virtualenvwrapper==4.8.4 \
        openslide-python==1.1.1 \
        pandas==1.0.1 \
        opencv-python==4.1.2.30 \
        ipdb==0.12.3 \
        tqdm==4.42.1 \
        progressbar==2.5 \
        torch==1.4.0 \
        torchvision==0.5.0 \
        zmq

RUN pip install cython

# RUN git clone https://github.com/dingjiansw101/AerialDetection.git && \
#     cd AerialDetection && \
#     sudo bash compile.sh && \
#     pip install -r requirements.txt && \
#     sudo python setup.py develop

# RUN sudo apt-get update && apt-get install -y --no-install-recommends swig && \
#     cd DOTA_devkit && \
#     swig -c++ -python polyiou.i && \
#     python setup.py build_ext --inplace

