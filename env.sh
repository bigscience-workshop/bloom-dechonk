#!/usr/bin/env bash
set -euxo pipefail

############################################################################
# core libraries
############################################################################

apt-get update --allow-unauthenticated --allow-insecure-repositories

apt-get install -y --no-install-recommends \
    build-essential \
    g++ gdb subversion \
    software-properties-common

apt-get install -y --no-install-recommends \
    wget curl vim nano ssh git libssl-dev

apt-get remove -y swig || true
apt-get install -y --no-install-recommends libstdc++6
apt-get install -y --no-install-recommends swig3.0
ln -s /usr/bin/swig3.0 /usr/bin/swig

############################################################################
# install anaconda (because native python stopped working
############################################################################

wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh -b -p /anaconda3
source /anaconda3/bin/activate

############################################################################
# common python libraries (project specfic libs these are installed later)
############################################################################

conda update -y conda
conda install -y python=3.8.12 --strict-channel-priority
conda install -y numpy scipy cython pandas h5py numba
pip install --upgrade setuptools

# common + devops
pip install \
    PyYAML==5.4.1 \
    Pillow==8.3.0 \
    isort==5.9.1 \
    icecream==2.1.1 \
    flake8==3.9.2 \
    uvloop==0.15.2 \
    packaging==19.0 \
    msgpack==0.5.6 \
    sortedcontainers==2.4.0 \
    configargparse==1.2.3 \
    tqdm==4.48.2 \
    protobuf==3.20.1 \
    termcolor==1.0.0

# common data science libs
pip install \
    ninja==1.10.0.post1 \
    tensorboardX==2.4 \
    wandb==0.10.33 \
    matplotlib==3.4.2

# pytorch utils
conda install -y cudatoolkit=11.3 -c pytorch
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# transformers bloom dev branch
pip install https://github.com/younesbelkada/transformers/archive/ba1d9fc05fda160bda968cc77c4c5dbb21049aa9.zip
pip install datasets==2.2.2 accelerate==0.9.0 fairscale==0.4.6

# domain-specific ML libs
pip install \
    sentencepiece==0.1.96 \
    nltk==3.6.2 \
    scikit-learn==1.1.1 \
    gensim==4.0.1 \
    sacrebleu==1.5.1 \
    sacremoses==0.0.45 \
    subword-nmt==0.3.7 \
    youtokentome==1.0.6

pip uninstall -y enum34


############################################################################
# Set locale
############################################################################
locale-gen en_US.UTF-8
update-locale

############################################################################
# Clean
############################################################################
apt-get autoremove
apt-get clean
apt-get autoclean
rm -rf /var/lib/apt/lists/*
rm -rf /tmp/*
rm -rf /.cache
rm -rf /var/cache/apt/*.bin
find /var/log -iname '*.gz' -delete
find /var/log -iname '*.1' -delete

