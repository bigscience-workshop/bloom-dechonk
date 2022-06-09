############################################################################
# apt-get part
############################################################################

apt-get --allow-insecure-repositories --allow-unauthenticated update
apt-get install -y --allow-unauthenticated --no-install-recommends software-properties-common gnupg2
curl -sSL \
'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0x7FCD11186050CD1A' \
| apt-key add -
# apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 7FCD11186050CD1A
apt-get purge -y gnupg2
add-apt-repository -y -n ppa:ubuntu-toolchain-r/test

echo "deb http://dk.archive.ubuntu.com/ubuntu/ xenial main" | tee -a /etc/apt/sources.list
echo "deb http://dk.archive.ubuntu.com/ubuntu/ xenial universe" | tee -a /etc/apt/sources.list

apt-get autoremove -y

apt-get --allow-insecure-repositories --allow-unauthenticated update || true

apt-get install  -y --allow-unauthenticated --no-install-recommends --force-yes gcc-9 g++-9
update-alternatives --remove-all gcc || true
update-alternatives --remove-all g++ || true
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 60
apt-get --allow-insecure-repositories --allow-unauthenticated update || true
apt-get upgrade  -y --allow-unauthenticated --no-install-recommends --force-yes binutils libc6
apt-get install -y --allow-unauthenticated --no-install-recommends --force-yes libmpfr6 libmpfr-dev
apt-get upgrade -y --allow-unauthenticated --no-install-recommends --force-yes libmpfr6 libmpfr-dev
apt-get install  -y --allow-unauthenticated --no-install-recommends --force-yes build-essential gdb wget git nano

# conda
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.11.0-Linux-x86_64.sh
chmod +x Miniconda3-py39_4.11.0-Linux-x86_64.sh
./Miniconda3-py39_4.11.0-Linux-x86_64.sh -bs -p /conda
rm Miniconda3-py39_4.11.0-Linux-x86_64.sh

/conda/bin/conda create -y -n main python==3.8.12
source /conda/bin/activate main

ln -s `which python` /usr/bin/python-conda
ln -s `which pip` /usr/bin/pip-conda

# cuda
conda install -y -c conda-forge cudatoolkit-dev==11.3.1
conda install -y -c conda-forge cudatoolkit==11.3.1 cudnn==8.2.1.32

export CUDA_PATH=$CONDA_PREFIX
# bin libs
conda install -y -c conda-forge sox==14.4.2

ln -s `which sox` /usr/bin/sox
ln -s `which soxi` /usr/bin/soxi

conda install -y -c conda-forge libsndfile==1.0.31


############################################################################
# common python libraries (project specfic libs these are installed later)
############################################################################

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# common + devops
pip install --no-input cython==0.29.30

pip install --no-input \
  cmake==3.22.4 \
  lmdb==1.3.0 \
  requests==2.27.1 \
  numpy==1.22.2 \
  scipy==1.8.1 \
  packaging==21.3 \
  ftfy==6.1.1 \
  regex==2022.6.2 \
  blobfile==1.3.1 \
  pyspng==0.1.0 \
  decorator==5.1.1 \
  pudb==2019.2 \
  omegaconf==2.1.1 \
  test-tube==0.7.5 \
  more_itertools==8.13.0

pip install --no-input \
  attrs==20.3.0 \
  einops==0.3.0 \
  librosa==0.8.1 \
  matplotlib==3.3.4 \
  nltk==3.6.2 \
  pandas==1.2.2 \
  pillow==8.3.1 \
  protobuf==3.17.3 \
  pytorch-lightning==1.6.1 \
  PyYAML==5.4.1 \
  sacremoses==0.0.49 \
  scikit-learn==0.24.1 \
  sentencepiece==0.1.96 \
  tensorboard==2.8.0 \
  tensorboardX==2.5 \
  tokenizers==0.12.1 \
  torchmetrics==0.8.0 \
  tqdm==4.61.2 \
  wandb==0.10.33


# transformers bloom dev branch
pip install https://github.com/younesbelkada/transformers/archive/ba1d9fc05fda160bda968cc77c4c5dbb21049aa9.zip
pip install datasets==2.2.2 accelerate==0.9.0
DS_BUILD_CPU_ADAM=1 DS_BUILD_AIO=1 DS_BUILD_UTILS=1 pip install deepspeed==0.6.5 \
 --global-option="build_ext" --global-option="-j8" --no-cache -v --disable-pip-version-check

# domain-specific ML libs
pip install \
    sentencepiece==0.1.96 \
    nltk==3.6.2 \
    gensim==4.0.1 \
    sacrebleu==1.5.1 \
    sacremoses==0.0.45 \
    subword-nmt==0.3.7

pip uninstall -y enum34

