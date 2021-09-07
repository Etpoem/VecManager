FROM nvidia/cuda:11.2.0-runtime-ubuntu16.04

WORKDIR /app
COPY ./* ./

# update source and install basic dependencies modify time zone
RUN sed -i s/archive.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list \
 && sed -i s/security.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list \
 && rm /etc/apt/sources.list.d/* \
 && apt-get update -y \
 && apt-get install -y --no-install-recommends tzdata \
 && ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
 && dpkg-reconfigure -f noninteractive tzdata \
 && rm -rf /var/lib/apt/lists/*

# install miniconda 
RUN chmod +x ./miniconda.sh \
 && ./miniconda.sh -b -p /miniconda \
 && rm ./miniconda.sh

ENV CONDA_AUTO_UPDATE_CONDA=false

# create environent
RUN /miniconda/bin/conda config --add channels http://mirrors.bfsu.edu.cn/anaconda/pkgs/free/ \
 && /miniconda/bin/conda config --add channels http://mirrors.bfsu.edu.cn/anaconda/pkgs/main/ \
 && /miniconda/bin/conda config --add channels http://mirrors.bfsu.edu.cn/anaconda/cloud/pytorch/ \
 && /miniconda/bin/conda config --add channels http://mirrors.bfsu.edu.cn/anaconda/cloud/conda-forge/ \
 && /miniconda/bin/conda config --remove channels defaults \
 && /miniconda/bin/conda create --name face python=3.8 \
 && /miniconda/bin/conda clean -ya

# set environment path
ENV CONDA_DEFAULT_ENV=face
ENV CONDA_PREFIX=/minconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=/miniconda/envs/face/bin:$PATH

# pip environment
RUN pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple \
 && pip config set install.trusted-host mirrors.bfsu.edu.cn \
 && pip install --no-cache-dir -r requirements.txt \
 && rm ./*

