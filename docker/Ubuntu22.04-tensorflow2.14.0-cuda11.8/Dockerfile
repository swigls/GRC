FROM tensorflow/tensorflow:2.14.0-gpu

# ENV http_proxy=http://aaa.bbb.cc.net:8080 \
#	https_proxy=http://aaa.bbb.cc.net:8080

# install normal source
# ARG DEBIAN_FRONTEND=noninteractive
# RUN rm /etc/apt/sources.list.d/*
RUN apt-get -y update 
RUN apt-get install -y \
     git \
     wget \
     dpkg \
     libsndfile1
# RUN apt-get update && \
    # apt-get install -y --no-install-recommends \
    #     g++ \
    #     make \
    #     automake \
    #     autoconf \
    #     bzip2 \
    #     unzip \
    #     wget \
    #     sox \
    #     libtool \
    #     git \
    #     subversion \
    #     zlib1g-dev \
    #     gfortran \
    #     ca-certificates \
    #     patch \
    #     ffmpeg \
    #     valgrind \
	# 	libssl-dev \
	#     vim \
	# 	curl

RUN git clone https://github.com/swigls/GRC.git /opt/GRC && \
    cd /opt/GRC && \
    git checkout newreturnn && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# # cmake
# RUN wget -P /opt https://cmake.org/files/v3.18/cmake-3.18.0.tar.gz && \
#     cd /opt && \
#     tar -zxvf cmake-3.18.0.tar.gz && \
#     cd cmake-3.18.0 && \
#     ./bootstrap && \
#     make && \
#     make install && \
#     rm -rf cmake-3.18.0.tar.gz && \
#     find /opt/cmake-3.18.0 -type f \( -name "*.o" -o -name "*.la" -o -name "*.a" \) -exec rm {} \; && \
#     cd -

# # flac
# RUN wget -P /opt https://downloads.xiph.org/releases/flac/flac-1.3.2.tar.xz  && \
#     cd /opt && \
#     xz -d flac-1.3.2.tar.xz && \
#     tar -xvf flac-1.3.2.tar && \
#     cd flac-1.3.2 && \
#     ./configure && \
#     make && make install && \
#     rm -rf flac-1.3.2.tar && \
#     find /opt/flac-1.3.2  -type f \( -name "*.o" -o -name "*.la" -o -name "*.a" \) -exec rm {} \; && \
#     cd -

# #RUN conda install -y -c pytorch torchaudio=0.12 && \
# RUN pip install graphviz


# RUN pip install kaldifeat
# RUN pip install gin-config

# # install  lhotse
# RUN cd /opt
# RUN pip install git+https://github.com/lhotse-speech/lhotse

# #install k2 from source
# #RUN git clone https://github.com/k2-fsa/k2.git /opt/k2 && \
# RUN git clone https://github.com/swigls/k2.git /opt/k2 && \
#     cd /opt/k2 && \
#     export K2_MAKE_ARGS="-j6" && \
#     mkdir build_debug && \
#     cd build_debug && \
#     cmake -DCMAKE_BUILD_TYPE=Debug .. && \
#     make -j
# RUN chmod -R 777 /opt/k2

# # install icefall
# RUN git clone https://github.com/k2-fsa/icefall /opt/icefall && \
# 	cd /opt/icefall && \
# 	pip install -r requirements.txt

# ENV PYTHONPATH /opt/icefall:$PYTHONPATH

# ENTRYPOINT cd /opt/k2; \
#     git checkout -- .; \
#     git pull; \
#     git checkout torch2; \
#     cd -; \
#     export PYTHONPATH=/opt/k2/k2/python:$PYTHONPATH; \
#     export PYTHONPATH=/opt/k2/build_debug/lib:$PYTHONPATH; \
#     /opt/nvidia/nvidia_entrypoint.sh
    

# WORKDIR /workspace/icefall