FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt-get update --fix-missing
RUN apt-get -y install libmkl-dev
RUN apt-get install nano
RUN apt-get install less
RUN apt install -y python3 python3-virtualenv python3-pip
RUN pip install Jupyter
RUN pip install ipykernel
RUN pip install numpy==1.23.5
RUN pip install make
RUN pip install cython
RUN pip install MKL
RUN pip install pyMKL
ENV CFLAGS="-I /usr/include/mkl/" 
RUN pip install -y pydiso
ADD . /simpeg_em1d_stiched
RUN cd /simpeg_em1d_stiched; pip install .
