FROM docker.io/library/ubuntu:latest

WORKDIR 

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN git clone https://github.com/loua19/muse