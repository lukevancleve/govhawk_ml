FROM ubuntu:18.04
RUN apt-get update && apt-get install -y make python3 python3-pip curl libcurl4-openssl-dev
COPY . /app
RUN cd /app && make