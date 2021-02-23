# sudo docker run --env DATA_VOL='/data/' --volume /home/luke/tmp1/:/data/ flask2
# sudo docker run --env DATA_VOL='/data/' --volume /home/luke/tmp1/:/data/ -it --entrypoint /bin/bash flask2
FROM python:3.8-slim
RUN apt-get update && apt-get install -y make curl libcurl4-openssl-dev
COPY . /app
RUN cd /app && make
WORKDIR /app
ENV PYTHONPATH "${PYTHONPATH}:/app/"
EXPOSE 5000
ENTRYPOINT ["python", "/app/src/flask_code.py"]