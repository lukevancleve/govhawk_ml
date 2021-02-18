FROM python:3.8-slim
RUN apt-get update && apt-get install -y make curl libcurl4-openssl-dev
COPY . /app
RUN cd /app && make
WORKDIR /app
ENV PYTHONPATH "${PYTHONPATH}:/app/"
EXPOSE 80
ENTRYPOINT ["python", "/app/src/flask_code.py"]