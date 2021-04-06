FROM python:3.8-slim
RUN apt-get update && apt-get install -y make curl unzip libcurl4-openssl-dev
WORKDIR /app
ENV PYTHONPATH "${PYTHONPATH}:/app/"

COPY requirements.txt /app/requirements.txt
#COPY Makefile /app/Makefile
RUN cd /app && pip install -r requirements.txt
RUN pip install awscli --upgrade --user
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && unzip awscliv2.zip && ./aws/install

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

