# sudo docker run --env DATA_VOL='/data/' --volume /home/luke/tmp1/:/data/ flask2
# sudo docker build --name dlbox
# sudo docker run --env DATA_VOL='/data/' --volume /datavol:/data/ -it --entrypoint /bin/bash dlbox
# sudo docker run --env DATA_VOL='/data/' -p 127.0.0.1:8000:8000  --volume /datavol:/data/ --volume /home/luke/repos/govhawk_ml/:/app:cached -it --entrypoint /bin/bash dlbox
#
# docker build -t dlbox .
# docker-compose up dev
# docker exec -it
##

FROM python:3.8-slim
RUN apt-get update && apt-get install -y make curl libcurl4-openssl-dev
WORKDIR /app
ENV PYTHONPATH "${PYTHONPATH}:/app/"

COPY requirements.txt /app/requirements.txt
COPY Makefile /app/Makefile
RUN cd /app && make && rm -rf /app/*

#CMD ["/bin/bash"]
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
#EXPOSE 5000
#ENTRYPOINT ["python", "/app/src/flask_code.py"]
