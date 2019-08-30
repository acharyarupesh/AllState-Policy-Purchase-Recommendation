# Instructions copied from - https://hub.docker.com/_/python/
FROM python:3-onbuild

RUN pip3 install pandas
RUN pip3 install numpy
RUN pip3 install scikit-learn

# tell the port number the container should expose
EXPOSE 5000

# run the command
CMD ["python", "./app.py"]