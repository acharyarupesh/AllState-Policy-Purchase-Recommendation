# Instructions copied from - https://hub.docker.com/_/python/
FROM python:3-onbuild

COPY . /FinalPipeline

WORKDIR /FinalPipeline

RUN pip3 install pandas
RUN pip3 install numpy
RUN pip3 install scikit-learn
RUN pip3 install luigi
RUN pip3 install smart_open
RUN pip3 install ConfigParser


# run the command
CMD ["python", "./app.py"]