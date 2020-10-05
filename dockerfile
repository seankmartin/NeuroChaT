FROM python:3

WORKDIR /usr/src/app

RUN apt-get update
RUN apt-get install python3-pyqt5 -y

COPY . .
RUN python -m pip --no-cache-dir install .

CMD ["python", "./cli.py"]
