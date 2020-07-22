FROM python:3

WORKDIR /usr/src/app

# Can't get PyQT to work currently
# RUN apt-get update
# RUN apt-get install python3-pyqt5 -y

COPY . .
RUN python -m pip --no-cache-dir install .

CMD ["bash"]
