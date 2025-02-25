FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing \
&& apt-get install -y libegl1-mesa libgl1-mesa-glx libglib2.0-0 git libglu1-mesa-dev libgl1-mesa-dev  \
&& apt-get clean

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src /project/src

ENV PYTHONPATH "${PYTHONPATH}:/project"
ENV PYTHONPATH "${PYTHONPATH}:/opt/project"

WORKDIR /project

CMD ["python", "/project/src/gui.py"]

