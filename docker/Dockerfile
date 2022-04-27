FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}

# every shell will spawn will set PYTHONPATH and correct environment
ENV PATH "$PATH:/home/.local/bin"
ENV PYTHONPATH=$PYTHONPATH:$CODE_PATH

ENV TF_GPU_MEMORY_FRACTION 0.3
ENV NVIDIA_VISIBLE_DEVICES 0

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# Install pip requirements
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install pip requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /src
