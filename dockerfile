
FROM python:3.7-slim AS compile-image
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

ADD ./requirements.txt /app
RUN pip3 install --upgrade pip
RUN --mount=type=cache,target=/root/.cache \
    pip3 install --user -r requirements.txt


FROM python:3.7-slim AS build-image
COPY --from=compile-image /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*



EXPOSE 8501

WORKDIR /app

# following entrypoint may cause single package error. still need to check.
# 
#ENTRYPOINT ["streamlit", "run", "sapp.py", "--server.port=8501", "--server.address=0.0.0.0"]

ENTRYPOINT ["python", "sapp.py", "--server.port=8501", "--server.address=0.0.0.0"]
COPY ./figure /app/figure
COPY ./*.py /app/
