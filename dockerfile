FROM python:3.7-slim

EXPOSE 8501

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY ./sapp.py /app/sapp.py
COPY ./requirements.txt /app/requirements.txt

RUN pip3 install -r requirements.txt

# following entrypoint may cause single package error. still need to check.
# 
#ENTRYPOINT ["streamlit", "run", "sapp.py", "--server.port=8501", "--server.address=0.0.0.0"]

ENTRYPOINT ["python", "sapp.py"]
