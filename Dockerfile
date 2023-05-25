FROM python:3.9

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y libheif-dev

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx

RUN pip install --no-cache-dir -r requirements.txt

# COPY ./model ~/.EasyOCR/model

COPY . .

EXPOSE 5000

CMD ["python", "./app.py"]