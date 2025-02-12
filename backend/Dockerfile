
FROM python:3.12

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg python3-pip && rm -rf /var/lib/apt/lists/*

COPY requirements.txt app.py ./


RUN pip install --no-cache-dir -r requirements.txt


ENV FLASK_ENV=production
ENV PORT=5510


EXPOSE 5510

CMD ["python", "app.py"]