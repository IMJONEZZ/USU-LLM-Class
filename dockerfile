FROM python:3.11

WORKDIR /app

RUN pip install --no-cache-dir "zenml[server]" transformers

RUN yes | zenml init

EXPOSE 8080

COPY main.py pipeline.py data_loader.py tokenizer.py dataset.py model.py train.py /app/
COPY SW_EpisodeIV_VI.json /app/


CMD ["sh", "-c", "zenml login --local --ip-address 0.0.0.0 --port 8080 && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && python /app/main.py && tail -f /dev/null"]
