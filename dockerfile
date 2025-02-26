FROM python:3.11

WORKDIR /app

# Update pip to ensure latest wheels are recognized
RUN pip install --upgrade pip

# Now allow PyTorch CPU wheels to come from the extra index, but keep PyPI as primary
RUN pip install --no-cache-dir \
    "zenml[server]" \
    transformers \
    torch \
    torchvision \
    torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN yes | zenml init

# Expose 8080 for the ZenML local server
EXPOSE 8080

COPY main.py pipeline.py data_loader.py tokenizer.py dataset.py model.py train.py /app/
COPY SW_EpisodeIV_VI.json /app/

CMD ["sh", "-c", "zenml login --local --ip-address 0.0.0.0 --port 8080 && python /app/main.py && tail -f /dev/null"]
