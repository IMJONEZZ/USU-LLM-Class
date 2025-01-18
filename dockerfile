FROM python:3.11

# Set your working directory
WORKDIR /C/Users/Natha/Desktop/LLM/USU-LLM-Class

RUN pip install --no-cache-dir "zenml[server]"
RUN yes | zenml init

COPY run.py .
COPY tokenizer.py .
COPY SW_EpisodeIV_VI.json .

# Expose the ZenML server port
EXPOSE 9999

CMD ["sh", "-c", "zenml login --local --ip-address 0.0.0.0 --port 9999 --blocking && python run.py"]
