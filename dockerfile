FROM python:3.11

# Set working directory inside the container
WORKDIR /app

# Install ZenML (before installing PyTorch)
RUN pip install --no-cache-dir "zenml[server]"

# Initialize ZenML
RUN yes | zenml init

# Expose the ZenML server port
EXPOSE 8080

# Copy required files (but NOT pytorch_packages)
COPY run.py /app/
COPY SW_EpisodeIV_VI.json /app/

# Start ZenML and run the tokenizer pipeline
CMD ["sh", "-c", "zenml login --local --ip-address 0.0.0.0 --port 8080 && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && python /app/run.py && tail -f /dev/null"]
