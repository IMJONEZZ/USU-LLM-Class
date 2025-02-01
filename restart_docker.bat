@echo off
echo Stopping and removing existing Docker container...
docker stop zenml-container >nul 2>&1
docker rm zenml-container >nul 2>&1

echo Building the Docker image...
docker build -t zenml-container .

echo Running the new Docker container with pytorch_packages mounted...
docker run -it -p 8080:8080 --name zenml-container -v C:\Users\Natha\Desktop\LLM\pytorch_packages:/pytorch_packages zenml-container
