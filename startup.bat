@echo off
REM Start Docker Desktop if not already running
echo Starting Docker Desktop...
start "Docker Desktop" "C:\Program Files\Docker\Docker\Docker Desktop.exe"

REM Give Docker time to initialize
timeout /t 10

REM Kill all running containers
for /f "tokens=*" %%i in ('docker ps -q') do (
    docker kill %%i
)

REM Build the Docker image from the local Dockerfile
echo Building Docker image...
docker build -t my-llama-app:latest .

REM Run the container
echo Running Docker container on port 8000...
docker run -p 8000:8000 my-llama-app:latest