@echo off

:: Name of the container and image
set CONTAINER_NAME=my-zenml-server
set IMAGE_NAME=zenml-local-9999

echo Stopping container (if running): %CONTAINER_NAME%
docker stop %CONTAINER_NAME% >nul 2>&1

echo Removing container (if present): %CONTAINER_NAME%
docker rm %CONTAINER_NAME% >nul 2>&1

echo Removing image (if present): %IMAGE_NAME%
docker rmi %IMAGE_NAME% >nul 2>&1

echo Rebuilding image: %IMAGE_NAME%
docker build -t %IMAGE_NAME% .

echo Running new container: %CONTAINER_NAME%
docker run -it -p 9999:9999 --name %CONTAINER_NAME% %IMAGE_NAME%

pause
