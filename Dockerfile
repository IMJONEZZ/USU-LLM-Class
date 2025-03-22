# Use an official Python 3.10 image (or whichever suits you).
FROM python:3.10

# Install any system packages you might need (git, etc.)
RUN apt-get update && apt-get install -y git

# Create a working directory
WORKDIR /app

# Copy your requirements.txt first so Docker can cache the pip install layer
COPY requirements.txt /app/

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app code (app.py, static folder, etc.)
COPY . /app


# Expose port 8000 for the FastAPI server
EXPOSE 8000

# By default, run the server
CMD ["python", "app.py"]
