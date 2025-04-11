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

# Copy the rest of your app code (streamlit_app.py, static folder, etc.)
COPY . /app

# Expose port 8501 for the Streamlit app
EXPOSE 8501

# By default, run the Streamlit app.
# Using "--server.enableCORS=false" helps avoid CORS issues,
# and we explicitly set the port to 8501.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
