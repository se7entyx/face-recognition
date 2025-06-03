# Use the Python 3 official image
# https://hub.docker.com/_/python
FROM python:3.10.6

# Run in unbuffered mode
ENV PYTHONUNBUFFERED=1 

# Create and change to the app directory.
WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Copy local code to the container image.
COPY . ./

# Install project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the web service on container startup.
CMD ["gunicorn", "face_recognition:app"]