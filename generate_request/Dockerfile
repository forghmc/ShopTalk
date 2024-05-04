# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set environment variables
ENV OPENAI_API_KEY="" \
PINE_CONE_API_KEY=""
  
# Set the working directory in the container
WORKDIR /app

# Copy the Python package files
# COPY setup.py .

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable for Flask
ENV FLASK_APP=generate_request.py

# Run flask when the container launches
#CMD ["generate_request", "run", "--host=0.0.0.0"]
CMD ["python", "generate_request.py"]
