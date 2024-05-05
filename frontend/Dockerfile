# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set environment variables
ENV OPENAI_API_KEY = "" \
PINE_CONE_API_KEY = ""
  
# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Copy Streamlit specific files
COPY .streamlit/ /root/.streamlit/

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 8501

# Define environment variable for Flask
# ENV FLASK_APP=frontend.py

# Run flask when the container launches
#CMD ["frontend", "run", "--host=0.0.0.0"]
CMD ["streamlit", "run", "frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]
