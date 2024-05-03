#FROM python:3.10-slim-buster

#RUN apt update -y && apt install awscli -y
#WORKDIR /app

#COPY . /app
#RUN pip install -r requirements.txt

#CMD ["python3", "appv2.py"]

# Use an official Python runtime as a parent image

FROM python:3.8

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit will run on
EXPOSE 8501

# Command to run the app
CMD ["python", "-m", "streamlit", "run", "appv2.py"]

