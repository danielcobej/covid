# Base image: use official lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt into the container at /app
COPY requirements.txt /app/

RUN apt-get update && apt-get install -y libgomp1


# Install the required Python packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of your application code to the container
COPY . /app

# Set the environment variable to prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Specify the port that the container will listen on at runtime
EXPOSE 8050

# Run the ExplainerDashboard application
CMD ["python", "dashboard.py"]