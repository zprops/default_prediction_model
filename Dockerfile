# 🛑 Use the official Python base image as a starting point
FROM --platform=linux/amd64 python:3.10-slim

# 🛑 Set the working directory for the container
WORKDIR /app

# 🛑 Copy the requirements.txt file into the container
COPY requirements.txt ./

# 🛑 Install the required packages using pip
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# 🛑 Install system dependencies
RUN apt-get update && apt-get install -y libgomp1 build-essential

# 🛑 Copy the Flask application files into the container
COPY . .

# 👇 Expose the Flask application on port 8000 (Read the third learning content link)
EXPOSE 8000

# 👇 Start the Flask application by running app.py with python and no extra arguments (Watch the second learning content link video)
CMD ["python", "app.py"]
