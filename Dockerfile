<<<<<<< HEAD
FROM python:3.10-slim

WORKDIR /app

# Copy requirements first
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy everything



EXPOSE 5000

<<<<<<< HEAD
# Run Flask app from correct path
CMD ["python", "app/api/app.py"]

=======
CMD ["python", "app.py"]
>>>>>>> 872d4b79e0adf2d4f3640631de5dc2deb0d483f8
