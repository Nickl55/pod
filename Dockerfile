FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir runpod diffusers transformers accelerate

# Copy your handler file
COPY handler.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run when container starts
CMD ["python", "handler.py"]
