# finetune.py
import json
import time
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


def upload_training_file(filepath: str) -> str:
    """Upload a JSONL file to OpenAI and return the file ID."""
    # TODO: use client.files.create()
    pass


def create_fine_tuning_job(file_id: str) -> str:
    """Create a fine-tuning job and return the job ID."""
    # TODO: use client.fine_tuning.jobs.create()
    # Model: gpt-4o-mini-2024-07-18
    # Hyperparameters: n_epochs=3
    pass


def monitor_job(job_id: str) -> str:
    """Poll until the job completes. Return the fine-tuned model ID."""
    # TODO: poll client.fine_tuning.jobs.retrieve() every 30 seconds
    # Print status updates and training metrics
    # Return the fine-tuned model name when done
    pass


if __name__ == "__main__":
    print("Uploading training file...")
    file_id = upload_training_file("train.jsonl")
    print(f"File uploaded: {file_id}")

    print("\nCreating fine-tuning job...")
    job_id = create_fine_tuning_job(file_id)
    print(f"Job created: {job_id}")

    print("\nMonitoring job progress...")
    model_id = monitor_job(job_id)
    print(f"\nFine-tuned model ready: {model_id}")

    # Save model ID for eval
    with open("model_id.txt", "w") as f:
        f.write(model_id)
    print("Model ID saved to model_id.txt")
