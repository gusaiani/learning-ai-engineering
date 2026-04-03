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
    with open(filepath, "rb") as f:
        response = client.files.create(file=f, purpose="fine-tune")
    return response.id

def create_fine_tuning_job(file_id: str) -> str:
    """Create a fine-tuning job and return the job ID."""
    job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model="gpt-4o-mini-2024-07-18",
        hyperparameters={"n_epochs": 3},
    )
    return job.id

def monitor_job(job_id: str) -> str:
    """Poll until the job completes. Return the fine-tuned model ID."""
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"  Status: {job.status}")

        if job.status == "succeeded":
            return job.fine_tuned_model
        if job.status == "failed":
            raise RuntimeError(f"Fine-tuning failed: {job.error}")

        time.sleep(30)

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
