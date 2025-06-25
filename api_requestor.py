import requests
import torch
import numpy as np
import json
import io
import sys
import base64
import pickle
import os
import time
from torchvision import transforms
from torch.utils.data import Dataset
from typing import Tuple
from PIL import Image

# === Step 1: Launch API ===
TOKEN = "12910150"  # Replace with your actual token

# response = requests.get("http://34.122.51.94:9090/stealing_launch", headers={"token": TOKEN})
# answer = response.json()

# print("API Response:", answer)
# if 'detail' in answer:
#     sys.exit("Failed to launch API.")

# SEED = str(answer['seed'])
# PORT = str(answer['port'])

SEED = "69713536"
PORT = "9025"

# === Step 2: Load Public Dataset ===
class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)

torch.serialization.add_safe_globals({'TaskDataset': TaskDataset})

# Load dataset from file
dataset = torch.load("ModelStealingPub.pt", weights_only=False)
all_indices = np.random.permutation(len(dataset.imgs))

print(f"Dataset loaded with {len(dataset.imgs)} images.")


# === Step 3: Define Query Function ===
def model_stealing(images, port):
    endpoint = "/query"
    url = f"http://34.122.51.94:{port}" + endpoint
    image_data = []
    for img in images:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        image_data.append(img_base64)

    payload = json.dumps(image_data)
    response = requests.get(url, files={"file": payload}, headers={"token": TOKEN})
    if response.status_code == 200:
        return response.json()["representations"]
    else:
        raise Exception(f"API query failed. Code: {response.status_code}, content: {response.json()}")

# === Query in Batches with Delay ===
n_queries = 13            # Number of batches (you can increase later)
batch_size = 1000
delay_seconds = 90          # 1.5 minutes

print(f"Starting {n_queries} queries with {delay_seconds}s delay between each...")

for i in range(n_queries):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size
    if end_idx > len(dataset.imgs):
        print(f"Not enough images to fill batch {i+1}")
        break

    batch_indices = all_indices[start_idx:end_idx]
    batch_images = [dataset.imgs[idx] for idx in batch_indices]

    print(f"[{i+1}/{n_queries}] Querying API for images {start_idx}–{end_idx - 1}...")
    embeddings = model_stealing(batch_images, port=PORT)

    # Save results to out{i+1}.pickle
    save_data = {
        "indices": batch_indices.tolist(),
        "embeddings": embeddings,
    }
    filename = f"out{i+1}.pickle"
    with open(filename, "wb") as f:
        pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved results to {filename}")

    if i < n_queries - 1:
        print(f"Waiting {delay_seconds} seconds before next query...\n")
        time.sleep(delay_seconds)

print("All queries completed.")
