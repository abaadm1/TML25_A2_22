import requests
import torch
import torch.nn as nn
# Do install:
# conda install onnx
# conda install onnxruntime
import onnxruntime as ort
import numpy as np
import json
import io
import sys
import base64
from torch.utils.data import Dataset
from typing import Tuple
import pickle
import os

### REQUESTING NEW API ###
TOKEN = "12910150"
response = requests.get("http://34.122.51.94:9090" + "/stealing_launch",
headers={"token": TOKEN})
answer = response.json()
print(answer)
if 'detail' in answer:
    sys.exit(1)
# save the values
SEED = str(answer['seed'])
PORT = str(answer['port'])

print(f"SEED: {SEED}, PORT: {PORT}")