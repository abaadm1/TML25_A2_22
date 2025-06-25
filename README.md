## 1. Introduction
In this assignment, we successfully replicated the behavior of a protected machine learning encoder using model stealing techniques, despite the presence of B4B (Bucks for Buckets) defense. Our final surrogate encoder achieved a competitive L2 distance of **4.706**. This readme provides a thorough, step-by-step explanation of our methodology, the difficulties we encountered, and how each design choice assisted in resolving them.

### Repository Contents

This repository includes all necessary components for replicating our model stealing attack against the B4B-protected encoder:

- `api_requestor.py`: Automates the API querying process.

- `encoder-stealing.ipynb`: Core training pipeline notebook **(main code file)**

- `stolen_encoder_final.pt`: Trained PyTorch checkpoint of the stolen encoder.

- `stolen_encoder_optimized.onnx`: Final ONNX-exported model for evaluation and submission.

- `requirements.txt`: Lists all Python dependencies for reproducibility and environment setup.


## 2. Strategy and Challenges
### 2.1 Dataset & API Query Strategy
We first determined that the provided dataset `ModelStealingPub.pt` contained exactly 13,000 images. In the surrogate dataset approach, we start by shuffling the entire dataset. Then, we run 13 separate queries, each retrieving 1,000 samples, resulting in a final dataset of 13,000 samples arranged in the order the queries were made. Each query returned 1024-dimensional embeddings from the victim encoder:

```python
embeddings = model_stealing(batch_images, port=PORT)
```
These embeddings were saved as `out{i}.pickle` for later training. The dataset can be accessed [here](https://drive.google.com/drive/folders/1XYM_9pgHlzaAsTavjNof0jqvw2tHwq_a?usp=drive_link).
### 2.2 Initial Model
We initially trained a surrogate model (ResNet-50) via MoCo-style contrastive learning inspired by the paper https://yangzhangalmo.github.io/papers/CVPR23.pdf. However, the results were disappointing, with L2 ≈ 120. We diagnosed three main problems:
- We faced the issue of **inconsistent channels** in the images, some of the images or embeddings were not RGB which we fixed using 
```python
img.convert("RGB")
```
<br> 
- ResNet50 was **over-parameterized** for small 32x32 images. <br>
- We weren’t intendedly **mitigating B4B noise** at this point to see how strong the victim encoder’s B4B defense is, which was corrupting our training targets. <br>

## 3. Transition to BESA (Perturbation Recovery)
To counter B4B, we adopted the BESA strategy from https://arxiv.org/pdf/2506.04556

