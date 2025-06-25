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
We initially trained a surrogate model (ResNet-50) via MoCo-style contrastive learning inspired by this [research paper](https://yangzhangalmo.github.io/papers/CVPR23.pdf). However, the results were disappointing, with L2 ≈ 120. We diagnosed three main problems:
- We faced the issue of **inconsistent channels** in the images, some of the images or embeddings were not RGB which we fixed using 
```python
img.convert("RGB")
```
<br> 
- ResNet50 was **over-parameterized** for small 32x32 images. <br>
- We weren’t intendedly **mitigating B4B noise** at this point to see how strong the victim encoder’s B4B defense is, which was corrupting our training targets. <br>

## 3. Transition to BESA (Perturbation Recovery)
To counter B4B, we adopted the BESA strategy from this [research paper](https://arxiv.org/pdf/2506.04556)
### 3.1 How BESA Works
- **Train shadow encoders** on a similar domain to mimic the victim encoder.
- **Apply perturbations** (Gaussian noise, Top-k masking, quantization) to simulate B4B.
- **Train a meta-classifier** to detect both the existence and type of perturbation.
- **Train a generator** (often adversarially) to recover clean embeddings from noisy ones.

### 3.2 Results of BESA Phase
Using the BESA pipeline, we improved our performance:Score improved from ~120 → 44 and further tuning reduced L2 distance to 25.47.
The model architecture used was: 
- Encoder (ResNet50 Backbone) and Projector Head.
- Dataset augmentation: resize to 32x32 → Center crop → Normalize but no heavy augmentation (e.g., no ColorJitter/RandomFlip) to avoid distorting embeddings.
- Some hyperparameter tuning of batch size, learning rate and weight decay etc.
- HybridLoss combining MSE, Cosine and Contrastive Loss.

However, keeping BESA as a preprocessing-only pipeline was not optimal. We realized the need to embed noise-awareness directly into our training loop to mimick the behavior of B4B defence.


## 4. Our Final Approach

### 4.1. Custom Model Architecture  
We moved from ResNet50 to a lightweight model, `EnhancedResNetEncoder`, tailored for 32×32 images. It consists of simple residual blocks followed by a GELU-activated bottleneck:

```python
class EnhancedResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet_layers = ...  # simplified residual layers
        self.bottleneck = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.GELU(),
            nn.Dropout(0.314),
            nn.Linear(512, 1024)
        )
```

- The **512 bottleneck** reduced overfitting and handled noise more effectively.  
- **GELU activation** replaced ReLU, providing smoother gradients under perturbations.  
- **Dropout** helped regularize noisy supervision from B4B-affected targets.

---

### 4.2. Hybrid Loss Function  
We developed a custom `HybridLoss` function to improve training stability under noisy labels:

```python
class HybridLoss(nn.Module):
    def forward(self, pred, target):
        mse = F.mse_loss(pred, target)
        cosine = 1 - F.cosine_similarity(pred, target).mean()
        shuffled = target[torch.randperm(target.size(0))]
        contrastive = F.cosine_similarity(pred, shuffled).mean()
        return alpha * mse + (1 - alpha) * cosine + 0.1 * contrastive
```

- **MSE** captured magnitude differences despite noise.  
- **Cosine similarity** enforced directional similarity between vectors.  
- **Contrastive loss** helped the model distinguish noise by comparing to shuffled targets.

---

### 4.3. Training Enhancements  
We introduced several robustness-improving techniques:

**Data augmentation:**
```python
transforms.Compose([
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])
```

**Gradient clipping:**
Prevented exploding gradients under high noise  
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

**Learning rate scheduling:**
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
```

**Early stopping:** Patience = 10 to avoid overfitting noisy patterns

---

## 5. Hyperparameter Tuning with Optuna
To identify optimal architecture and training hyperparameters, we used Optuna and ran 20 trials.

### Search Space and Best Trial:

| Parameter         | Range           | Best Value (Trial 17) |
|------------------|------------------|------------------------|
| Learning Rate     | 1e-5 → 1e-3       | 0.000863               |
| Batch Size        | 32, 64, 128, 256  | 64                     |
| Bottleneck Size   | 512, 1024, 2048   | 512                    |
| Dropout Rate      | 0.1 → 0.5         | 0.314                  |
| Alpha (loss mix)  | 0.4 → 0.9         | 0.591                  |
| Weight Decay      | 1e-6 → 1e-4       | 3.49e-5                |

The final model was trained with early stopping but it still converged in 100 epochs.

---

## 6. How We Tackled B4B (Bucks for Buckets)  
B4B applies stochastic perturbations to mislead model stealing attacks. Our model's architecture, loss, and training were all crafted with defense awareness:

| B4B Behavior               | Our Countermeasure                                     |
|---------------------------|--------------------------------------------------------|
| Gaussian noise injection   | MSE + dropout + bottleneck width control              |
| Directional distortion     | Cosine loss + contrastive regularization              |
| Query-to-query instability | Data augmentation + LR restarts + GELU activations    |
| Gradient masking           | Gradient clipping + early stopping + smoother optimization |

## 7. Conclusion 
This project demonstrates that advanced defenses like B4B can be countered with a carefully designed, defense-aware training strategy. By combining compact architecture, hybrid loss functions, augmentation, and tuning, we replicated the protected encoder with high fidelity. Our pipeline generalizes to other perturbation-based defenses and inspires robust learning under noise.


https://github.com/abaadm1/TML25_A2_22/releases/tag/final-release



