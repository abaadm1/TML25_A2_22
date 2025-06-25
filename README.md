## Introduction
In this assignment, we successfully replicated the behavior of a protected machine learning encoder using model stealing techniques, despite the presence of B4B (Bucks for Buckets) defense. Our final surrogate encoder achieved a competitive L2 distance of 4.706. This report provides a thorough, step-by-step explanation of our methodology, the difficulties we encountered, and how each design choice assisted in resolving them.
## Starting Strategy and Early Challenges
## 2.1 Dataset & API Query Strategy
We first determined that the provided dataset (ModelStealingPub.pt) contained exactly 13,000 images. In the surrogate dataset approach, we start by shuffling the entire dataset. Then, we run 13 separate queries, each retrieving 1,000 samples, resulting in a final dataset of 13,000 samples arranged in the order the queries were made.Each query returned 1024-dimensional embeddings from the victim encoder:
```python
embeddings = model_stealing(batch_images, port=PORT)

