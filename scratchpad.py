from src.metrics.custom_metrics import CosineSimilarity
import torch
import numpy as np

y_pred = torch.randn(2, 3)
y_true = torch.randn(2, 3)
y = (y_pred, y_true)

y_pred_np = y_pred.numpy()
y_true_np = y_true.numpy()
dot_product_np = []
norm_np = []
cos_sim = []

print('Numpy y_pred:', y_pred_np)
print('Numpy y_true:', y_true_np)

for batch in range(y_true_np.shape[0]):
    dot_product_np.append(np.dot(y_pred_np[batch, :], y_true_np[batch, :]))

print('Numpy dot product:', dot_product_np)

for batch in range(y_true_np.shape[0]):
    norm_np.append(np.linalg.norm(y_pred_np[batch, :]) * np.linalg.norm(y_true_np[batch, :]))

print('Numpy norms:', norm_np)

for batch in range(y_true_np.shape[0]):
    cos_sim.append(dot_product_np[batch] / norm_np[batch])

print('Numpy cos sim:', cos_sim)

cos_sim = CosineSimilarity()
cos_sim.update(y)
cos_sim.compute()
