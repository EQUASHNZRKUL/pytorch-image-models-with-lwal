""" Binary Cross Entropy w/ a few extras

Hacked together by / Copyright 2021 Ross Wightman
"""
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_dist(A, B):
    na = torch.sum(A**2, dim=1)
    nb = torch.sum(B**2, dim=1)

    na = na.reshape(-1, 1)
    nb = nb.reshape(1, -1)

    D = torch.sqrt(torch.maximum(na - 2 * torch.matmul(A, B.T) + nb, torch.tensor(1e-12)))
    return D

def normalize_tensor_vectors_vmap(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")

    if tensor.ndim != 2:
        raise ValueError("Input tensor must be 2D.")

    def normalize_single_vector(vector):
        norm = torch.linalg.norm(vector)
        if norm == 0:
          raise ValueError("Vector with zero norm. Cannot normalize.")
        return vector / norm

    return torch.vmap(normalize_single_vector)(tensor)

def calculate_vector_norms(vectors):
    norms = torch.linalg.norm(vectors, dim=1, keepdim=True)
    return norms

def get_max_element(tensor):
    return torch.max(tensor)

def compute_centroids(z, in_y, num_classes=10):
    true_y = torch.argmax(in_y, dim=1)
    class_mask = torch.nn.functional.one_hot(true_y, num_classes=num_classes).float()
    sum_z = torch.matmul(class_mask.T, z)
    count_per_class = class_mask.sum(dim=0)
    count_per_class = torch.clamp(count_per_class, min=1e-12)
    centroids = sum_z / count_per_class.unsqueeze(1)
    return centroids


def update_learnt_centroids(learnt_y, centroids, decay_factor=1.0):
    nonzero_mask = torch.any(centroids != 0, dim=1)

    updated_centroids = torch.where(
        nonzero_mask.unsqueeze(1),  # Expand mask to match the second dimension
        centroids,
        learnt_y,
    )

    new_learnt_y = decay_factor * updated_centroids + (1 - decay_factor) * learnt_y
    # new_learnt_y = normalize_tensor_vectors_vmap(new_learnt_y)

    return new_learnt_y

def cross_entropy_pull_loss(enc_x, in_y, learnt_y):
    # Compute pairwise distances between enc_x and learnt_y
    enc_x_dist = pairwise_dist(enc_x, learnt_y)
    
    logits = F.log_softmax(-1.0 * enc_x_dist, dim=1)
    loss = torch.sum(-in_y * logits, dim=-1)
    return loss.mean()


def st_cce_forward(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
    return loss.mean()


def cos_repel_loss_z(z, in_y, num_labels):
    norm_z = z / torch.norm(z, dim=1, keepdim=True)
    cos_dist = torch.matmul(norm_z, norm_z.T)
    true_y = torch.argmax(in_y, dim=1).unsqueeze(1)

    # Create a mask for same-class pairs
    same_class_mask = torch.ones((in_y.shape[0], in_y.shape[0]), device=z.device, dtype=torch.float32)
    for i in range(num_labels):
        # Mask for class `i`
        true_y_i = (true_y == i).float()
        class_i_mask = 1 - torch.matmul(true_y_i, true_y_i.T)  # 0 if same class, 1 otherwise
        same_class_mask *= class_i_mask

    # Compute the loss: mean of cosine distances for different-class pairs
    return torch.mean(cos_dist * same_class_mask)


def cos_repel_loss_z_optimized(z, in_y):
    # Normalize the vectors
    norm_z = z / torch.norm(z, dim=1, keepdim=True)

    # Compute cosine similarity matrix
    cos_dist = torch.matmul(norm_z, norm_z.T)

    # Get the class labels (assumes one-hot encoded input)
    true_y = torch.argmax(in_y, dim=1)  # Shape: [batch_size]

    # Create a mask where same-class pairs are 0, and different-class pairs are 1
    true_y_expanded = true_y.unsqueeze(0)  # Shape: [1, batch_size]
    class_mask = (true_y_expanded != true_y_expanded.T).float()  # Shape: [batch_size, batch_size]

    # Compute the loss: mean of cosine distances for different-class pairs
    return torch.mean(cos_dist * class_mask)


def cross_entropy_nn_pred(enc_x, in_y, learnt_y):
    """Cross Entropy NN Prediction based on learnt_y."""

    enc_x_to_learnt_y_dist = pairwise_dist(enc_x, learnt_y)
    logits = F.softmax(-1. * enc_x_to_learnt_y_dist, dim=1)
    # print('logits', logits.shape)
    preds = torch.argmax(logits, dim=1)
    # print('in_y', in_y.shape)
    # print('in_y values', in_y)

    true_y = torch.argmax(in_y, dim=1)
    return preds, true_y


class LearningWithAdaptiveLabels(nn.Module):
    """ BCE with optional one-hot from dense targets, label smoothing, thresholding
    NOTE for experiments comparing CE to BCE /w label smoothing, may remove
    """
    def __init__(
            self,
            latent_dim: int,
            num_classes: int,
            stationary_steps: int,
            device: torch.device,
            current_step: int = 1,
            num_features: int = 2048,
            # BCE args
            # smoothing=0.1,
            # target_threshold: Optional[float] = None,
            # weight: Optional[torch.Tensor] = None,
            # reduction: str = 'mean',
            # sum_classes: bool = False,
            # pos_weight: Optional[Union[torch.Tensor, float]] = None,
    ):
        super(LearningWithAdaptiveLabels, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.stationary_steps = stationary_steps
        self.current_step = current_step
        self.learnt_y = torch.eye(num_classes, latent_dim, device=device)
        self.maximum_element = 0
        self.maximum_norm = 0
    
    def get_learnt_y(self):
        return self.learnt_y

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        assert batch_size == target.shape[0]

        z = x.clone()
        self.device = x.device
        num_labels = self.num_classes
        structure_loss=0.0
        if self.current_step % self.stationary_steps == 0:
            centroids = compute_centroids(z, target, self.num_classes)
            centroids = centroids.detach()
            # print('updating centroids')
            self.learnt_y = update_learnt_centroids(self.learnt_y, centroids)
            structure_loss = cos_repel_loss_z_optimized(x, target)
        self.current_step += 1

        if self.current_step == 4800:
            print('learnt_y (near the end of training)')
            print(self.learnt_y)
        self.maximum_element = max(self.maximum_element, get_max_element(z))
        self.maximum_norm = max(self.maximum_norm, get_max_element(calculate_vector_norms(z)))

        input_loss = cross_entropy_pull_loss(x, target, self.learnt_y)
        # input_loss = st_cce_forward(x, target)
        em_loss = 10.0 * structure_loss + 1.0 * input_loss
        # em_loss = input_loss

        return em_loss, self.learnt_y
    
    def test(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        assert batch_size == target.shape[0]

        print('max element', self.maximum_element)
        print('max norm', self.maximum_norm)
        one_hot_target = torch.nn.functional.one_hot(target, num_classes=10)

        input_loss = cross_entropy_pull_loss(x, one_hot_target, self.learnt_y)
        structure_loss = cos_repel_loss_z_optimized(x, one_hot_target)
        # input_loss = st_cce_forward(x, target)
        em_loss = 10.0 * structure_loss + 1.0 * input_loss
        # em_loss = input_loss

        return em_loss
    
    def accuracy(self, output, target, learnt_y, topk=(1,)):
        """Computes the 1-accuracy for lwal loss."""
        x = output.to(torch.float32)
        # x = self.fc(output)
        one_hot_target = torch.nn.functional.one_hot(target, num_classes=10)
        pred_y, true_y = cross_entropy_nn_pred(x, one_hot_target, learnt_y)

        acc1 = (pred_y == true_y).float().mean() * 100.
        return acc1, 0.0
