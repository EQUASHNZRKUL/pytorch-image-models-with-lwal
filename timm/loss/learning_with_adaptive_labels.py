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

def pairwise_cosine_similarity(A, B):
    # Get norms of the vectors
    A_norm = torch.linalg.norm(A, dim=1, keepdim=True)
    B_norm = torch.linalg.norm(B, dim=1, keepdim=True)

    # # Avoid division by zero
    # A_norm = torch.where(A_norm == 0, torch.tensor(1e-12, device=A.device), A_norm)
    # B_norm = torch.where(B_norm == 0, torch.tensor(1e-12, device=B.device), B_norm)

    A_normalized = A / A_norm
    B_normalized = B / B_norm

    # Calculate cosine similarity
    # print('pairwise shapes', A_normalized.shape, B_normalized.T.shape)
    # print('pairwise devices', A.device, B.device)
    similarity = torch.matmul(A_normalized, B_normalized.T)
    return 1-similarity

def normalize_tensor_vectors_vmap(tensor):
    return tensor / torch.linalg.norm(tensor, dim=1, keepdim=True)

def calculate_vector_norms(vectors):
    norms = torch.linalg.norm(vectors, dim=1, keepdim=True)
    return norms

def get_max_element(tensor):
    return torch.max(tensor)

def compute_centroids_new(z, in_y, num_classes=10):
    true_y = torch.argmax(in_y, dim=1)
    class_mask = torch.nn.functional.one_hot(true_y, num_classes=num_classes).float()
    sum_z = torch.matmul(class_mask.T, z)
    count_per_class = class_mask.sum(dim=0)
    count_per_class = torch.clamp(count_per_class, min=1e-12)
    centroids = sum_z / count_per_class.unsqueeze(1)
    return centroids


def compute_centroids(z, in_y, num_classes=10):
    true_y = torch.argmax(in_y, dim=1)
    centroids = []
    
    for i in range(num_classes):
        class_i_mask = (true_y == i).float().unsqueeze(1)  # Create mask
        num_class_i = class_i_mask.sum()
        
        if num_class_i == 0:
            centroids.append(torch.zeros(z.shape[1], device=z.device))
        else:
            class_i_mask = torch.ones_like(z) * class_i_mask
            masked_z_i = z * class_i_mask
            centroid_i = masked_z_i.sum(dim=0) / num_class_i
            centroids.append(centroid_i)
    
    return torch.stack(centroids)


def update_learnt_centroids_new(learnt_y, centroids, decay_factor=1.0, norm_learnt_y: bool=False):
    nonzero_mask = torch.any(centroids != 0, dim=1)

    updated_centroids = torch.where(
        nonzero_mask.unsqueeze(1),  # Expand mask to match the second dimension
        centroids,
        learnt_y,
    )

    new_learnt_y = decay_factor * updated_centroids + (1 - decay_factor) * learnt_y
    if norm_learnt_y:
        new_learnt_y = normalize_tensor_vectors_vmap(new_learnt_y)

    return new_learnt_y

def update_learnt_centroids(learnt_y, centroids, decay_factor=1.0, norm_learnt_y: bool=False):
    num_classes, latent_dim = learnt_y.shape  # Get dimensions
    new_learnt_y = []
    
    for i in range(num_classes):
        enc_y = centroids[i]
        if torch.count_nonzero(enc_y) == 0:  # Check if all zero
            enc_y = learnt_y[i]
        new_enc_y = decay_factor * enc_y + (1 - decay_factor) * learnt_y[i]
        new_learnt_y.append(new_enc_y)

    if norm_learnt_y:
        new_learnt_y = normalize_tensor_vectors_vmap(new_learnt_y)
    
    return torch.stack(new_learnt_y)


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
    # abs_cos_dist = torch.relu(cos_dist)
    return torch.mean(cos_dist * same_class_mask)


def cos_repel_loss_z_optimized(z, in_y):
    # Normalize the vectors
    norm_z = z / torch.norm(z, dim=1, keepdim=True)

    # Compute cosine similarity matrix
    cos_dist = torch.matmul(norm_z, norm_z.T)
    # adj_cos_dist = torch.relu(cos_dist)
    # p_dist = pairwise_dist(norm_z, norm_z)
    # learnt_y_dist = torch.relu(-torch.matmul(z, z))

    # Get the class labels (assumes one-hot encoded input)
    true_y = torch.argmax(in_y, dim=1)  # Shape: [batch_size]

    # Create a mask where same-class pairs are 0, and different-class pairs are 1
    true_y_expanded = true_y.unsqueeze(0)  # Shape: [1, batch_size]
    class_mask = (true_y_expanded != true_y_expanded.T).float()  # Shape: [batch_size, batch_size]

    # Compute the loss: mean of cosine distances for different-class pairs
    return torch.mean(cos_dist * class_mask)


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
            decay_factor: float = 1.0,
            structure_loss_weight: float = 10.0,
            pairwise_fn: str = 'dist',
            num_features: int = 2048,
            verbose: bool = False,
            early_stop: bool = Optional[int],
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
        self.decay_factor = decay_factor
        self.structure_loss_weight = structure_loss_weight
        self.pairwise_fn_name = pairwise_fn
        self.pairwise_fn = pairwise_cosine_similarity if pairwise_fn == 'cos' else pairwise_dist
        self.verbose = verbose
        self.early_stop = early_stop
        self.maximum_element = 0
        self.maximum_norm = 0
    
    def get_learnt_y(self):
        return self.learnt_y

    def cross_entropy_pull_loss(self, enc_x, in_y, learnt_y):
        enc_x_dist = self.pairwise_fn(enc_x, learnt_y)
        logits = F.log_softmax(-1.0 * enc_x_dist, dim=1)
        loss = torch.sum(-in_y * logits, dim=-1)
        return loss.mean()

    def cross_entropy_nn_pred(self, enc_x, in_y, learnt_y):
        """Cross Entropy NN Prediction based on learnt_y."""
        enc_x_dist = self.pairwise_fn(enc_x, learnt_y)
        logits = F.log_softmax(-1.0 * enc_x_dist, dim=1)
        preds = torch.argmax(logits, dim=1)
        true_y = torch.argmax(in_y, dim=1)
        return preds, true_y

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        assert batch_size == target.shape[0]

        z = x.clone()
        self.device = x.device
        num_labels = self.num_classes
        if self.current_step % self.stationary_steps == 0:
            centroids = compute_centroids(z, target, self.num_classes)
            centroids = centroids.detach()
            # print('updating centroids')
            self.learnt_y = update_learnt_centroids(self.learnt_y, centroids, self.decay_factor, self.pairwise_fn == 'cos')
        self.current_step += 1

        if self.early_stop and self.current_step == (self.early_stop*195):
            if self.verbose: 
                print('learnt_y (near the end of training)')
                print(self.learnt_y)
            print('pairwise cosine sim of learnt_y x learnt_y')
            print(pairwise_cosine_similarity(self.learnt_y, self.learnt_y))
            raise KeyboardInterrupt()
        
        self.maximum_element = max(self.maximum_element, get_max_element(z))
        self.maximum_norm = max(self.maximum_norm, get_max_element(calculate_vector_norms(z)))
        if (self.current_step % 195) == 194 and self.verbose:
            print('z', self.maximum_element, self.maximum_norm, z)
            # print('learnt_y', 
            #       get_max_element(self.learnt_y), 
            #       get_max_element(calculate_vector_norms(self.learnt_y)), 
            #       self.learnt_y)
            if self.pairwise_fn == pairwise_cosine_similarity:
                cossim = pairwise_cosine_similarity(normalize_tensor_vectors_vmap(z), self.learnt_y)
                print('cosine sim', 
                    get_max_element(-cossim),
                    get_max_element(calculate_vector_norms(-cossim)),
                    cossim)
            else:
                dists = pairwise_dist(z, self.learnt_y)
                normed_dists = pairwise_dist(normalize_tensor_vectors_vmap(z), normalize_tensor_vectors_vmap(self.learnt_y))
                print('dists', 
                      get_max_element(dists),
                      get_max_element(calculate_vector_norms(dists)),
                      dists)
                print('normed_dists', 
                      get_max_element(normed_dists),
                      get_max_element(calculate_vector_norms(normed_dists)),
                      normed_dists)

        structure_loss = cos_repel_loss_z_optimized(x, target)
        input_loss = self.cross_entropy_pull_loss(x, target, self.learnt_y)
        em_loss = self.structure_loss_weight * structure_loss + 1.0 * input_loss

        return em_loss, self.learnt_y
    
    def test(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor: 
        z = x.clone()
        self.device = x.device

        one_hot_target = torch.nn.functional.one_hot(target, num_classes=self.num_classes)
        # one_hot_target.to
        input_loss = self.cross_entropy_pull_loss(z, one_hot_target, self.learnt_y)
        structure_loss = cos_repel_loss_z_optimized(z, one_hot_target)
        em_loss = self.structure_loss_weight * structure_loss + 1.0 * input_loss

        return em_loss
    
    def accuracy(self, output, target, learnt_y, topk=(1,)):
        """Computes the 1-accuracy for lwal loss."""
        z = output.clone()
        z = z.to(torch.float32)
        # x = self.fc(output)
        one_hot_target = torch.nn.functional.one_hot(target, num_classes=self.num_classes)
        pred_y, true_y = self.cross_entropy_nn_pred(z, one_hot_target, learnt_y)
        structure_loss = cos_repel_loss_z_optimized(z, one_hot_target)

        acc1 = (pred_y == true_y).float().mean() * 100.
        return acc1, structure_loss
