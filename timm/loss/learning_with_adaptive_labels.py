""" Binary Cross Entropy w/ a few extras

Hacked together by / Copyright 2021 Ross Wightman
"""
from typing import Optional, Union

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict

def pairwise_dist(A, B):
    na = torch.sum(A**2, dim=1)
    nb = torch.sum(B**2, dim=1)

    na = na.reshape(-1, 1)
    nb = nb.reshape(1, -1)

    D = torch.sqrt(torch.maximum(na - 2 * torch.matmul(A, B.T) + nb, torch.tensor(1e-12)))
    return D

def pairwise_cosine_similarity(A, B):
    # Get norms of the vectors
    A = A.to(torch.float16)
    B = B.to(torch.float16)
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

def update_learnt_centroids(learnt_y, centroids, decay_factor=1.0, norm_learnt_y: bool=False, exp_centroid_decay_factor=0.0):
    num_classes, latent_dim = learnt_y.shape  # Get dimensions
    new_learnt_y = []
    
    for i in range(num_classes):
        enc_y = centroids[i]
        if torch.count_nonzero(enc_y) == 0:  # Check if all zero
            enc_y = learnt_y[i]
        adj_decay_factor = decay_factor * math.exp(exp_centroid_decay_factor)
        new_enc_y = adj_decay_factor * enc_y + (1 - adj_decay_factor) * learnt_y[i]
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


def contrastive_loss(centroids):
    # criterion = nn.CrossEntropyLoss()
    # loss = criterion(centroids, centroids)
    # return loss.mean()
    detached_centroids = centroids.detach()
    contrastive_score = torch.einsum(
        "id, jd->ij",
        centroids, # / self.args.temperature,
        centroids,
        # detatched_centroids,
    )

    bsz = centroids.shape[0]
    labels = torch.arange(
        0, bsz, dtype=torch.long, device=contrastive_score.device
    )
    contrastive_loss = torch.nn.functional.cross_entropy(
        contrastive_score, labels
    )
    # print('contrastive_loss & score', contrastive_loss, contrastive_score)
    return contrastive_loss


def generate_random_orthogonal_vectors(num_classes, latent_dim, device, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    random_matrix = np.random.rand(latent_dim, num_classes)
    Q, _ = np.linalg.qr(random_matrix)
    orthogonal = torch.from_numpy(Q.T)
    orthogonal = orthogonal.to(device)
    return orthogonal


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
            structure_loss_weight: float = 1.0,
            init_fn: str = 'onehot',
            pairwise_fn: str = 'dist',
            num_features: int = 2048,
            verbose: bool = False,
            early_stop: bool = Optional[int],
            lwal_centroid_freeze_steps: Optional[int] = None,
            exp_centroid_decay_factor: float = 0.0,
            exp_stationary_step_decay_factor: float = 0.0,
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
        self.learnt_y = (
            generate_random_orthogonal_vectors(num_classes, latent_dim, device) 
            if init_fn == 'random' 
            else torch.eye(num_classes, latent_dim, device=device))
        print(self.learnt_y)
        self.decay_factor = decay_factor
        self.structure_loss_weight = structure_loss_weight
        self.pairwise_fn_name = pairwise_fn
        self.pairwise_fn = pairwise_cosine_similarity if pairwise_fn == 'cos' else pairwise_dist
        self.verbose = verbose
        self.early_stop = early_stop
        self.lwal_centroid_freeze_steps = lwal_centroid_freeze_steps
        self.exp_centroid_decay_factor = exp_centroid_decay_factor
        self.exp_stationary_step_decay_factor = exp_stationary_step_decay_factor
        self.maximum_element = 0
        self.maximum_norm = 0
        self.last_z_of_label = torch.zeros(num_classes, latent_dim, device=device)
    
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
        structure_loss = 0
        adj_stationary_steps = int(self.stationary_steps * math.exp(self.exp_stationary_step_decay_factor)
        update_centroids = (self.current_step % adj_stationary_steps) == 0)
        # For freezing experiment
        update_centroids = update_centroids and (self.lwal_centroid_freeze_steps is None or self.current_step <= self.lwal_centroid_freeze_steps)
        # For experiment C 
        # update_centroids = update_centroids and ((self.current_step // 195) > 19)
        if update_centroids:
            centroids = compute_centroids(x, target, self.num_classes)
            structure_loss = contrastive_loss(centroids)
            centroids = centroids.detach()
            self.learnt_y = update_learnt_centroids(self.learnt_y, centroids, self.decay_factor, self.pairwise_fn == 'cos', self.exp_centroid_decay_factor)
            # print(self.learnt_y)
            # structure_loss = cos_repel_loss_z_optimized(x, target)

        if self.early_stop and self.current_step == (self.early_stop*195):
            if self.verbose: 
                print('learnt_y (near the end of training)')
                print(self.learnt_y)
            print('pairwise cosine sim of learnt_y x learnt_y')
            print(pairwise_cosine_similarity(self.learnt_y, self.learnt_y))
            raise KeyboardInterrupt()
        
        self.maximum_element = max(self.maximum_element, get_max_element(z))
        self.maximum_norm = max(self.maximum_norm, get_max_element(calculate_vector_norms(z)))
        # Accuracy prints (every 50 steps)
        if (self.current_step % 5) == 1 and self.verbose: 
            print('train_acc @ %s steps' % self.current_step, self.acc_helper(z, target, self.learnt_y))
        # Experiment C
        # if (self.current_step // 195) == 19:
        #     # print(target)
        #     idx = torch.argmax(target, dim=-1)
        #     # self.last_z_of_label[idx] = z.detach()
        #     for i in range(z.size(0)):
        #         label = idx[i].item()
        #         self.last_z_of_label[label] = z[i].detach()
        self.current_step += 1
        # Experiment C
        # if self.current_step == 3901:
        #     print("Switching over centroids mode")
        #     self.learnt_y = self.last_z_of_label
        #     print("Centroids are: ", self.learnt_y)
        # # Print data every epoch.
        # if (self.current_step % 195) == 194 and self.verbose:
        #     print('z', self.maximum_element, self.maximum_norm, z)
        #     if self.pairwise_fn == pairwise_cosine_similarity:
        #         cossim = pairwise_cosine_similarity(normalize_tensor_vectors_vmap(z), self.learnt_y)
        #         print('cosine sim', 
        #             get_max_element(-cossim),
        #             get_max_element(calculate_vector_norms(-cossim)),
        #             cossim)
        #     else:
        #         dists = pairwise_dist(z, self.learnt_y)
        #         normed_dists = pairwise_dist(normalize_tensor_vectors_vmap(z), normalize_tensor_vectors_vmap(self.learnt_y))
        #         print('dists', 
        #               get_max_element(dists),
        #               get_max_element(calculate_vector_norms(dists)),
        #               dists)
        #         print('normed_dists', 
        #               get_max_element(normed_dists),
        #               get_max_element(calculate_vector_norms(normed_dists)),
        #               normed_dists)
        input_loss = self.cross_entropy_pull_loss(x, target, self.learnt_y)
        em_loss = self.structure_loss_weight * structure_loss + 1.0 * input_loss

        return em_loss, self.learnt_y
    
    def test(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor: 
        z = x.clone()
        self.device = x.device

        one_hot_target = torch.nn.functional.one_hot(target, num_classes=self.num_classes)
        # one_hot_target.to
        input_loss = self.cross_entropy_pull_loss(z, one_hot_target, self.learnt_y)
        # structure_loss = cos_repel_loss_z_optimized(z, one_hot_target)
        structure_loss = 0
        em_loss = self.structure_loss_weight * structure_loss + 1.0 * input_loss

        return em_loss

    def acc_helper(self, z, target, learnt_y):
        pred_y, true_y = self.cross_entropy_nn_pred(z, target, learnt_y)
        acc1 = (pred_y == true_y).float().mean() * 100.
        return acc1

    def accuracy(self, output, target, learnt_y, topk=(1,)):
        """Computes the 1-accuracy for lwal loss."""
        z = output.clone()
        z = z.to(torch.float32)
        one_hot_target = torch.nn.functional.one_hot(target, num_classes=self.num_classes)
        structure_loss = cos_repel_loss_z_optimized(z, one_hot_target)
        acc1 = self.acc_helper(z, one_hot_target, learnt_y)
        return acc1, structure_loss
