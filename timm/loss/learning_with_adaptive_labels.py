""" Binary Cross Entropy w/ a few extras

Hacked together by / Copyright 2021 Ross Wightman
"""
from typing import Optional, Union, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict

from timm.loss.lwal_constants import *

# tensor([[ 2.0538e+02,  1.4069e+02, -5.9971e+00,  4.2445e+01,  6.9692e+01, -3.7138e+01,  5.3473e+01, -2.6922e+01, -1.5774e+01, -1.8526e+02],
#         [ 1.4069e+02,  2.4068e+02,  2.4944e+01, -6.3079e+01,  9.4263e+01, -6.7743e+01,  6.7902e+01, -3.5117e+00,  1.2541e+02, -1.6306e+02],
#         [-5.9971e+00,  2.4944e+01,  2.7574e+01, -6.0725e+01,  7.7422e+00, -4.0184e+01,  6.2750e-01, -2.8379e+01,  2.7861e+01,  5.4217e+01],
#         [ 4.2445e+01, -6.3079e+01, -6.0725e+01,  2.2441e+02, -3.2279e+01, 1.3298e+02, -1.7046e+01,  1.0301e+02, -1.0480e+02, -3.4305e+02],
#         [ 6.9692e+01,  9.4263e+01,  7.7422e+00, -3.2279e+01,  4.6213e+01, -4.6051e+01,  3.9497e+01, -2.5225e+01,  5.1077e+01, -1.7917e+01],
#         [-3.7138e+01, -6.7743e+01, -4.0184e+01,  1.3298e+02, -4.6051e+01, 1.4322e+02, -7.1937e+01,  1.0362e+02, -7.7988e+01, -1.2365e+02],
#         [ 5.3473e+01,  6.7902e+01,  6.2750e-01, -1.7046e+01,  3.9497e+01, -7.1937e+01,  9.2068e+01, -3.7335e+01,  5.6542e+01, -1.3259e+02],
#         [-2.6922e+01, -3.5117e+00, -2.8379e+01,  1.0301e+02, -2.5225e+01, 1.0362e+02, -3.7335e+01,  1.4571e+02, -1.8481e+00, -2.0151e+02],
#         [-1.5774e+01,  1.2541e+02,  2.7861e+01, -1.0480e+02,  5.1077e+01, -7.7988e+01,  5.6542e+01, -1.8481e+00,  2.1198e+02, -1.0287e+02],
#         [-1.8526e+02, -1.6306e+02,  5.4217e+01, -3.4305e+02, -1.7917e+01, -1.2365e+02, -1.3259e+02, -2.0151e+02, -1.0287e+02,  1.6259e+03]])

def regular_simplex(n: int, d: int = None) -> np.ndarray:
    M = np.eye(n) - np.ones((n, n)) / n
    coords = M[:, :-1]  # shape (n, n-1)
    coords /= np.linalg.norm(coords[0])
    if d > n - 1:
        coords = np.hstack([coords, np.zeros((n, d - (n - 1)))])
    return torch.from_numpy(coords)

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

def perturb_embeddings_gaussian(learnt_y, eps = 0.01):
    noise = torch.randn_like(learnt_y) * eps
    noise = noise.abs()
    learnt_y = learnt_y + noise
    learnt_y = torch.nn.functional.normalize(learnt_y, p=2, dim=1)
    return learnt_y

def make_equally_spaced_embeddings(
    N: int, 
    d: int = None, 
    dot: float = None, 
    angle_deg: float = None,
    device = None):
    """
    Create N embedding vectors of dimension d that all have equal pairwise dot products.

    Args:
        N (int): number of vectors
        d (int, optional): dimensionality (default N)
        dot (float, optional): desired pairwise dot product between distinct vectors
        angle_deg (float, optional): alternatively specify angle in degrees

    Returns:
        X (torch.Tensor): [N, d] embedding vectors
        G (torch.Tensor): [N, N] Gram matrix (X @ X.T)
    """
    if (dot is None) == (angle_deg is None):
        raise ValueError("Specify exactly one of `dot` or `angle_deg`")

    if angle_deg is not None:
        dot = math.cos(math.radians(angle_deg))

    if d is None:
        d = N

    # For G to be positive semidefinite, dot >= -1/(N-1)
    if dot < -1 / (N - 1):
        raise ValueError(f"Invalid dot={dot:.3f}: too negative for N={N} (must be ≥ {-1/(N-1):.3f})")

    # Construct Gram matrix
    I = torch.eye(N)
    ones = torch.ones((N, N))
    G = (1 - dot) * I + dot * ones

    # Eigen-decompose G = V Λ Vᵀ and construct X = V √Λ
    eigvals, eigvecs = torch.linalg.eigh(G)
    eigvals_clamped = torch.clamp(eigvals, min=0.0)
    X_full = eigvecs @ torch.diag(torch.sqrt(eigvals_clamped))

    # Truncate or pad to desired dimensionality
    if d < N:
        X = X_full[:, :d]
    elif d > N:
        pad = torch.zeros((N, d - N))
        X = torch.cat([X_full, pad], dim=1)
    else:
        X = X_full

    return X.to(device)

def make_rotated_onehot(N=10, rotate_pair=(1, 5), angle_deg=10.0, device = None):
    """
    Create N one-hot embeddings, except rotate one of them by a given angle
    relative to another.

    Args:
        N (int): number of embeddings / dimensions
        rotate_pair (tuple[int, int]): (anchor_idx, rotated_idx)
        angle_deg (float): rotation angle in degrees

    Returns:
        torch.Tensor: [N, N] embeddings (each row is a vector)
    """
    angle = math.radians(angle_deg)
    X = torch.eye(N)  # start with one-hot vectors

    i, j = rotate_pair
    e_i = X[i].clone()
    e_j = X[j].clone()

    # Rotate e_j toward e_i by the desired angle
    X[j] = math.cos(angle) * e_j + math.sin(angle) * e_i

    # Renormalize to keep unit length (just in case)
    X = X / X.norm(dim=1, keepdim=True)
    return X.to(device)


def make_two_angle_embeddings(N=10, dim=10, n1=5, angle_pair = (5, 20), device=None, seed=None):
    """
    Create embeddings such that:
      - first n1 vectors are 'angle1_deg' away from e_0
      - remaining N-n1 vectors are 'angle2_deg' away from e_{dim-1}

    Args:
        N (int): total number of embeddings
        dim (int): embedding dimensionality
        n1 (int): number of vectors near e_0
        angle1_deg (float): angle for first group
        angle2_deg (float): angle for second group
        seed (int, optional): RNG seed for reproducibility

    Returns:
        torch.Tensor: [N, dim] embeddings (rows)
    """
    angle1_deg, angle2_deg = angle_pair
    if seed is not None:
        torch.manual_seed(seed)

    X = torch.zeros((N, dim))
    e0 = torch.zeros(dim); e0[0] = 1.0
    e9 = torch.zeros(dim); e9[-1] = 1.0

    # group 1: near e0
    for i in range(n1):
        v = torch.randn(dim)
        v[0] = 0.0  # make sure it's orthogonal to e0
        v = v / v.norm()
        angle = math.radians(angle1_deg)
        X[i] = math.cos(angle) * e0 + math.sin(angle) * v

    # group 2: near e9
    for i in range(n1, N):
        v = torch.randn(dim)
        v[-1] = 0.0
        v = v / v.norm()
        angle = math.radians(angle2_deg)
        X[i] = math.cos(angle) * e9 + math.sin(angle) * v

    # normalize all to unit length (for safety)
    X = X / X.norm(dim=1, keepdim=True)
    return X.to(device)


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
            averaging_centroids: bool = False,
            sigma: Optional[float] = None,
            dot: Optional[float] = None,
            ang_deg: Optional[float] = None,
            rotate_pair: Tuple[int, int] = (0, 1),
            angle_pair: Tuple[int, int] = (5, 20)
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
        self.learnt_y = None
        self.device = device
        match init_fn:
            case 'random':
                self.learnt_y = generate_random_orthogonal_vectors(num_classes, latent_dim, device) 
            case 'perturbed':
                self.learnt_y = torch.eye(num_classes, latent_dim, device=device)
            case 'angled':
                self.learnt_y = make_equally_spaced_embeddings(
                    num_classes, latent_dim, dot, ang_deg, device
                )
            case 'single_angled':
                self.learnt_y = make_rotated_onehot(
                    num_classes, rotate_pair=rotate_pair, angle_deg=ang_deg, device=device
                )
            case 'angled_groups':
                self.learnt_y = make_two_angle_embeddings(
                    N=num_classes,
                    dim=latent_dim,
                    n1=5,
                    angle_pair=angle_pair,
                    device=device
                )
            case 'learnt':
                self.learnt_y = LAST_Z_OF_LABEL.to(device)
            case 'vit':
                self.learnt_y = VIT_Z_OF_LABEL.to(device)
            case 'simplex':
                self.learnt_y = regular_simplex(n=num_classes, d=latent_dim).to(device)
            case _:
                self.learnt_y = torch.eye(num_classes, latent_dim, device=device)
        if sigma is not None and sigma > 0:
            self.learnt_y = perturb_embeddings_gaussian(self.learnt_y, sigma)
            print('pairwise cosine sim of learnt_y x learnt_y after perturbation.')
            print(pairwise_cosine_similarity(self.learnt_y, self.learnt_y))
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
        self.averaging_centroids = averaging_centroids
        self.learnt_y_sums = torch.zeros(num_classes, latent_dim, device=device)
        self.learnt_y_counts = torch.zeros(num_classes, device=device)
    
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
        stationary_steps_adj = self.stationary_steps

        if self.averaging_centroids:
            target_clone = target.detach()
            class_indices = torch.argmax(target_clone, dim=-1)
            self.learnt_y_sums.index_add_(0, class_indices, z.to(torch.float32).detach())
            self.learnt_y_counts.index_add_(
                0, 
                class_indices, 
                torch.ones_like(class_indices, dtype=torch.float)
            )

        update_centroids = (self.current_step % int(self.stationary_steps) == 0)
        # For freezing experiment
        update_centroids = update_centroids and (self.lwal_centroid_freeze_steps is None or self.current_step <= self.lwal_centroid_freeze_steps)
        # For experiment C 
        # update_centroids = update_centroids and ((self.current_step // 195) > 19)
        if update_centroids and self.averaging_centroids:
            # centroids = compute_centroids()
            centroids = self.learnt_y_sums / self.learnt_y_counts.unsqueeze(1)
            self.learnt_y_sums = torch.zeros(self.num_classes, self.latent_dim, device=self.device)
            self.learnt_y_counts = torch.zeros(self.num_classes, device=self.device)
            structure_loss = contrastive_loss(centroids)
            centroids = centroids.detach()
            decay_factor_adj = self.decay_factor * math.exp(self.current_step / self.stationary_steps * self.exp_centroid_decay_factor)
            self.learnt_y = update_learnt_centroids(self.learnt_y, centroids, decay_factor_adj, self.pairwise_fn == 'cos', self.exp_centroid_decay_factor)
        elif update_centroids:
            centroids = compute_centroids(x, target, self.num_classes)
            structure_loss = contrastive_loss(centroids)
            centroids = centroids.detach()
            decay_factor_adj = self.decay_factor * math.exp(self.current_step / self.stationary_steps * self.exp_centroid_decay_factor)
            self.learnt_y = update_learnt_centroids(self.learnt_y, centroids, decay_factor_adj, self.pairwise_fn == 'cos', self.exp_centroid_decay_factor)
            # print(self.learnt_y)
            # structure_loss = cos_repel_loss_z_optimized(x, target)
            # self.stationary_steps *= math.exp(self.exp_stationary_step_decay_factor)
            # self.decay_factor *= math.exp(self.exp_centroid_decay_factor)
            # print('new stationary_steps and decay_factor', stationary_steps_adj, decay_factor_adj)

        if self.early_stop and self.current_step == (self.early_stop*195):
            if self.verbose: 
                print('learnt_y (near the end of training)')
                print(self.learnt_y)
            pwise = pairwise_cosine_similarity(self.learnt_y, self.learnt_y)
            print('pairwise cosine sim of learnt_y x learnt_y')
            print(pwise)
            print(pwise.mean())
            print("last z's of each label (to be used as centroids for next run)")
            print(self.last_z_of_label)
            raise KeyboardInterrupt()
        
        self.maximum_element = max(self.maximum_element, get_max_element(z))
        self.maximum_norm = max(self.maximum_norm, get_max_element(calculate_vector_norms(z)))
        # Accuracy prints (every 50 steps)
        if (self.current_step % 5) == 1 and self.verbose: 
            print('train_acc @ %s steps' % self.current_step, self.acc_helper(z, target, self.learnt_y))
        # Experiment C
        if (self.current_step // 195) == 19:
            # print(target)
            idx = torch.argmax(target, dim=-1)
            # self.last_z_of_label[idx] = z.detach()
            for i in range(z.size(0)):
                label = idx[i].item()
                self.last_z_of_label[label] = z[i].detach()
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

        # Normalize to class indices
        if true_y.ndim > 1:
            true_y = true_y.argmax(dim=1)
        if pred_y.ndim > 1:
            pred_y = pred_y.argmax(dim=1)

        acc1 = (pred_y == true_y).float().mean() * 100.

        num_classes = self.num_classes
        per_class_correct = torch.zeros(num_classes, device=z.device)
        per_class_total = torch.zeros(num_classes, device=z.device)

        for c in range(num_classes):
            mask = (true_y == c)
            per_class_total[c] = mask.sum()
            per_class_correct[c] = (pred_y[mask] == c).sum()

        return acc1, per_class_correct, per_class_total

    def accuracy(self, output, target, learnt_y, topk=(1,)):
        """Computes the 1-accuracy for lwal loss."""
        z = output.clone()
        z = z.to(torch.float32)
        one_hot_target = torch.nn.functional.one_hot(target, num_classes=self.num_classes)
        structure_loss = cos_repel_loss_z_optimized(z, one_hot_target)
        acc1 = self.acc_helper(z, one_hot_target, learnt_y)
        return acc1, structure_loss

    def accuracy_with_per_class(self, output, target, learnt_y, topk=(1,)):
        z = output.clone().to(torch.float32)
        one_hot_target = torch.nn.functional.one_hot(target, num_classes=self.num_classes)
        structure_loss = cos_repel_loss_z_optimized(z, one_hot_target)
        acc1, per_class_correct, per_class_total = self.acc_helper(z, one_hot_target, learnt_y)
        return acc1, structure_loss, per_class_correct, per_class_total
    
    def accuracy_with_confusion_matrix(self, output, target, learnt_y, confmat_metric=None):
        # Get predictions and labels once
        one_hot_target = torch.nn.functional.one_hot(target, num_classes=self.num_classes)
        pred_y, true_y = self.cross_entropy_nn_pred(output, target, learnt_y)

        if true_y.ndim > 1:
            true_y = true_y.argmax(dim=1)
        if pred_y.ndim > 1:
            pred_y = pred_y.argmax(dim=1)

        # Compute accuracy
        acc1 = (pred_y == true_y).float().mean() * 100.0

        # Per-class counts
        num_classes = self.num_classes
        per_class_correct = torch.zeros(num_classes, device=output.device)
        per_class_total = torch.zeros(num_classes, device=output.device)
        for c in range(num_classes):
            mask = (true_y == c)
            per_class_total[c] = mask.sum()
            per_class_correct[c] = (pred_y[mask] == c).sum()

        # Update confusion matrix if provided
        if confmat_metric is not None:
            confmat_metric.update(pred_y, true_y)

        # Optional structure loss, if you’re using it
        z = output.to(torch.float32)
        one_hot_target = torch.nn.functional.one_hot(target, num_classes=num_classes)
        structure_loss = cos_repel_loss_z_optimized(z, one_hot_target)

        return acc1, structure_loss