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


def compute_centroids(z, in_y, num_classes=10):
    # print('24 ENTERING compute_centroids')
    # print('z', z.shape)
    # print('in_y', in_y.shape)
    true_y = torch.argmax(in_y, dim=1)
    # print('true_y', true_y.shape)
    class_mask = torch.nn.functional.one_hot(true_y, num_classes=num_classes).float()
    # print('class_mask', class_mask.shape)
    sum_z = torch.matmul(class_mask.T, z)
    # print('sum_z', sum_z.shape)
    count_per_class = class_mask.sum(dim=0)
    # print('count_per_class', count_per_class.shape)
    count_per_class = torch.clamp(count_per_class, min=1e-12)
    # print('count_per_class', count_per_class.shape)
    centroids = sum_z / count_per_class.unsqueeze(1)
    # print('centroids', centroids.shape)
    # Crop out the 
    return centroids


def update_learnt_centroids(learnt_y, centroids, decay_factor=1.0):
    # Extract latent dimensions and number of classes
    latent_dim = learnt_y.shape[1]
    num_classes = learnt_y.shape[0]
    # print('44 ENTERING update_learnt_centroids')

    # print('learnt_y', learnt_y.shape)
    # print('centroids', centroids.shape)
    # Create a mask to check if rows in centroids are all zeros
    nonzero_mask = torch.any(centroids != 0, dim=1)
    # print('nonzero_mask', nonzero_mask.shape)

    # Use the mask to update centroids: replace zero rows with corresponding rows from learnt_y
    updated_centroids = torch.where(
        nonzero_mask.unsqueeze(1),  # Expand mask to match the second dimension
        centroids,
        learnt_y
    )

    # Apply decay factor to blend centroids with learnt_y
    new_learnt_y = decay_factor * updated_centroids + (1 - decay_factor) * learnt_y

    return new_learnt_y


def cross_entropy_pull_loss(enc_x, in_y, learnt_y):
    # Compute pairwise distances between enc_x and learnt_y
    enc_x_dist = pairwise_dist(enc_x, learnt_y)

    # Compute logits by applying softmax to the negative distances
    logits = F.softmax(-1.0 * enc_x_dist, dim=1)

    # Cross-entropy loss with label smoothing
    cce = torch.nn.CrossEntropyLoss(label_smoothing=1e-6)

    # Compute the loss (input needs to be logits and class indices or probabilities)
    loss = cce(logits, in_y)

    return loss


def binary_cross_entropy_pull_loss(logits, labels):
    # Apply sigmoid activation to logits to get probabilities
    probs = torch.sigmoid(logits)
    
    # Compute BCE loss for each class independently
    bce_loss = F.binary_cross_entropy(probs, labels, reduction='none')
    
    # Take the mean over all classes and samples
    return bce_loss.mean()


def cross_entropy_nn_pred(enc_x, in_y, learnt_y):
    enc_x_to_learnt_y_dist = pairwise_dist(enc_x, learnt_y)
    logits = F.softmax(-1. * enc_x_to_learnt_y_dist, dim=1)
    preds = torch.argmax(logits, dim=1)
    true_y = torch.argmax(in_y, dim=1)
    return preds, true_y


def cos_repel_loss_z(z, in_y, num_labels):
    # Normalize the vectors
    norm_z = z / torch.norm(z, dim=1, keepdim=True)

    # Compute cosine distance (dot product of normalized vectors)
    cos_dist = torch.matmul(norm_z, norm_z.T)

    # Get the class labels (assumes one-hot encoded input)
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


class LearningWithAdaptiveLabels(nn.Module):
    """ BCE with optional one-hot from dense targets, label smoothing, thresholding
    NOTE for experiments comparing CE to BCE /w label smoothing, may remove
    """
    def __init__(
            self,
            latent_dim: int,
            num_classes: int,
            stationary_steps: int,
            current_step: int = 0,
            # BCE args
            # smoothing=0.1,
            # target_threshold: Optional[float] = None,
            # weight: Optional[torch.Tensor] = None,
            # reduction: str = 'mean',
            # sum_classes: bool = False,
            # pos_weight: Optional[Union[torch.Tensor, float]] = None,
    ):
        # print('TRACE: Initializing LWAL')
        super(LearningWithAdaptiveLabels, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.stationary_steps = stationary_steps
        self.current_step = current_step
        self.learnt_y = torch.eye(num_classes, latent_dim)

        # BCE inits
        # assert 0. <= smoothing < 1.0
        # if pos_weight is not None:
        #     if not isinstance(pos_weight, torch.Tensor):
        #         pos_weight = torch.tensor(pos_weight)
        # self.smoothing = smoothing
        # self.target_threshold = target_threshold
        # self.reduction = 'none' if sum_classes else reduction
        # self.sum_classes = sum_classes
        # self.register_buffer('weight', weight)
        # self.register_buffer('pos_weight', pos_weight)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # print('TRACE: Entering LWAL.forward()')
        # print('x', x.shape)
        # print('target', target.shape)
        batch_size = x.shape[0]
        assert batch_size == target.shape[0]

        # if target.shape != x.shape:
        #     # NOTE currently assume smoothing or other label softening is applied upstream if targets are already sparse
        #     num_classes = x.shape[-1]
        #     # FIXME should off/on be different for smoothing w/ BCE? Other impl out there differ
        #     off_value = self.smoothing / num_classes
        #     on_value = 1. - self.smoothing + off_value
        #     target = target.long().view(-1, 1)
        #     target = torch.full(
        #         (batch_size, num_classes),
        #         off_value,
        #         device=x.device, dtype=x.dtype).scatter_(1, target, on_value)

        # if self.target_threshold is not None:
        #     # Make target 0, or 1 if threshold set
        #     target = target.gt(self.target_threshold).to(dtype=target.dtype)

        # print('x/z', x.shape)
        # print('target/in_y', target.shape)

        # lwal loss is 10 * structure_loss + input_loss
        z = x.clone()
        num_labels = self.num_classes
        centroids = compute_centroids(z, target, self.num_classes)
        centroids = centroids.detach()
        self.learnt_y = update_learnt_centroids(self.learnt_y, centroids)
        self.current_step += 1

        input_loss = cross_entropy_pull_loss(x, target, self.learnt_y)
        structure_loss = cos_repel_loss_z(x, target, num_labels)
        em_loss = 10.0 * structure_loss + 1.0 * input_loss

        # BCE loss, keep for testing checkpoint #1
        # loss = F.binary_cross_entropy_with_logits(
        #     x, target,
        #     self.weight,
        #     pos_weight=self.pos_weight,
        #     reduction=self.reduction,
        # )
        # if self.sum_classes:
        #     loss = loss.sum(-1).mean()
        
        # print('TRACE: leaving LWAL.forward()', em_loss)
        return em_loss
