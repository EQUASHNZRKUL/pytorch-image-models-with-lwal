""" Binary Cross Entropy w/ a few extras

Hacked together by / Copyright 2021 Ross Wightman
"""
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorflow as tf
import numpy as np

class TimmLwal():
    def __init__(self, current_step=0, stationary_steps=0, learnt_y=None, num_classes=4):
        self.current_step=current_step
        self.stationary_steps=stationary_steps
        self.learnt_y=learnt_y
        self.num_classes=num_classes

    def pairwise_dist(self, A, B):
        na = torch.sum(A**2, dim=1)
        nb = torch.sum(B**2, dim=1)

        na = na.reshape(-1, 1)
        nb = nb.reshape(1, -1)

        D = torch.sqrt(torch.maximum(na - 2 * torch.matmul(A, B.T) + nb, torch.tensor(1e-12)))
        return D
    
    def pairwise_cosine_similarity(self, A, B):
        # Get norms of the vectors
        A_norm = torch.linalg.norm(A, dim=1, keepdim=True)
        B_norm = torch.linalg.norm(B, dim=1, keepdim=True)

        # # Avoid division by zero
        # A_norm = torch.where(A_norm == 0, torch.tensor(1e-12, device=A.device), A_norm)
        # B_norm = torch.where(B_norm == 0, torch.tensor(1e-12, device=B.device), B_norm)

        A_normalized = A / A_norm
        B_normalized = B / B_norm

        # Calculate cosine similarity
        similarity = torch.matmul(A_normalized, B_normalized.T)
        return 1-similarity

    def compute_centroids(self, z, in_y, num_classes=10):
        true_y = torch.argmax(in_y, dim=1)
        class_mask = torch.nn.functional.one_hot(true_y, num_classes=num_classes).float()
        sum_z = torch.matmul(class_mask.T, z)
        count_per_class = class_mask.sum(dim=0)
        count_per_class = torch.clamp(count_per_class, min=1e-12)
        centroids = sum_z / count_per_class.unsqueeze(1)
        return centroids

    def update_learnt_centroids_new(self, learnt_y, centroids, decay_factor=1.0):
        nonzero_mask = torch.any(centroids != 0, dim=1)

        updated_centroids = torch.where(
            nonzero_mask.unsqueeze(1),  # Expand mask to match the second dimension
            centroids,
            learnt_y,
        )

        new_learnt_y = decay_factor * updated_centroids + (1 - decay_factor) * learnt_y

        return new_learnt_y

    def update_learnt_centroids(self, learnt_y, centroids, decay_factor=1.0):
        num_classes, latent_dim = learnt_y.shape  # Get dimensions
        new_learnt_y = []
        
        for i in range(num_classes):
            enc_y = centroids[i]
            if torch.count_nonzero(enc_y) == 0:  # Check if all zero
                enc_y = learnt_y[i]
            new_enc_y = decay_factor * enc_y + (1 - decay_factor) * learnt_y[i]
            new_learnt_y.append(new_enc_y)
        
        return torch.stack(new_learnt_y)

    def cross_entropy_pull_loss(self, enc_x, in_y, learnt_y):
        # Compute pairwise distances between enc_x and learnt_y
        enc_x_dist = self.pairwise_dist(enc_x, learnt_y)

        # Compute logits by applying softmax to the negative distances
        logits = F.log_softmax(-1.0 * enc_x_dist, dim=1)

        # Cross-entropy loss with label smoothing
        loss = torch.sum(-in_y * logits, dim=-1)
        return loss.mean()

    def binary_cross_entropy_pull_loss(self, enc_x, in_y, learnt_y):
        enc_x_dist = self.pairwise_dist(enc_x, learnt_y)
        logits = F.softmax(-1.0 * enc_x_dist, dim=1)
        bce_loss = F.binary_cross_entropy_with_logits(logits, in_y, reduction='none')
        return bce_loss.mean()

    def cos_repel_loss_z(self, z, in_y, num_labels):
        norm_z = z / torch.norm(z, dim=1, keepdim=True)
        cos_dist = torch.matmul(norm_z, norm_z.T)
        abs_cos_dist = torch.relu(-cos_dist)
        true_y = torch.argmax(in_y, dim=1).unsqueeze(1)
        same_class_mask = torch.ones((in_y.shape[0], in_y.shape[0]), device=z.device, dtype=torch.float32)
        for i in range(num_labels):
            # Mask for class `i`
            true_y_i = (true_y == i).float()
            class_i_mask = 1 - torch.matmul(true_y_i, true_y_i.T)  # 0 if same class, 1 otherwise
            same_class_mask *= class_i_mask
        return torch.mean(abs_cos_dist * same_class_mask)

    def cos_repel_loss_z_optimized(self, z, in_y):
        norm_z = z / torch.norm(z, dim=1, keepdim=True)
        cos_dist = torch.matmul(norm_z, norm_z.T)
        abs_cos_dist = torch.relu(cos_dist)
        true_y = torch.argmax(in_y, dim=1)  # Shape: [batch_size]
        true_y_expanded = true_y.unsqueeze(0)  # Shape: [1, batch_size]
        class_mask = (true_y_expanded != true_y_expanded.T).float()  # Shape: [batch_size, batch_size]
        return torch.mean(abs_cos_dist * class_mask)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        assert batch_size == target.shape[0]
        # # x = self.fc(x)

        # # lwal loss is 10 * structure_loss + input_loss
        z = x.clone()
        self.device = x.device
        num_labels = self.num_classes
        structure_loss=0.0
        if self.current_step % self.stationary_steps == 0:
            centroids = self.compute_centroids(z, target, self.num_classes)
            centroids = centroids.detach()
            self.learnt_y = self.update_learnt_centroids(self.learnt_y, centroids)
            structure_loss = self.cos_repel_loss_z_optimized(x, target)
        self.current_step += 1

        input_loss = self.cross_entropy_pull_loss(x, target, self.learnt_y)
        # input_loss = st_cce_forward(x, target)
        # em_loss = 10.0 * structure_loss + 1.0 * input_loss
        em_loss = input_loss

        return em_loss
    
    def cross_entropy_nn_pred(self, enc_x, in_y, learnt_y):
        """Cross Entropy NN Prediction based on learnt_y."""
        enc_x_dist = self.pairwise_dist(enc_x, learnt_y)
        logits = F.log_softmax(-1.0 * enc_x_dist, dim=1)
        preds = torch.argmax(logits, dim=1)
        print('torch preds', logits)
        true_y = torch.argmax(in_y, dim=1)
        print('torch true', in_y)
        return preds, true_y

    def accuracy(self, output, target, learnt_y, topk=(1,)):
        """Computes the 1-accuracy for lwal loss."""
        z = output.clone()
        z = z.to(torch.float32)
        # x = self.fc(output)
        # one_hot_target = torch.nn.functional.one_hot(target, num_classes=10)
        pred_y, true_y = self.cross_entropy_nn_pred(z, target, learnt_y)

        acc1 = (pred_y == true_y).float().mean() * 100.
        return acc1, 0.0

class XiaoLwal():
    def __init__(self, current_step=0, warmup_steps=0, stationary_steps=0, rloss=None, learnt_y=None, num_classes=10):
        self.current_step=current_step
        self.stationary_steps=stationary_steps
        self.warmup_steps=warmup_steps
        self.rloss=rloss
        self.learnt_y=learnt_y
        self.num_classes=num_classes

    def pairwise_dist(self, A, B):
        na = tf.reduce_sum(tf.square(A), 1)
        nb = tf.reduce_sum(tf.square(B), 1)

        na = tf.reshape(na, [-1, 1])
        nb = tf.reshape(nb, [1, -1])

        D = tf.sqrt(tf.maximum(na - 2 * tf.matmul(A, B, False, True) + nb, 1e-12))
        return D
    
    def compute_centroids(self, z, in_y, num_classes=10):
        true_y = tf.argmax(in_y, axis=1)
        centroids = []
        for i in range(num_classes): # num classes may not be equal to latend dim
            class_i_mask = tf.cast(tf.expand_dims(true_y, axis=1) == i, tf.float32)
            num_class_i = tf.reduce_sum(class_i_mask)
            if num_class_i == 0:
                centroids.append(tf.zeros([z.shape[1]]))
            else:
                class_i_mask = tf.math.multiply(tf.ones_like(z), class_i_mask)
                masked_z_i = tf.math.multiply(z, class_i_mask)
                centroid_i = tf.reduce_sum(masked_z_i, axis=0) / num_class_i
                centroids.append(centroid_i)
        return tf.stack(centroids)


    def update_learnt_centroids(self, learnt_y, centroids, decay_factor=1.0):
        latent_dim = learnt_y.shape[1] 
        num_classes = learnt_y.shape[0] # this is always correct
        new_learnt_y = []
        for i in range(num_classes):
            enc_y = centroids[i]
            if tf.math.count_nonzero(enc_y) == 0: # check if all zero
                enc_y = learnt_y[i]
            new_enc_y = decay_factor * enc_y + (1 - decay_factor) * learnt_y[i]
            new_learnt_y.append(new_enc_y)
        return tf.stack(new_learnt_y)

    def cos_repel_loss_z(self, z, in_y, num_labels):
        norm_z = z/tf.norm(z, axis=1, keepdims=True)
        cos_dist = norm_z @ tf.transpose(norm_z)
        print('cos_dist', cos_dist)
        abs_cos_dist = tf.nn.relu(cos_dist)
        print('abs_cos_dist', abs_cos_dist)

        # we only penalize vectors that are in differnt classes
        true_y = tf.argmax(in_y, axis=1)
        true_y_ = tf.expand_dims(true_y, axis=1)
        print('in_y', in_y)
        print('true_y', true_y)
        print('true_y_', true_y_)
        same_class_mask = tf.ones((in_y.shape[0], in_y.shape[0]), dtype=tf.float32)
        print('same_class_mask before', same_class_mask)
        for i in range(num_labels):
            true_y_i = tf.cast(true_y_ == i, dtype=tf.float32)
            class_i_mask = 1 - (true_y_i @ tf.transpose(true_y_i)) # 0 if same class, 1 otherwise
            same_class_mask *= class_i_mask
        
        print('same_class_mask after', same_class_mask)
        return tf.reduce_mean(abs_cos_dist * same_class_mask)


    def ce_pull_loss(self, enc_x, in_y, learnt_y):
        cce = tf.keras.losses.CategoricalCrossentropy(label_smoothing=1e-6, from_logits=False) # label_smoothing
        enc_x_dist = self.pairwise_dist(enc_x, learnt_y) # + 1e-6 no longer needed since we added 1e-10 when doing pairwise_dist
        probs = tf.nn.softmax(-1. * enc_x_dist, axis=1)  # the closer the better
        loss = cce(in_y, probs)
        return loss


    def ce_nn_pred(self, enc_x, in_y, learnt_y):
        enc_x_dist = self.pairwise_dist(enc_x, learnt_y)
        probs = tf.nn.softmax(-1. * enc_x_dist, axis=1)  # the closer the better
        preds = tf.argmax(probs, axis=1)
        true_y = tf.math.argmax(in_y, axis=1)
        return preds, true_y
    
    def forward(self, z, in_y):
        structure_loss = 0.0
        warming_up = self.current_step < self.warmup_steps
        __current_step = self.current_step - self.warmup_steps
        if not warming_up and (__current_step % self.stationary_steps == 0): # will contain first update
            centroids = self.compute_centroids(z, in_y, 10)
            self.learnt_y = self.update_learnt_centroids(self.learnt_y, centroids)
            
            if self.rloss == "cos_repel_loss_z":
                structure_loss = self.cos_repel_loss_z(z, in_y, 10)
            
        self.current_step += 1
        
        # promote centroids to be further apart
        input_loss = self.ce_pull_loss(z, in_y, self.learnt_y)
        # em_loss = 10.0 * structure_loss + 1.0 * input_loss
        em_loss = input_loss
        return em_loss
    
    def call(self, x, training = False):
        z = self.encoder(x, training=training)
        z = self.classification_head(z)
        # print(tf.reduce_max(z))
        return z, None

    def test_step(self, data, learnt_y):
        z, in_y = data
        print(z, in_y, learnt_y)

        input_loss = self.ce_pull_loss(z, in_y, learnt_y)
        structure_loss = self.cos_repel_loss_z(z, in_y, self.num_classes)
        em_loss = 10.0 * structure_loss + 1.0 * input_loss

        predictions, true_y = self.ce_nn_pred(z, in_y, learnt_y)
        metric = tf.keras.metrics.Accuracy()
        metric.update_state(true_y, predictions)
        return metric.result()


def are_tensors_equivalent(tf_tensor, torch_tensor, rtol=1e-5, atol=1e-8):
    tf_array = tf_tensor.numpy() if isinstance(tf_tensor, tf.Tensor) else tf_tensor
    torch_array = torch_tensor.detach().cpu().numpy() if isinstance(torch_tensor, torch.Tensor) else torch_tensor

    if tf_array.shape != torch_array.shape:
        return False

    return np.allclose(tf_array, torch_array, rtol=rtol, atol=atol)

def create_dummy_tensors(shape):
    tf_arr = tf.random.uniform(shape, minval=-3, maxval=3)
    torch_arr = torch.tensor(tf_arr.numpy())
    return tf_arr, torch_arr

def calculate_vector_norms(vectors):
    norms = torch.linalg.norm(vectors, dim=1, keepdim=True)
    return norms

def get_max_element(tensor):
    return torch.max(tensor)

print("Creating dummy random values")

batch_size=2
num_classes=4
latent_dim=5
tf_z, torch_z = create_dummy_tensors((batch_size, latent_dim))
tf_in_y, torch_in_y = create_dummy_tensors((batch_size, num_classes))
tf_learnt_y, torch_learnt_y = create_dummy_tensors((num_classes, latent_dim))
tf_centroids, torch_centroids = create_dummy_tensors((num_classes, latent_dim))

print("Testing pairwise_dist")
tf_out = XiaoLwal().pairwise_dist(tf_z, tf_learnt_y)
torch_out = TimmLwal().pairwise_dist(torch_z, torch_learnt_y)
assert(are_tensors_equivalent(tf_out, torch_out))

print("Testing pairwise_cosine_similarity")
cosine_out = TimmLwal().pairwise_cosine_similarity(torch_z, torch_learnt_y)
print(torch_out)
print(cosine_out)

print("Testing cc_nn_pred")
tf_out = XiaoLwal().ce_nn_pred(tf_z, tf_in_y, tf_learnt_y)
torch_out = TimmLwal().cross_entropy_nn_pred(torch_z, torch_in_y, torch_learnt_y)
print(tf_out[0])
print(torch_out[0])
print(tf_out[1])
print(torch_out[1])
assert(are_tensors_equivalent(tf_out[0], torch_out[0]))
assert(are_tensors_equivalent(tf_out[1], torch_out[1])) 

print("Testing accuracy")
tf_out = XiaoLwal().test_step((tf_z, tf_in_y), tf_learnt_y)
torch_out = TimmLwal().accuracy(torch_z, torch_in_y, torch_learnt_y)
print(tf_out, torch_out)

print("Testing compute_centroids")
tf_out = XiaoLwal().compute_centroids(tf_z, tf_in_y)
torch_out = TimmLwal().compute_centroids(torch_z, torch_in_y)
assert(are_tensors_equivalent(tf_out, torch_out))
# print(tf_out, torch_out)

print("Testing update_learnt_centroids")
tf_out = XiaoLwal().update_learnt_centroids(tf_learnt_y, tf_centroids)
torch_out = TimmLwal().update_learnt_centroids(torch_learnt_y, torch_centroids)
assert(are_tensors_equivalent(tf_out, torch_out))
# print(tf_out, torch_out)

print("Testing update_learnt_centroids")
tf_out = XiaoLwal().update_learnt_centroids(tf_learnt_y, tf_centroids)
torch_out = TimmLwal().update_learnt_centroids_new(torch_learnt_y, torch_centroids)
assert(are_tensors_equivalent(tf_out, torch_out))
# print(tf_out, torch_out)

print("Testing pull loss")
tf_out = XiaoLwal().ce_pull_loss(tf_z, tf_in_y, tf_learnt_y)
torch_out = TimmLwal().cross_entropy_pull_loss(torch_z, torch_in_y, torch_learnt_y)
# print(tf_out, torch_out)
assert(are_tensors_equivalent(tf_out, torch_out))

print("Testing repel loss")
tf_out = XiaoLwal().cos_repel_loss_z(tf_z, tf_in_y, 10)
torch_out = TimmLwal().cos_repel_loss_z_optimized(torch_z, torch_in_y)
print(tf_out, torch_out)
assert(are_tensors_equivalent(tf_out, torch_out))

# print("Testing forward()")
# tf_out = XiaoLwal(current_step=0, warmup_steps=0, stationary_steps=2, rloss="cos_repel_loss_z", learnt_y=tf_learnt_y).forward(tf_z, tf_in_y)
# torch_out = TimmLwal(current_step=0, stationary_steps=2, learnt_y=torch_learnt_y).forward(torch_z, torch_in_y)
# # print(tf_out, torch_out)
# assert(are_tensors_equivalent(tf_out, torch_out))



# print("Testing exploding centroids")
# BATCH_SIZE=500
# LATENT_DIM=100
# NUM_CLASSES=10
# _, learnt_y = create_dummy_tensors((NUM_CLASSES, LATENT_DIM))
# _, centroids = create_dummy_tensors((NUM_CLASSES, LATENT_DIM))
# for i in range(1000):
#     _, z = create_dummy_tensors((BATCH_SIZE, LATENT_DIM))
#     _, in_y = create_dummy_tensors((BATCH_SIZE, NUM_CLASSES))
#     centroids = TimmLwal().compute_centroids(z, in_y, NUM_CLASSES)
#     learnt_y = TimmLwal().update_learnt_centroids(learnt_y, centroids)
#     if i % 100 == 0:
#         print(i, 
#               get_max_element(learnt_y).item(), 
#               get_max_element(calculate_vector_norms(learnt_y)).item())
# print(z.shape, learnt_y.shape)
# print(TimmLwal().pairwise_cosine_similarity(z, learnt_y))


# print('Testing cossim')
# z = torch.tensor(np.array([[6.7383e-02,  8.3008e-02,  1.3428e-02, 9.0332e-03, -1.5430e-01,  1.9727e-01],
#               [-5.1514e-02,  8.3008e-02, -1.0449e-01, -1.9336e-01, -2.4414e-03,  7.4463e-03],
#               [-3.8477e-01,  2.2217e-02,  8.0078e-02,  -1.8848e-01, 3.6621e-04,  1.8945e-01]]))
# learnt_y = torch.tensor(np.array([[2.7586e-01, -7.0674e-02, -1.9153e-01, -8.1438e-02, -1.5261e-01, -3.5059e-02],
#                      [2.8019e-01,  2.1880e-01,  5.5808e-02, -4.3636e-02,  1.5164e-01, 1.4643e-01],
#                      [2.2926e-01, -4.7721e-02,  2.9392e-01,  6.4710e-02,  3.2113e-01, -4.6744e-02],
#                      [8.6757e-02,  2.1305e-01, -5.0460e-01, -6.7577e-02,  3.5086e-01, 1.2281e-01,]]))
# print(z.shape, learnt_y.shape)
# print(-TimmLwal().pairwise_cosine_similarity(z, learnt_y))