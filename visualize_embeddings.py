import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Step 1: Generate random 10D vectors
num_vectors = 100
dim = 10
vectors = np.random.randn(num_vectors, dim)

# Step 2: Reduce to 2D using PCA
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)

# Step 3: Plot the vectors
plt.figure(figsize=(8, 6))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c='blue', alpha=0.7)
plt.title('PCA Visualization of 10D Vectors')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()
