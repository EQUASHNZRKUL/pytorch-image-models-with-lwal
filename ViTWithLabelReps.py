import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel

class ViTWithLabelReps(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        embed_dim: int = 768,
        use_normalization: bool = True,
        reg_strength: float = 0.0,
    ):
        """
        Args:
            model_name: HuggingFace ViT model name (e.g., "google/vit-base-patch16-224-in21k")
            num_classes: number of target classes
            embed_dim: dimension of label and feature embeddings
            use_normalization: if True, use cosine similarity + learnable scale
            reg_strength: coefficient for orthogonality regularization on label reps
        """
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.vit.config.hidden_size, embed_dim)

        # Trainable label representations
        self.label_reps = nn.Parameter(torch.randn(num_classes, embed_dim))

        # Normalization & scaling
        self.use_normalization = use_normalization
        self.scale = nn.Parameter(torch.tensor(10.0))  # learnable temperature

        # Regularization
        self.reg_strength = reg_strength

    def forward(self, pixel_values, labels=None):
        # Extract CLS token
        outputs = self.vit(pixel_values)
        cls_emb = outputs.last_hidden_state[:, 0]  # [batch, hidden]
        h = self.proj(cls_emb)  # [batch, embed_dim]

        # Normalize embeddings and label reps
        if self.use_normalization:
            h = F.normalize(h, dim=-1)
            label_reps = F.normalize(self.label_reps, dim=-1)
        else:
            label_reps = self.label_reps

        # Compute logits via dot product
        logits = torch.matmul(h, label_reps.T)  # [batch, num_classes]
        if self.use_normalization:
            logits = self.scale * logits

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

            # Orthogonality regularization (optional)
            if self.reg_strength > 0:
                gram = torch.matmul(label_reps, label_reps.T)  # [C, C]
                I = torch.eye(gram.size(0), device=gram.device)
                ortho_loss = ((gram - I) ** 2).mean()
                loss = loss + self.reg_strength * ortho_loss

        return {"loss": loss, "logits": logits}

class ViTWithStaticLabelReps(nn.Module):
    def __init__(
        self,
        model_name: str,
        label_reps: torch.Tensor,
        use_normalization: bool = True,
    ):
        """
        Args:
            model_name: HuggingFace ViT model name
            label_reps: [num_classes, embed_dim] tensor with fixed label reps
            use_normalization: whether to use cosine similarity + scaling
        """
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.vit.config.hidden_size, label_reps.size(1))

        # Store as a buffer (non-trainable)
        self.register_buffer("label_reps", label_reps)

        # Normalization & scaling
        self.use_normalization = use_normalization
        self.scale = nn.Parameter(torch.tensor(10.0))  # still trainable

    def forward(self, pixel_values, labels=None):
        # Extract CLS token
        outputs = self.vit(pixel_values)
        cls_emb = outputs.last_hidden_state[:, 0]  # [batch, hidden]
        h = self.proj(cls_emb)  # [batch, embed_dim]

        if self.use_normalization:
            h = F.normalize(h, dim=-1)
            label_reps = F.normalize(self.label_reps, dim=-1)
        else:
            label_reps = self.label_reps

        logits = torch.matmul(h, label_reps.T)
        if self.use_normalization:
            logits = self.scale * logits

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}
