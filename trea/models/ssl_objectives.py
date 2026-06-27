"""Self-supervised learning objectives for time series pre-training."""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedPatchPredictor:
    """Masked patch prediction for time series (like BERT for sequences)."""

    def __init__(
        self,
        mask_ratio: float = 0.15,
        replace_ratio: float = 0.8,
        random_ratio: float = 0.1,
        mask_token_value: float = 0.0,
    ):
        """Initialize masked patch predictor.

        Args:
            mask_ratio: Ratio of patches to mask
            replace_ratio: Ratio of masked patches to replace with mask token
            random_ratio: Ratio of masked patches to replace with random values
            mask_token_value: Value to use for mask token
        """
        self.mask_ratio = mask_ratio
        self.replace_ratio = replace_ratio
        self.random_ratio = random_ratio
        self.mask_token_value = mask_token_value

    def create_mask(self, num_patches: int, device: torch.device) -> torch.Tensor:
        """Create random mask for patches.

        Args:
            num_patches: Number of patches in sequence
            device: Device to create mask on

        Returns:
            Boolean mask [num_patches] where True = masked
        """
        num_masked = int(num_patches * self.mask_ratio)

        # Randomly select patches to mask
        indices = torch.randperm(num_patches, device=device)
        mask = torch.zeros(num_patches, dtype=torch.bool, device=device)
        mask[indices[:num_masked]] = True

        return mask

    def apply_masking(
        self, patches: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply masking to patches.

        Args:
            patches: Input patches [B, num_patches, patch_dim]
            mask: Boolean mask [num_patches]

        Returns:
            Tuple of (masked_patches, targets)
        """
        B, num_patches, patch_dim = patches.shape
        device = patches.device

        # Create targets (original values of masked patches)
        targets = patches.clone()

        # Apply masking strategy
        masked_patches = patches.clone()
        masked_indices = torch.where(mask)[0]

        for idx in masked_indices:
            rand_val = random.random()

            if rand_val < self.replace_ratio:
                # Replace with mask token
                masked_patches[:, idx, :] = self.mask_token_value
            elif rand_val < self.replace_ratio + self.random_ratio:
                # Replace with random patch from same batch
                random_batch_idx = torch.randint(0, B, (1,), device=device)
                random_patch_idx = torch.randint(0, num_patches, (1,), device=device)
                masked_patches[:, idx, :] = patches[
                    random_batch_idx, random_patch_idx, :
                ]
            # else: keep original (for robustness)

        return masked_patches, targets


class TemporalOrderPredictor:
    """Temporal order prediction for learning temporal dependencies."""

    def __init__(self, shuffle_ratio: float = 0.3):
        """Initialize temporal order predictor.

        Args:
            shuffle_ratio: Ratio of sequences to shuffle
        """
        self.shuffle_ratio = shuffle_ratio

    def create_shuffled_sequence(
        self, patches: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create shuffled sequence and targets.

        Args:
            patches: Input patches [B, num_patches, patch_dim]

        Returns:
            Tuple of (shuffled_patches, order_targets)
        """
        B, num_patches, patch_dim = patches.shape
        device = patches.device

        shuffled_patches = patches.clone()
        order_targets = (
            torch.arange(num_patches, device=device).unsqueeze(0).repeat(B, 1)
        )

        # Shuffle some sequences
        num_shuffle = int(B * self.shuffle_ratio)

        for i in range(num_shuffle):
            # Create random permutation
            perm = torch.randperm(num_patches, device=device)
            shuffled_patches[i] = patches[i, perm]
            order_targets[i] = perm

        return shuffled_patches, order_targets


class ContrastiveLearner:
    """Contrastive learning for cross-domain representations."""

    def __init__(self, temperature: float = 0.1, augmentation_strength: float = 0.1):
        """Initialize contrastive learner.

        Args:
            temperature: Temperature for contrastive loss
            augmentation_strength: Strength of data augmentation
        """
        self.temperature = temperature
        self.augmentation_strength = augmentation_strength

    def augment_time_series(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to time series.

        Args:
            x: Time series [B, C, T]

        Returns:
            Augmented time series
        """
        device = x.device
        B, C, T = x.shape

        # Jittering (add noise)
        noise = torch.randn_like(x) * self.augmentation_strength
        x_aug = x + noise

        # Scaling
        scale_factor = (
            1 + torch.randn(B, C, 1, device=device) * self.augmentation_strength
        )
        x_aug = x_aug * scale_factor

        # Time warping (simple version - random time shifts)
        if T > 10:
            shift = int(T * 0.1)  # Max 10% shift
            for i in range(B):
                if random.random() < 0.5:
                    random_shift = random.randint(-shift, shift)
                    if random_shift > 0:
                        x_aug[i, :, random_shift:] = x_aug[i, :, :-random_shift]
                        x_aug[i, :, :random_shift] = 0
                    elif random_shift < 0:
                        x_aug[i, :, :random_shift] = x_aug[i, :, -random_shift:]
                        x_aug[i, :, random_shift:] = 0

        return x_aug

    def contrastive_loss(
        self, embeddings1: torch.Tensor, embeddings2: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive loss between two sets of embeddings.

        Args:
            embeddings1: First set of embeddings [B, D]
            embeddings2: Second set of embeddings [B, D]

        Returns:
            Contrastive loss
        """
        B = embeddings1.shape[0]

        # Normalize embeddings
        embeddings1 = F.normalize(embeddings1, dim=1)
        embeddings2 = F.normalize(embeddings2, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings1, embeddings2.T) / self.temperature

        # Positive pairs are on the diagonal
        labels = torch.arange(B, device=embeddings1.device)

        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss


class SSLObjectives:
    """Combined SSL objectives for time series pre-training."""

    def __init__(
        self,
        mask_ratio: float = 0.15,
        temporal_shuffle_ratio: float = 0.3,
        contrastive_temperature: float = 0.1,
        augmentation_strength: float = 0.1,
        lambda_masked: float = 1.0,
        lambda_temporal: float = 0.5,
        lambda_contrastive: float = 0.3,
    ):
        """Initialize combined SSL objectives.

        Args:
            mask_ratio: Ratio of patches to mask for MPP
            temporal_shuffle_ratio: Ratio of sequences to shuffle for temporal order
            contrastive_temperature: Temperature for contrastive loss
            augmentation_strength: Strength of augmentation for contrastive learning
            lambda_masked: Weight for masked patch prediction loss
            lambda_temporal: Weight for temporal order prediction loss
            lambda_contrastive: Weight for contrastive loss
        """
        self.masked_predictor = MaskedPatchPredictor(mask_ratio=mask_ratio)
        self.temporal_predictor = TemporalOrderPredictor(
            shuffle_ratio=temporal_shuffle_ratio
        )
        self.contrastive_learner = ContrastiveLearner(
            temperature=contrastive_temperature,
            augmentation_strength=augmentation_strength,
        )

        self.lambda_masked = lambda_masked
        self.lambda_temporal = lambda_temporal
        self.lambda_contrastive = lambda_contrastive

    def compute_ssl_losses(
        self,
        model: nn.Module,
        x_num: torch.Tensor,
        feature_mask: torch.Tensor,
        ssl_heads: dict[str, nn.Module],
    ) -> dict[str, torch.Tensor]:
        """Compute all SSL losses.

        Args:
            model: The main model (for getting patch embeddings)
            x_num: Numeric input [B, max_features, T]
            feature_mask: Feature mask [B, max_features, T]
            ssl_heads: Dictionary of SSL prediction heads

        Returns:
            Dictionary of losses
        """
        B, C, T = x_num.shape
        losses = {}

        # Get model's internal patch representations
        # (This requires the model to expose intermediate representations)
        with torch.no_grad():
            # Forward through model to get patch embeddings
            patches = model.create_patches_with_features(x_num, feature_mask)
            # patches: [B, num_patches, patch_dim]

        num_patches = patches.shape[1]

        # 1. Masked Patch Prediction
        if "reconstruction" in ssl_heads:
            mask = self.masked_predictor.create_mask(num_patches, x_num.device)
            masked_patches, targets = self.masked_predictor.apply_masking(patches, mask)

            # Forward through model with masked patches
            patch_embeddings = model.patch_embedding(
                masked_patches.view(B, num_patches, -1)
            )
            pos_enc = model.get_positional_encoding(num_patches)
            patch_embeddings = patch_embeddings + pos_enc
            transformer_out = model.transformer(patch_embeddings)

            # Predict original patches
            reconstructed = ssl_heads["reconstruction"](transformer_out)

            # Compute reconstruction loss only for masked patches
            mask_expanded = mask.unsqueeze(0).unsqueeze(-1).expand_as(reconstructed)
            reconstruction_loss = F.mse_loss(
                reconstructed[mask_expanded],
                targets.view(B, num_patches, -1)[mask_expanded],
            )
            losses["reconstruction"] = reconstruction_loss * self.lambda_masked

        # 2. Temporal Order Prediction
        if "temporal_order" in ssl_heads:
            shuffled_patches, order_targets = (
                self.temporal_predictor.create_shuffled_sequence(patches)
            )

            # Forward through model
            patch_embeddings = model.patch_embedding(
                shuffled_patches.view(B, num_patches, -1)
            )
            pos_enc = model.get_positional_encoding(num_patches)
            patch_embeddings = patch_embeddings + pos_enc
            transformer_out = model.transformer(patch_embeddings)

            # Predict temporal order
            order_logits = ssl_heads["temporal_order"](transformer_out.mean(dim=1))

            # Compute order prediction loss
            temporal_loss = F.mse_loss(order_logits, order_targets.float())
            losses["temporal_order"] = temporal_loss * self.lambda_temporal

        # 3. Contrastive Learning
        if "contrastive" in ssl_heads:
            # Create augmented version
            x_aug = self.contrastive_learner.augment_time_series(x_num)

            # Get embeddings for original and augmented
            orig_patches = model.create_patches_with_features(x_num, feature_mask)
            aug_patches = model.create_patches_with_features(x_aug, feature_mask)

            # Forward through model
            orig_embeddings = model.get_global_embedding(orig_patches)
            aug_embeddings = model.get_global_embedding(aug_patches)

            # Compute contrastive loss
            contrastive_loss = self.contrastive_learner.contrastive_loss(
                orig_embeddings, aug_embeddings
            )
            losses["contrastive"] = contrastive_loss * self.lambda_contrastive

        # Total SSL loss
        total_ssl_loss = sum(losses.values())
        losses["total_ssl"] = total_ssl_loss

        return losses


if __name__ == "__main__":
    # Test SSL objectives
    print("Testing SSL objectives...")

    # Create dummy data
    B, num_patches, patch_dim = 8, 12, 64
    patches = torch.randn(B, num_patches, patch_dim)

    # Test Masked Patch Prediction
    mpp = MaskedPatchPredictor(mask_ratio=0.15)
    mask = mpp.create_mask(num_patches, patches.device)
    masked_patches, targets = mpp.apply_masking(patches, mask)

    print(f"Original patches: {patches.shape}")
    print(f"Mask: {mask.sum()}/{len(mask)} patches masked")
    print(f"Masked patches: {masked_patches.shape}")
    print(f"Targets: {targets.shape}")

    # Test Temporal Order Prediction
    top = TemporalOrderPredictor(shuffle_ratio=0.5)
    shuffled_patches, order_targets = top.create_shuffled_sequence(patches)

    print(f"Shuffled patches: {shuffled_patches.shape}")
    print(f"Order targets: {order_targets.shape}")

    # Test Contrastive Learning
    cl = ContrastiveLearner()
    x = torch.randn(B, 10, 96)  # [B, C, T]
    x_aug = cl.augment_time_series(x)

    embeddings1 = torch.randn(B, 128)
    embeddings2 = torch.randn(B, 128)
    contrastive_loss = cl.contrastive_loss(embeddings1, embeddings2)

    print(f"Original time series: {x.shape}")
    print(f"Augmented time series: {x_aug.shape}")
    print(f"Contrastive loss: {contrastive_loss.item():.4f}")

    print("✅ SSL objectives test complete!")
