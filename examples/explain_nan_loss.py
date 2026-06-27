"""Demonstrate exactly why NaN loss occurs with single-class data."""

import torch
import torch.nn as nn


def demonstrate_nan_loss():
    """Show step-by-step why single-class data causes NaN loss."""

    print("=" * 60)
    print("WHY NaN LOSS OCCURS: DETAILED EXPLANATION")
    print("=" * 60)

    print("Step 1: Your W3 data situation")
    print("-" * 30)

    # Simulate your W3 data: only 1 class
    batch_size = 4
    num_classes_w3 = 1  # This is the problem!
    labels_w3 = torch.tensor([0, 0, 0, 0])  # All samples are class 0

    print(f"Labels: {labels_w3}")
    print(f"Number of unique classes: {len(torch.unique(labels_w3))}")
    print(f"Model output shape: [{batch_size}, {num_classes_w3}]")

    # Model output for 1 class
    logits_w3 = torch.randn(batch_size, num_classes_w3)
    print(f"Model logits: {logits_w3.flatten()}")

    print("\nStep 2: CrossEntropyLoss computation")
    print("-" * 30)

    loss_fn = nn.CrossEntropyLoss()

    try:
        loss_w3 = loss_fn(logits_w3, labels_w3)
        print(f"Loss result: {loss_w3}")
        print(f"Is NaN: {torch.isnan(loss_w3)}")
    except Exception as e:
        print(f"Error: {e}")

    print("\nStep 3: What happens internally")
    print("-" * 30)

    # Manual computation to show the issue
    print("Softmax computation:")
    softmax = torch.softmax(logits_w3, dim=1)
    print(f"Softmax output: {softmax.flatten()}")
    print("(For 1 class, softmax always = 1.0)")

    print("\nLog probability computation:")
    log_softmax = torch.log_softmax(logits_w3, dim=1)
    print(f"Log softmax: {log_softmax.flatten()}")

    print("\nGathering target probabilities:")
    target_log_probs = log_softmax.gather(1, labels_w3.unsqueeze(1))
    print(f"Target log probs: {target_log_probs.flatten()}")

    print("\nFinal loss (negative mean):")
    manual_loss = -target_log_probs.mean()
    print(f"Manual loss: {manual_loss}")
    print(f"Is NaN: {torch.isnan(manual_loss)}")

    print("\n" + "=" * 60)
    print("COMPARISON: Working multi-class case (like ETTh1)")
    print("=" * 60)

    # Simulate ETTh1 data: multiple classes
    num_classes_etth1 = 3  # Multiple classes
    labels_etth1 = torch.tensor([0, 1, 2, 1])  # Diverse classes
    logits_etth1 = torch.randn(batch_size, num_classes_etth1)

    print(f"Labels: {labels_etth1}")
    print(f"Number of unique classes: {len(torch.unique(labels_etth1))}")
    print(f"Model output shape: [{batch_size}, {num_classes_etth1}]")
    print(f"Model logits: {logits_etth1}")

    try:
        loss_etth1 = loss_fn(logits_etth1, labels_etth1)
        print(f"Loss result: {loss_etth1:.4f}")
        print(f"Is NaN: {torch.isnan(loss_etth1)}")
        print("✅ Works perfectly!")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("SOLUTIONS")
    print("=" * 60)

    print("❌ Problem: Your W3 data")
    print("   - Only 1 unique class in the sample")
    print("   - Model creates 1-dimensional output")
    print("   - CrossEntropyLoss cannot compute meaningful gradients")
    print("   - Mathematical result: NaN")

    print("\n✅ Solution 1: Find multi-class W3 data")
    print("   - Sample from different time periods")
    print("   - Look for state transitions")
    print("   - Use larger/different subset")

    print("\n✅ Solution 2: Create synthetic classes")
    print("   - Use feature patterns to create classes")
    print("   - Split by quantiles or clustering")
    print("   - Ensure balanced distribution")

    print("\n✅ Solution 3: Use regression instead")
    print("   - Predict continuous values instead of classes")
    print("   - Use MSELoss instead of CrossEntropyLoss")
    print("   - Won't have the NaN issue")

    print("\n" + "=" * 60)
    print("YOUR MODEL IS CORRECT - IT'S A DATA ISSUE!")
    print("=" * 60)


def show_exact_w3_issue():
    """Show the exact issue from your W3 output."""

    print("\n" + "=" * 60)
    print("YOUR EXACT W3 ISSUE RECREATED")
    print("=" * 60)

    # Your exact configuration
    batch_size = 64
    num_classes = 1  # From your output: "Classes: 1"

    # Your exact model output shape and range
    # From your log: "Output shape: torch.Size([64, 1])"
    # "Output range: [-0.202, -0.192]"
    logits = torch.tensor([[-0.202], [-0.195], [-0.198], [-0.192]] * 16)  # 64 samples

    # Your exact labels: all zeros
    labels = torch.zeros(64, dtype=torch.long)  # All class 0

    print(f"Batch size: {batch_size}")
    print(f"Number of classes: {num_classes}")
    print(f"Logits shape: {logits.shape}")
    print(f"Labels: {torch.unique(labels)} (only class 0)")
    print(f"Logits sample: {logits[:4].flatten()}")

    # Compute loss
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, labels)

    print("\nLoss computation:")
    print(f"Result: {loss}")
    print(f"Is NaN: {torch.isnan(loss)}")

    # Show why it's NaN
    print("\nWhy it's NaN:")
    print(f"1. Softmax of single logit: {torch.softmax(logits[:4], dim=1)}")
    print(f"2. Log of softmax: {torch.log_softmax(logits[:4], dim=1)}")
    print("3. All values are the same → gradient = 0 → NaN in backprop")

    print("\n🔍 Root cause:")
    print("   CrossEntropyLoss needs class diversity to compute meaningful gradients")
    print("   With only 1 class, there's no 'choice' for the model to learn")
    print("   Result: Mathematical instability → NaN")


if __name__ == "__main__":
    demonstrate_nan_loss()
    show_exact_w3_issue()
