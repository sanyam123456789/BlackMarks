"""
Model Verification Script for BlackMarks Compact CNN

Verifies that the CompactCNN architecture:
1. Instantiates properly via build_model()
2. Accepts a dummy tensor of shape (2, 3, 32, 32)
3. Produces the expected output shape (2, 10)
4. Executes forward pass correctly with finite numerical outputs (no NaN/Inf)
5. Reports total and trainable parameter counts
6. Integrates seamlessly with real CIFAR-10 DataLoader batches from src.classifier.data

NOTE: This script ONLY verifies the architecture through forward inference.
      No training or weight updates are performed.
"""

import sys
from pathlib import Path

# Add project root to sys.path so script can be run directly from any directory
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
from src.classifier.model import CompactCNN, build_model, count_parameters, get_model_summary
from src.classifier.data import get_cifar10_dataloaders


def verify_compact_cnn(data_dir: str = "./data") -> bool:
    """
    Executes full suite of architectural verification tests on CompactCNN.

    Returns:
        True if all verification checks pass, False otherwise.
    """
    print("=" * 68)
    print("BLACKMARKS: COMPACT CNN ARCHITECTURE VERIFICATION (CLEAN MODEL)")
    print("=" * 68)

    all_passed = True

    # -------------------------------------------------------------
    # 1. Model Instantiation
    # -------------------------------------------------------------
    print("[1/5] Instantiating CompactCNN via build_model()...")
    try:
        model = build_model(num_classes=10, dropout_rate=0.5)
        model.eval()  # Verification mode (inference only)
        print("      Model instantiated successfully.")
    except Exception as e:
        print(f"      [FAIL] Model instantiation failed: {e}")
        return False

    # -------------------------------------------------------------
    # 2. Parameter Count & Summary
    # -------------------------------------------------------------
    print("\n[2/5] Inspecting model parameters and structure...")
    total_params, trainable_params = count_parameters(model)
    print(f"      Total parameters:     {total_params:,d}")
    print(f"      Trainable parameters: {trainable_params:,d}")
    print(f"      Non-trainable params: {total_params - trainable_params:,d}")

    # Print layer breakdown
    print("\n" + get_model_summary(model, input_size=(3, 32, 32)))

    if total_params <= 0 or trainable_params <= 0:
        print("      [FAIL] Invalid parameter counts.")
        all_passed = False

    # -------------------------------------------------------------
    # 3. Dummy Forward Pass Verification
    # -------------------------------------------------------------
    print("\n[3/5] Testing forward pass with dummy tensor (batch_size=2)...")
    dummy_input = torch.randn(2, 3, 32, 32)
    print(f"      Input tensor shape: {tuple(dummy_input.shape)}")

    try:
        with torch.no_grad():
            dummy_output = model(dummy_input)

        output_shape = tuple(dummy_output.shape)
        print(f"      Output tensor shape: {output_shape}")

        if output_shape != (2, 10):
            print(f"      [FAIL] Expected output shape (2, 10), got {output_shape}")
            all_passed = False
        else:
            print("      [PASS] Output shape matches expected (2, 10).")

        # Numerical sanity check: finite floats, no NaNs or Infs
        if not torch.isfinite(dummy_output).all():
            print("      [FAIL] Model output contains NaN or Inf values.")
            all_passed = False
        else:
            print("      [PASS] All output logits are finite numbers.")

    except Exception as e:
        print(f"      [FAIL] Forward pass raised an exception: {e}")
        all_passed = False

    # -------------------------------------------------------------
    # 4. CIFAR-10 DataLoader Compatibility Check
    # -------------------------------------------------------------
    print("\n[4/5] Testing compatibility with CIFAR-10 DataLoader...")
    try:
        train_loader, test_loader = get_cifar10_dataloaders(
            data_dir=data_dir,
            batch_size=64,
            num_workers=0,
            download=False,
            normalize=True,
        )

        real_images, real_labels = next(iter(train_loader))
        print(f"      Real batch input shape:  {tuple(real_images.shape)}")
        print(f"      Real batch labels shape: {tuple(real_labels.shape)}")

        with torch.no_grad():
            real_output = model(real_images)

        real_output_shape = tuple(real_output.shape)
        print(f"      Real batch output shape: {real_output_shape}")

        if real_output_shape != (64, 10):
            print(f"      [FAIL] Expected real batch output (64, 10), got {real_output_shape}")
            all_passed = False
        else:
            print("      [PASS] Successfully processed real CIFAR-10 DataLoader batch.")

        if not torch.isfinite(real_output).all():
            print("      [FAIL] Real batch output contains NaN or Inf values.")
            all_passed = False
        else:
            print("      [PASS] Real batch outputs are all finite floats.")

    except Exception as e:
        print(f"      [FAIL] DataLoader compatibility test failed: {e}")
        all_passed = False

    # -------------------------------------------------------------
    # 5. Summary & Verdict
    # -------------------------------------------------------------
    print("\n" + "=" * 68)
    if all_passed:
        print("[SUCCESS] All CompactCNN verification checks PASSED!")
        print("The clean compact CNN architecture is verified and ready for training pipeline.")
    else:
        print("[ERROR] CompactCNN verification checks FAILED.")
    print("=" * 68)

    return all_passed


if __name__ == "__main__":
    success = verify_compact_cnn()
    sys.exit(0 if success else 1)
