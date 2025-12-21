"""
Test script for experiment tracking (W&B and TensorBoard)
"""
import os
os.environ['WANDB_SILENT'] = 'true'

def test_wandb():
    print("Testing Weights & Biases...")
    import wandb

    # Initialize a test run (offline mode for quick test)
    run = wandb.init(
        project="adaptface-test",
        name="env-test",
        mode="offline",  # Don't actually upload
        config={"test": True}
    )

    # Log some test metrics
    wandb.log({"test_metric": 0.95, "loss": 0.05})

    # Finish
    wandb.finish()
    print("W&B test: PASSED")
    return True

def test_tensorboard():
    print("Testing TensorBoard...")
    from torch.utils.tensorboard import SummaryWriter
    import tempfile
    import shutil

    # Create a temporary directory for test
    log_dir = tempfile.mkdtemp()

    try:
        writer = SummaryWriter(log_dir)
        writer.add_scalar('test/accuracy', 0.95, 0)
        writer.add_scalar('test/loss', 0.05, 0)
        writer.close()
        print("TensorBoard test: PASSED")
        return True
    finally:
        shutil.rmtree(log_dir)

if __name__ == "__main__":
    print("=" * 50)
    print("Experiment Tracking Test")
    print("=" * 50)

    try:
        test_tensorboard()
    except Exception as e:
        print(f"TensorBoard test: FAILED - {e}")

    try:
        test_wandb()
    except Exception as e:
        print(f"W&B test: FAILED - {e}")

    print("=" * 50)
    print("Tracking systems ready!")
    print("=" * 50)