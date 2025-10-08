"""
Quick setup verification script for WandB integration
Checks if all dependencies are installed and configured correctly
"""
import sys


def check_wandb_installation():
    """Check if wandb is installed"""
    try:
        import wandb
        print(f"✓ wandb installed (version {wandb.__version__})")
        return True
    except ImportError:
        print("✗ wandb not installed")
        print("  Install with: pip install wandb")
        return False


def check_wandb_login():
    """Check if user is logged into wandb"""
    try:
        import wandb
        # Try to get API key
        api = wandb.Api()
        print(f"✓ wandb logged in (user: {api.default_entity or 'default'})")
        return True
    except Exception as e:
        print("✗ wandb not logged in")
        print("  Login with: wandb login")
        return False


def check_dependencies():
    """Check if all required dependencies are installed"""
    dependencies = {
        'yaml': 'pyyaml',
        'torch': 'torch',
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'albumentations': 'albumentations'
    }

    all_installed = True
    for module_name, package_name in dependencies.items():
        try:
            __import__(module_name)
            print(f"✓ {package_name} installed")
        except ImportError:
            print(f"✗ {package_name} not installed")
            print(f"  Install with: pip install {package_name}")
            all_installed = False

    return all_installed


def check_config_file():
    """Check if config file exists and has wandb settings"""
    try:
        import yaml
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # Check if wandb is enabled
        if config.get('logging', {}).get('use_wandb', False):
            print("✓ WandB enabled in config.yaml")
        else:
            print("✗ WandB disabled in config.yaml")
            print("  Set 'use_wandb: true' in configs/config.yaml")
            return False

        # Check if wandb settings exist
        if 'wandb' in config.get('logging', {}):
            print("✓ WandB settings found in config")
        else:
            print("⚠ WandB settings section missing in config")
            print("  This is optional, defaults will be used")

        return True

    except FileNotFoundError:
        print("✗ config.yaml not found")
        print("  Expected at: configs/config.yaml")
        return False
    except Exception as e:
        print(f"✗ Error reading config: {e}")
        return False


def check_dataset():
    """Check if dataset exists"""
    from pathlib import Path

    dataset_path = Path('datasets/Images/Train')
    if dataset_path.exists():
        num_images = len(list(dataset_path.glob('*.jpg')))
        print(f"✓ Dataset found ({num_images} training images)")
        return True
    else:
        print("✗ Dataset not found")
        print("  Expected at: datasets/Images/Train")
        return False


def check_files():
    """Check if required files exist"""
    from pathlib import Path

    required_files = [
        'dataloader.py',
        'dataset.py',
        'wandb_logger.py',
        'test_with_wandb.py',
        'configs/config.yaml'
    ]

    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} missing")
            all_exist = False

    return all_exist


def run_quick_test():
    """Run a quick test of the WandB logger"""
    try:
        print("\nRunning quick WandB test...")
        import torch
        import numpy as np
        from wandb_logger import WandbLogger

        # Create minimal config
        config = {
            'logging': {
                'project_name': 'aerial-person-detection-test',
                'experiment_name': 'setup_verification',
                'use_wandb': True,
                'wandb': {
                    'entity': None,
                    'tags': ['setup-test'],
                    'notes': 'Testing WandB setup',
                    'log_images': True,
                    'log_predictions': True,
                    'max_images_to_log': 2
                }
            },
            'data': {'img_size': [640, 640], 'num_classes': 1}
        }

        # Initialize logger
        logger = WandbLogger(config)

        # Log test metrics
        logger.log_metrics({'test_metric': 0.99}, step=0)

        # Create dummy data
        images = [torch.randn(3, 640, 640) for _ in range(2)]
        gt_boxes = [np.array([[100, 100, 200, 200]]), np.array([[150, 150, 250, 250]])]
        gt_labels = [np.array([0]), np.array([0])]
        pred_boxes = [np.array([[105, 105, 205, 205]]), np.array([[155, 155, 255, 255]])]
        pred_labels = [np.array([0]), np.array([0])]
        pred_scores = [np.array([0.95]), np.array([0.87])]

        # Log images
        logger.log_images_with_boxes(
            images, gt_boxes, gt_labels,
            pred_boxes, pred_labels, pred_scores,
            step=0, split='test', max_images=2
        )

        print(f"✓ WandB test successful!")
        print(f"  View at: {logger.run.url}")

        # Finish
        logger.finish()
        return True

    except Exception as e:
        print(f"✗ WandB test failed: {e}")
        return False


def main():
    """Main verification function"""
    print("="*60)
    print("WandB Setup Verification")
    print("="*60)

    print("\n1. Checking Dependencies")
    print("-" * 40)
    deps_ok = check_dependencies()

    print("\n2. Checking WandB Installation")
    print("-" * 40)
    wandb_installed = check_wandb_installation()

    wandb_logged_in = False
    if wandb_installed:
        wandb_logged_in = check_wandb_login()

    print("\n3. Checking Files")
    print("-" * 40)
    files_ok = check_files()

    print("\n4. Checking Configuration")
    print("-" * 40)
    config_ok = check_config_file()

    print("\n5. Checking Dataset")
    print("-" * 40)
    dataset_ok = check_dataset()

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    all_ok = deps_ok and wandb_installed and wandb_logged_in and files_ok and config_ok and dataset_ok

    if all_ok:
        print("✓ All checks passed!")
        print("\nYou're ready to use WandB!")
        print("\nNext steps:")
        print("  1. Test WandB logger: python wandb_logger.py")
        print("  2. Run inference: python test_with_wandb.py")
        print("  3. Start training with WandB logging")

        # Ask if user wants to run quick test
        print("\n" + "-"*60)
        try:
            response = input("Run quick WandB test now? (y/n): ")
            if response.lower() == 'y':
                run_quick_test()
        except:
            pass

    else:
        print("✗ Some checks failed")
        print("\nPlease fix the issues above and try again")

        # Provide specific help
        if not wandb_installed:
            print("\nTo install wandb:")
            print("  pip install wandb")

        if wandb_installed and not wandb_logged_in:
            print("\nTo login to wandb:")
            print("  wandb login")
            print("  (Get your API key from https://wandb.ai/authorize)")

        if not config_ok:
            print("\nTo enable WandB in config:")
            print("  Edit configs/config.yaml")
            print("  Set: use_wandb: true")

        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
