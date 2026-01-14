"""
Simple test to verify the implementation structure.
This test checks imports and basic functionality without running heavy computations.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.models import UNet
        print("✓ UNet model imported successfully")
    except Exception as e:
        print(f"✗ Failed to import UNet: {e}")
        return False
    
    try:
        from src.reconstruction import DIPReconstructor, reconstruct_with_dip
        print("✓ Reconstruction module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import reconstruction: {e}")
        return False
    
    try:
        from src.utils import (
            UnderSamplingMask,
            CompressedSensingOperator,
            add_noise,
            normalize_image,
            calculate_psnr,
            calculate_ssim
        )
        print("✓ Utilities imported successfully")
    except Exception as e:
        print(f"✗ Failed to import utilities: {e}")
        return False
    
    return True


def test_structure():
    """Test that the directory structure is correct."""
    print("\nTesting directory structure...")
    
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    
    required_files = [
        'src/__init__.py',
        'src/models/__init__.py',
        'src/models/unet.py',
        'src/utils/__init__.py',
        'src/utils/cs_utils.py',
        'src/utils/metrics.py',
        'src/utils/visualization.py',
        'src/reconstruction.py',
        'examples/basic_reconstruction.py',
        'configs/default_config.yaml',
        'requirements.txt',
        'setup.py',
        'README.md',
        '.gitignore'
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = os.path.join(base_dir, file_path)
        if os.path.exists(full_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ Missing: {file_path}")
            all_exist = False
    
    return all_exist


def main():
    """Run all tests."""
    print("=" * 60)
    print("Structure and Import Test")
    print("=" * 60)
    
    structure_ok = test_structure()
    imports_ok = test_imports()
    
    print("\n" + "=" * 60)
    if structure_ok and imports_ok:
        print("✓ All tests passed!")
        print("The implementation is ready to use.")
    else:
        print("✗ Some tests failed.")
        print("Please check the errors above.")
    print("=" * 60)


if __name__ == '__main__':
    main()
