import subprocess
import sys
from pathlib import Path
import os


def install_basic_dependencies():
    """Install basic dependencies"""
    print("\n" + "="*60)
    print("Installing basic dependencies...")
    print("="*60)
    
    packages = [
        'torch',
        'torchvision',
        'opencv-python',
        'pillow',
        'numpy',
        'tqdm',
        'transformers',
        'ftfy',
        'regex'
    ]
    
    for package in packages:
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '-q', package
            ])
            print(f"✓ {package}")
        except:
            print(f"⚠ {package} - installation failed")


def setup_egovlp():
    """Clone and setup EgoVLP repository"""
    print("\n" + "="*60)
    print("Setting up EgoVLP...")
    print("="*60)
    
    egovlp_dir = Path('./EgoVLP')
    
    # Clone repository
    if not egovlp_dir.exists():
        print("\nCloning EgoVLP repository...")
        try:
            subprocess.run([
                'git', 'clone',
                'https://github.com/facebookresearch/EgoVLP.git'
            ], check=True)
            print("✓ Repository cloned")
        except Exception as e:
            print(f"❌ Failed to clone repository: {e}")
            print("\nManual steps:")
            print("1. Run: git clone https://github.com/facebookresearch/EgoVLP.git")
            return False
    else:
        print("✓ EgoVLP repository already exists")
    
    # Install requirements
    requirements_file = egovlp_dir / 'requirements.txt'
    if requirements_file.exists():
        print("\nInstalling EgoVLP requirements...")
        try:
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-q',
                '-r', str(requirements_file)
            ])
            print("✓ Requirements installed")
        except Exception as e:
            print(f"⚠ Some requirements failed to install: {e}")
    
    # Add to Python path
    if str(egovlp_dir) not in sys.path:
        sys.path.insert(0, str(egovlp_dir))
    
    return True


def download_egovlp_checkpoint():
    """Download EgoVLP pretrained checkpoint"""
    print("\n" + "="*60)
    print("Downloading EgoVLP checkpoint...")
    print("="*60)
    
    checkpoint_dir = Path('./EgoVLP/pretrained')
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    checkpoint_path = checkpoint_dir / 'egovlp.pth'
    
    if checkpoint_path.exists() and checkpoint_path.stat().st_size > 1000:
        print("✓ Checkpoint already exists")
        return True
    
    # Try different checkpoint URLs
    checkpoint_urls = [
        ('EgoVLP (EgoClip)', 
         'https://dl.fbaipublicfiles.com/egovlp/egovlp.pth'),
        ('EgoVLP (Frozen-in-Time)',
         'https://dl.fbaipublicfiles.com/egovlp/frozen_in_time.pth'),
    ]
    
    for name, url in checkpoint_urls:
        print(f"\nTrying to download {name}...")
        print(f"URL: {url}")
        
        try:
            # Try wget
            result = subprocess.run([
                'wget', '-q', '--show-progress',
                '-O', str(checkpoint_path), url
            ], timeout=600)
            
            if result.returncode == 0 and checkpoint_path.exists():
                size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
                if size_mb > 1:
                    print(f"✓ Downloaded {name} ({size_mb:.1f} MB)")
                    return True
        except:
            pass
        
        try:
            # Try curl
            result = subprocess.run([
                'curl', '-L', '-o', str(checkpoint_path), url
            ], timeout=600)
            
            if result.returncode == 0 and checkpoint_path.exists():
                size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
                if size_mb > 1:
                    print(f"✓ Downloaded {name} ({size_mb:.1f} MB)")
                    return True
        except:
            pass
    
    print("\n⚠ Could not automatically download checkpoint")
    print("\nManual download instructions:")
    print("1. Visit: https://github.com/facebookresearch/EgoVLP")
    print("2. Download pretrained model")
    print(f"3. Save to: {checkpoint_path}")
    print("\nAlternatively, the code will use random initialization")
    
    return False


def test_egovlp_import():
    """Test if EgoVLP can be imported"""
    print("\n" + "="*60)
    print("Testing EgoVLP import...")
    print("="*60)
    
    try:
        sys.path.insert(0, './EgoVLP')
        
        # Try importing EgoVLP modules
        from model.model import FrozenInTime
        print("✓ Can import FrozenInTime")
        
        from args import get_args
        print("✓ Can import get_args")
        
        print("\n✓ EgoVLP imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        print("\nTrying CLIP fallback...")
        
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '-q',
                'git+https://github.com/openai/CLIP.git'
            ])
            import clip
            print("✓ CLIP installed as fallback")
            return True
        except:
            print("❌ CLIP fallback also failed")
            return False


def verify_cuda():
    """Check CUDA availability"""
    print("\n" + "="*60)
    print("Checking CUDA...")
    print("="*60)
    
    import torch
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        
        # Test GPU
        try:
            x = torch.randn(10, 10).cuda()
            y = x @ x.T
            print(f"✓ GPU test passed")
        except:
            print(f"⚠ GPU test failed")
    else:
        print("⚠ No CUDA available - will use CPU (slow!)")


def main():
    """Run complete EgoVLP setup"""
    print("\n" + "="*60)
    print("EGOVLP SETUP")
    print("="*60)
    
    # 1. Install basic dependencies
    install_basic_dependencies()
    
    # 2. Setup EgoVLP repository
    egovlp_ok = setup_egovlp()
    
    # 3. Download checkpoint (optional)
    download_egovlp_checkpoint()
    
    # 4. Test imports
    import_ok = test_egovlp_import()
    
    # 5. Verify CUDA
    verify_cuda()
    
    # Summary
    print("\n" + "="*60)
    print("SETUP SUMMARY")
    print("="*60)
    
    if egovlp_ok and import_ok:
        print("✓ EgoVLP setup complete!")
        print("\nNext steps:")
        print("1. Update paths in extract_egovlp_features.py")
        print("2. Run feature extraction:")
        print("   python extract_egovlp_features.py --splits train")
        print("3. Train models:")
        print("   python train_new_backbone.py --backbone egovlp")
    else:
        print("⚠ Setup incomplete")
        if not egovlp_ok:
            print("  - EgoVLP repository setup failed")
        if not import_ok:
            print("  - EgoVLP imports failed")
            print("  - CLIP fallback is available")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()