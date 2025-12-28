import torch
from core.config import Config
from core.models.blocks import fetch_input_dim, MLP
from constants import Constants as const

def test_constants():
    """Test that new constants exist"""
    print("=" * 60)
    print("Test 1: Checking Constants")
    print("=" * 60)
    
    try:
        print(f"✓ EGOVLP constant: {const.EGOVLP}")
        print(f"✓ PERCEPTION_ENCODER constant: {const.PERCEPTION_ENCODER}")
        print(f"✓ VIDEOMAE constant: {const.VIDEOMAE}")
        return True
    except AttributeError as e:
        print(f"✗ Error: {e}")
        print("Make sure you added the constants to constants.py")
        return False


def test_feature_dimensions():
    """Test that feature dimensions are correct"""
    print("\n" + "=" * 60)
    print("Test 2: Checking Feature Dimensions")
    print("=" * 60)
    
    config = Config()
    
    # Test each backbone
    test_cases = [
        (const.OMNIVORE, 1024),
        (const.SLOWFAST, 400),
        (const.X3D, 400),
        (const.RESNET3D, 400),
        (const.EGOVLP, 768),
        (const.PERCEPTION_ENCODER, 1024),
        (const.VIDEOMAE, 768),
    ]
    
    all_passed = True
    for backbone, expected_dim in test_cases:
        config.backbone = backbone
        try:
            actual_dim = fetch_input_dim(config)
            if actual_dim == expected_dim:
                print(f"✓ {backbone:20s}: {actual_dim:4d} (correct)")
            else:
                print(f"✗ {backbone:20s}: {actual_dim:4d} (expected {expected_dim})")
                all_passed = False
        except Exception as e:
            print(f"✗ {backbone:20s}: Error - {e}")
            all_passed = False
    
    return all_passed


def test_model_creation():
    """Test that models can be created with new backbones"""
    print("\n" + "=" * 60)
    print("Test 3: Testing Model Creation")
    print("=" * 60)
    
    config = Config()
    
    # Test with EgoVLP
    config.backbone = const.EGOVLP
    input_dim = fetch_input_dim(config)
    
    try:
        model = MLP(input_dim, 512, 1)
        print(f"✓ MLP model created with EgoVLP (input_dim={input_dim})")
        print(f"  Model architecture:")
        print(f"    Layer 1: {input_dim} -> 512")
        print(f"    Layer 2: 512 -> 1")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")
        
        return True
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        return False


def test_forward_pass():
    """Test forward pass with different input shapes"""
    print("\n" + "=" * 60)
    print("Test 4: Testing Forward Pass")
    print("=" * 60)
    
    config = Config()
    config.backbone = const.EGOVLP
    input_dim = fetch_input_dim(config)
    
    model = MLP(input_dim, 512, 1)
    
    test_cases = [
        ("Single clip", torch.randn(1, input_dim)),
        ("Multiple clips", torch.randn(1, 10, input_dim)),
        ("Batch of clips", torch.randn(4, 10, input_dim)),
    ]
    
    all_passed = True
    for name, test_input in test_cases:
        try:
            output = model(test_input)
            print(f"✓ {name:20s}: input {list(test_input.shape)} -> output {list(output.shape)}")
        except Exception as e:
            print(f"✗ {name:20s}: Error - {e}")
            all_passed = False
    
    return all_passed


def test_imagebind_multimodal():
    """Test ImageBind with multiple modalities"""
    print("\n" + "=" * 60)
    print("Test 5: Testing ImageBind Multi-modal")
    print("=" * 60)
    
    config = Config()
    config.backbone = const.IMAGEBIND
    
    test_cases = [
        (["video"], 1024),
        (["video", "audio"], 2048),
        (["video", "audio", "text"], 3072),
        (["video", "audio", "text", "depth"], 4096),
    ]
    
    all_passed = True
    for modalities, expected_dim in test_cases:
        config.modality = modalities
        actual_dim = fetch_input_dim(config)
        
        if actual_dim == expected_dim:
            print(f"✓ {str(modalities):30s}: {actual_dim:4d} (correct)")
        else:
            print(f"✗ {str(modalities):30s}: {actual_dim:4d} (expected {expected_dim})")
            all_passed = False
    
    return all_passed


def main():
    """Run all tests"""
    print("\n" + "🧪 " * 20)
    print("BACKBONE INTEGRATION TEST SUITE")
    print("🧪 " * 20 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Constants", test_constants()))
    results.append(("Feature Dimensions", test_feature_dimensions()))
    results.append(("Model Creation", test_model_creation()))
    results.append(("Forward Pass", test_forward_pass()))
    results.append(("ImageBind Multi-modal", test_imagebind_multimodal()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:30s}: {status}")
        all_passed = all_passed and passed
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All tests passed! You're ready to train with new backbones!")
        print("\nNext steps:")
        print("1. Extract features: python extract_features.py --backbone egovlp --split train")
        print("2. Train model: python train_er.py --backbone egovlp --variant mlp --split S")
    else:
        print("\n⚠️  Some tests failed. Please fix the errors above before proceeding.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)