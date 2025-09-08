#!/usr/bin/env python3
"""
Quick test for Parakeet TDT model integration
"""

import sys
import os

# Add the app directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_parakeet_import():
    """Test if Parakeet can be imported"""
    try:
        import nemo.collections.asr as nemo_asr
        print("✅ NEMO ASR imported successfully")
        return True
    except ImportError as e:
        print(f"❌ NEMO ASR import failed: {e}")
        return False

def test_parakeet_model_loading():
    """Test if Parakeet TDT model can be loaded"""
    try:
        import nemo.collections.asr as nemo_asr
        print("🎤 Loading Parakeet TDT model...")
        model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v2"
        )
        print("✅ Parakeet TDT model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Parakeet model loading failed: {e}")
        return False

def test_dubbing_system_init():
    """Test if dubbing system can be initialized"""
    try:
        from app import initialize_dubbing_system, PARAKEET_MODEL
        print("🔧 Initializing dubbing system...")
        initialize_dubbing_system()
        
        if PARAKEET_MODEL:
            print("✅ Dubbing system initialized")
            return True
        else:
            print("❌ Parakeet model not initialized")
            return False
    except Exception as e:
        print(f"❌ Dubbing system initialization failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🎤 Parakeet TDT Model Test Suite")
    print("=" * 40)
    
    tests = [
        ("NEMO ASR Import", test_parakeet_import),
        ("Parakeet Model Loading", test_parakeet_model_loading),
        ("Dubbing System Init", test_dubbing_system_init)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 40)
    print("📊 Test Results:")
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    if all(results):
        print("\n🎉 All tests passed! Parakeet TDT is ready for dubbing.")
    else:
        print("\n⚠️ Some tests failed. Check the requirements:")
        print("1. Install NEMO: pip install nemo_toolkit[asr]")
        print("2. Ensure CUDA is available for GPU acceleration")
        print("3. Check internet connection for model download")

if __name__ == "__main__":
    main()