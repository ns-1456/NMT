#!/usr/bin/env python3
"""
Helper script to fix transformers backend issues.

This script helps resolve the "Backend should be defined in BACKENDS_MAPPING" error
by either uninstalling TensorFlow (if not needed) or properly configuring it.
"""

import subprocess
import sys

def check_tensorflow():
    """Check if TensorFlow is installed."""
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow is installed (version: {tf.__version__})")
        return True
    except ImportError:
        print("❌ TensorFlow is not installed")
        return False
    except Exception as e:
        print(f"⚠️  TensorFlow is installed but has issues: {e}")
        return True

def uninstall_tensorflow():
    """Uninstall TensorFlow."""
    print("\n🔄 Uninstalling TensorFlow...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "tensorflow", "tensorflow-cpu", "tensorflow-gpu"])
        print("✅ TensorFlow uninstalled successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to uninstall TensorFlow: {e}")
        return False

def main():
    print("🔧 Transformers Backend Fix Script")
    print("=" * 50)
    
    if check_tensorflow():
        print("\n⚠️  TensorFlow is installed, which may cause conflicts with transformers.")
        print("Since this project uses PyTorch, TensorFlow is not needed.")
        print("\nOptions:")
        print("1. Uninstall TensorFlow (recommended)")
        print("2. Keep TensorFlow and use workarounds in scripts")
        
        response = input("\nWould you like to uninstall TensorFlow? (y/n): ").strip().lower()
        if response == 'y':
            if uninstall_tensorflow():
                print("\n✅ Fix complete! You can now run demo.py without issues.")
                print("If you still see errors, try restarting your Python environment.")
            else:
                print("\n⚠️  Could not uninstall TensorFlow automatically.")
                print("Please run manually: pip uninstall tensorflow tensorflow-cpu tensorflow-gpu")
        else:
            print("\n⚠️  Keeping TensorFlow. The scripts should work with the workarounds.")
            print("If you still see errors, try:")
            print("  export TRANSFORMERS_BACKEND=pt")
            print("  python3 demo.py")
    else:
        print("\n✅ TensorFlow is not installed. The backend error might be from a corrupted transformers installation.")
        print("Try reinstalling transformers:")
        print("  pip uninstall transformers")
        print("  pip install transformers")

if __name__ == "__main__":
    main()
