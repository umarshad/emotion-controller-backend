"""
Verification script for Emotion Detection API Server setup
Checks if all dependencies and model files are present
"""
import os
import sys

# Fix encoding for Windows console
if sys.platform == 'win32':
    try:
        import codecs
        if sys.stdout.encoding != 'utf-8':
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except:
        pass  # If encoding fix fails, continue anyway

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"[OK] {description}: {filepath}")
        return True
    else:
        print(f"[MISSING] {description}: {filepath} - NOT FOUND")
        return False

def check_python_package(package_name, import_name=None):
    """Check if a Python package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"[OK] {package_name} is installed")
        return True
    except ImportError:
        print(f"[MISSING] {package_name} is NOT installed")
        print(f"  Install with: pip install {package_name}")
        return False

def main():
    print("=" * 60)
    print("Emotion Detection API Server - Setup Verification")
    print("=" * 60)
    print()
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check model files
    print("Checking model files...")
    model_json = os.path.join(script_dir, "facialemotionmodel.json")
    model_h5 = os.path.join(script_dir, "facialemotionmodel.h5")
    
    model_json_exists = check_file_exists(model_json, "Model JSON file")
    model_h5_exists = check_file_exists(model_h5, "Model weights file")
    
    # Note: haarcascade is included with OpenCV, so we check OpenCV instead
    print()
    print("Checking Python dependencies...")
    
    # Check required packages
    packages = [
        ("flask", "flask"),
        ("flask-cors", "flask_cors"),
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("tensorflow", "tensorflow"),
    ]
    
    all_packages_ok = True
    for package_name, import_name in packages:
        if not check_python_package(package_name, import_name):
            all_packages_ok = False
    
    # Check if streamlit is installed (optional, not needed for API)
    print()
    streamlit_installed = check_python_package("streamlit", "streamlit")
    if streamlit_installed:
        print("  (Note: streamlit is optional - only needed for web interface)")
    
    print()
    print("=" * 60)
    
    # Summary
    if model_json_exists and model_h5_exists and all_packages_ok:
        print("[SUCCESS] All required files and dependencies are present!")
        print("[SUCCESS] API server should start successfully")
        print()
        print("To start the server, run:")
        print("  python api_server.py")
        print("  or")
        print("  start_api_server.bat (Windows)")
        print()
        print("IMPORTANT: For physical Android devices:")
        print("  1. Find your computer's IP address (ipconfig on Windows)")
        print("  2. Set manualBaseUrl in lib/data/config/api_config.dart")
        print("  3. Ensure phone and computer are on same Wi-Fi network")
        return 0
    else:
        print("[ERROR] Some required files or dependencies are missing")
        print()
        if not model_json_exists or not model_h5_exists:
            print("Missing model files. Please ensure:")
            print("  - facialemotionmodel.json")
            print("  - facialemotionmodel.h5")
            print("  are in the same directory as api_server.py")
        if not all_packages_ok:
            print()
            print("Missing Python packages. Install with:")
            print("  pip install -r requirements_mobile.txt")
            print("  (or requirements.txt if you also want Streamlit)")
        return 1

if __name__ == '__main__':
    sys.exit(main())
