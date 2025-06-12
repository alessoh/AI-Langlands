#!/usr/bin/env python3
"""
Verify that all dependencies are correctly installed for the AI Langlands project.
"""

import sys
import importlib
from importlib import metadata

def check_package(package_name, import_name=None):
    """Check if a package is installed and can be imported."""
    if import_name is None:
        import_name = package_name.split('==')[0].replace('-', '_')
    
    try:
        # Check if package is installed
        version = metadata.version(import_name)
        
        # Try to import it
        importlib.import_module(import_name)
        
        return True, version
    except Exception as e:
        return False, str(e)

def main():
    """Main verification function."""
    print("Verifying AI Langlands Program dependencies...\n")
    
    packages = [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("tensorflow", "tensorflow"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("plotly", "plotly"),
        ("sympy", "sympy"),
        ("tqdm", "tqdm"),
        ("joblib", "joblib"),
        ("click", "click"),
        ("h5py", "h5py"),
    ]
    
    all_ok = True
    
    for package, import_name in packages:
        success, result = check_package(package, import_name)
        
        if success:
            print(f"✓ {import_name:<15} {result:<10} OK")
        else:
            print(f"✗ {import_name:<15} FAILED: {result}")
            all_ok = False
    
    print("\n" + "="*50)
    
    if all_ok:
        print("✓ All dependencies installed successfully!")
        print("\nYou can now run: python murmuration_discovery.py")
        return 0
    else:
        print("✗ Some dependencies are missing.")
        print("\nPlease run: pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())