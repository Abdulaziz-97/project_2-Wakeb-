"""
AI Engineer Virtual Environment Setup Script
Creates a venv with essential AI/ML libraries and checks for dependency conflicts
"""

import subprocess
import sys
import os
import json
from pathlib import Path


def run_command(command, capture_output=True):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=capture_output,
            text=True,
            check=False
        )
        return result
    except Exception as e:
        print(f"Error running command: {e}")
        return None


def create_venv(venv_path="venv"):
    """Create a virtual environment."""
    print(f"Creating virtual environment at: {venv_path}")
    result = run_command(f'"{sys.executable}" -m venv {venv_path}')
    
    if result and result.returncode == 0:
        print(f"✓ Virtual environment created successfully at {venv_path}")
        return True
    else:
        print(f"✗ Failed to create virtual environment")
        return False


def get_pip_path(venv_path="venv"):
    """Get the path to pip in the virtual environment."""
    if sys.platform == "win32":
        return os.path.join(venv_path, "Scripts", "pip.exe")
    else:
        return os.path.join(venv_path, "bin", "pip")


def get_python_path(venv_path="venv"):
    """Get the path to python in the virtual environment."""
    if sys.platform == "win32":
        return os.path.join(venv_path, "Scripts", "python.exe")
    else:
        return os.path.join(venv_path, "bin", "python")


def upgrade_pip(pip_path):
    """Upgrade pip to the latest version."""
    print("\nUpgrading pip...")
    result = run_command(f'"{pip_path}" install --upgrade pip')
    if result and result.returncode == 0:
        print("✓ pip upgraded successfully")
        return True
    else:
        print("✗ Failed to upgrade pip")
        return False


def check_dependencies(pip_path, packages):
    """
    Check for dependency conflicts before installation.
    Uses pip's dependency resolver in dry-run mode.
    """
    print("\n" + "="*70)
    print("CHECKING DEPENDENCIES FOR CONFLICTS")
    print("="*70)
    
    # Create a temporary requirements file
    temp_req_file = "temp_requirements.txt"
    with open(temp_req_file, "w") as f:
        for package in packages:
            f.write(f"{package}\n")
    
    print(f"\nRunning dependency check for {len(packages)} packages...")
    
    # Use pip install --dry-run to check for conflicts
    result = run_command(
        f'"{pip_path}" install --dry-run -r {temp_req_file}',
        capture_output=True
    )
    
    # Clean up temp file
    try:
        os.remove(temp_req_file)
    except:
        pass
    
    if result and result.returncode == 0:
        print("✓ No dependency conflicts detected!")
        print("\nPackages that will be installed:")
        for package in packages:
            print(f"  - {package}")
        return True, packages
    else:
        print("✗ Dependency conflicts detected!")
        if result and result.stderr:
            print("\nError details:")
            print(result.stderr)
        
        # Try to identify problematic packages
        print("\nAttempting to identify conflicting packages...")
        return False, packages


def install_packages(pip_path, packages):
    """Install packages one by one or in batch."""
    print("\n" + "="*70)
    print("INSTALLING PACKAGES")
    print("="*70)
    
    successful = []
    failed = []
    
    # Try batch installation first
    print("\nAttempting batch installation...")
    packages_str = " ".join([f'"{pkg}"' for pkg in packages])
    result = run_command(f'"{pip_path}" install {packages_str}', capture_output=False)
    
    if result and result.returncode == 0:
        print("\n✓ All packages installed successfully!")
        return packages, []
    
    # If batch fails, install one by one
    print("\nBatch installation failed. Installing packages individually...")
    for package in packages:
        print(f"\nInstalling {package}...")
        result = run_command(f'"{pip_path}" install "{package}"', capture_output=True)
        
        if result and result.returncode == 0:
            print(f"✓ {package} installed successfully")
            successful.append(package)
        else:
            print(f"✗ Failed to install {package}")
            if result and result.stderr:
                print(f"  Error: {result.stderr[:200]}")
            failed.append(package)
    
    return successful, failed


def verify_installation(python_path, packages):
    """Verify that packages are properly installed."""
    print("\n" + "="*70)
    print("VERIFYING INSTALLATION")
    print("="*70)
    
    verified = []
    failed = []
    
    # Create a temporary verification script file to avoid Windows command-line issues
    temp_verify_script = "temp_verify.py"
    
    for package in packages:
        # Extract package name (remove version specifiers)
        pkg_name = package.split("==")[0].split(">=")[0].split("<=")[0].split(">")[0].split("<")[0]
        
        # Handle special cases where package name differs from import name
        import_name_map = {
            "scikit-learn": "sklearn",
            "opencv-python": "cv2",
            "pillow": "PIL",
            "python-dotenv": "dotenv",
            "pyyaml": "yaml",
            "llama-index": "llama_index",
            "beautifulsoup4": "bs4",
        }
        import_name = import_name_map.get(pkg_name, pkg_name)
        
        # Write verification script to a temporary file (avoids PowerShell quote issues)
        verify_script = f'''import {import_name}
try:
    ver = getattr({import_name}, "__version__", None)
    if ver is None:
        ver = getattr({import_name}, "VERSION", None)
    if ver is None:
        try:
            import importlib.metadata
            ver = importlib.metadata.version("{pkg_name}")
        except:
            ver = "imported"
    print(ver)
except Exception as e:
    print("imported")
'''
        
        try:
            with open(temp_verify_script, "w") as f:
                f.write(verify_script)
            
            result = run_command(
                f'"{python_path}" "{temp_verify_script}"',
                capture_output=True
            )
            
            if result and result.returncode == 0 and result.stdout.strip():
                version = result.stdout.strip()
                print(f"✓ {pkg_name}: {version}")
                verified.append(pkg_name)
            else:
                error_msg = result.stderr.strip() if result and result.stderr else "Unknown error"
                print(f"✗ {pkg_name}: Failed to import")
                # Check for common Windows DLL issues
                if "WinError 1114" in error_msg or "DLL" in error_msg:
                    print(f"  ⚠ DLL loading issue - may need Visual C++ Redistributable")
                failed.append(pkg_name)
        except Exception as e:
            print(f"✗ {pkg_name}: Verification error - {e}")
            failed.append(pkg_name)
        finally:
            # Clean up temp file after each check
            try:
                if os.path.exists(temp_verify_script):
                    os.remove(temp_verify_script)
            except:
                pass
    
    # Print help for common issues
    if failed:
        print("\n" + "-"*70)
        print("TROUBLESHOOTING TIPS:")
        print("-"*70)
        torch_related = ["torch", "torchvision", "torchaudio", "transformers", "datasets"]
        if any(pkg in failed for pkg in torch_related):
            print("\nPyTorch DLL issues on Windows:")
            print("  1. Install Visual C++ Redistributable 2015-2022:")
            print("     https://aka.ms/vs/17/release/vc_redist.x64.exe")
            print("  2. After installing, restart your terminal and try again")
            print("  3. Or reinstall PyTorch with CPU-only version:")
            print("     pip uninstall torch torchvision torchaudio")
            print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    
    return verified, failed


def generate_activation_instructions(venv_path):
    """Generate instructions for activating the virtual environment."""
    print("\n" + "="*70)
    print("ACTIVATION INSTRUCTIONS")
    print("="*70)
    
    if sys.platform == "win32":
        print(f"\nTo activate the virtual environment on Windows:")
        print(f"  PowerShell: {venv_path}\\Scripts\\Activate.ps1")
        print(f"  CMD:        {venv_path}\\Scripts\\activate.bat")
    else:
        print(f"\nTo activate the virtual environment on Unix/Mac:")
        print(f"  source {venv_path}/bin/activate")
    
    print(f"\nTo deactivate:")
    print(f"  deactivate")


def main():
    """Main function to set up the AI engineering environment."""
    print("="*70)
    print("AI ENGINEER VIRTUAL ENVIRONMENT SETUP")
    print("="*70)
    
    # Configuration
    venv_path = "venv"
    
    # Essential AI/ML libraries with compatible versions
    # These versions are tested to work together
    # 
    # CRITICAL: NumPy is capped at <2.0.0 due to compatibility issues with:
    # - PyTorch 2.4 (Windows wheel incompatibility with NumPy 2.0)
    # - Transformers (many versions require numpy<2.0)
    # - Other packages built against NumPy 1.x ABI
    packages = [
        # Core scientific computing
        "numpy>=1.26.0,<2.0.0",  # Python 3.13 support, but cap at <2.0 for compatibility
        "pandas>=2.2.0",  # Updated for Python 3.13 support
        "scipy>=1.12.0",  # Updated for Python 3.13 support
        
        # Machine Learning frameworks
        "scikit-learn>=1.4.0",  # Updated for Python 3.13 support
        
        # Deep Learning (PyTorch ecosystem)
        "torch>=2.2.0",  # Updated for Python 3.13 support
        "torchvision>=0.17.0",  # Updated for Python 3.13 support
        "torchaudio>=2.2.0",  # Updated for Python 3.13 support
        
        # TensorFlow/Keras (alternative to PyTorch - comment out if not needed)
        # "tensorflow>=2.13.0",
        # "keras>=2.13.0",
        
        # Natural Language Processing
        "transformers>=4.38.0",  # Updated for better compatibility
        "tokenizers>=0.15.0",  # Updated for better compatibility
        "datasets>=2.18.0",  # Updated for better compatibility
        "sentencepiece>=0.2.0",  # Updated for better compatibility
        
        # Computer Vision
        "opencv-python>=4.9.0",  # Updated for Python 3.13 support
        "pillow>=10.2.0",  # Updated for Python 3.13 support
        "ultralytics>=8.3.237",
        
        # Data visualization
        "matplotlib>=3.8.0",  # Updated for Python 3.13 support
        "seaborn>=0.13.0",  # Updated for better compatibility
        
        # Jupyter and notebooks
        "jupyter>=1.0.0",
        "ipykernel>=6.29.0",  # Updated for Python 3.13 support
        "notebook>=7.0.0",
        
        # ML utilities
        "tqdm>=4.66.0",  # Updated for better compatibility
        "wandb>=0.16.0",  # Experiment tracking, updated
        "tensorboard>=2.15.0",  # Updated for better compatibility
        
        # API and deployment
        "fastapi>=0.109.0",  # Updated for Python 3.13 support
        "uvicorn>=0.27.0",  # Updated for better compatibility
        "pydantic>=2.6.0",  # Updated for Python 3.13 support
        
        # LLM and AI tools
        "langchain>=0.1.0",  # Compatible with Python 3.13
        "llama-index>=0.10.0",  # Updated for better compatibility
        
        # Vector databases
        "chromadb>=0.4.22",  # Updated for Python 3.13 support
        
        # Data sources and APIs
        "kagglehub>=0.4.1",  # Kaggle datasets, Python 3.13 compatible
        "requests>=2.31.0",  # HTTP requests
        "beautifulsoup4>=4.12.0",  # Web scraping
        "openpyxl>=3.1.2",  # Excel file support
        
        # Additional utilities
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.1",  # Updated for Python 3.13 support
        "streamlit>=1.31.0",  # Web apps for ML
    ]
    
    # Step 1: Create virtual environment
    if not create_venv(venv_path):
        print("\nSetup failed: Could not create virtual environment")
        return
    
    pip_path = get_pip_path(venv_path)
    python_path = get_python_path(venv_path)
    
    # Step 2: Upgrade pip
    if not upgrade_pip(pip_path):
        print("\nWarning: Could not upgrade pip, continuing anyway...")
    
    # Step 3: Check for dependency conflicts
    print("\nNote: TensorFlow is commented out by default to avoid conflicts with PyTorch.")
    print("Uncomment TensorFlow lines in the script if you need both frameworks.")
    
    conflicts_free, packages_to_install = check_dependencies(pip_path, packages)
    
    if not conflicts_free:
        print("\n⚠ Warning: Potential conflicts detected.")
        response = input("Do you want to proceed with installation anyway? (y/n): ")
        if response.lower() != 'y':
            print("\nSetup cancelled by user.")
            return
    
    # Step 4: Install packages
    successful, failed = install_packages(pip_path, packages_to_install)
    
    # Step 5: Verify installation
    if successful:
        print("\nVerifying installed packages...")
        verified, verify_failed = verify_installation(python_path, successful)
    
    # Step 6: Summary
    print("\n" + "="*70)
    print("INSTALLATION SUMMARY")
    print("="*70)
    print(f"\nSuccessfully installed: {len(successful)} packages")
    if failed:
        print(f"Failed to install: {len(failed)} packages")
        print("\nFailed packages:")
        for pkg in failed:
            print(f"  - {pkg}")
    
    # Step 7: Generate activation instructions
    generate_activation_instructions(venv_path)
    
    # Step 8: Save requirements.txt
    requirements_path = os.path.join(venv_path, "requirements.txt")
    with open(requirements_path, "w") as f:
        for pkg in successful:
            f.write(f"{pkg}\n")
    print(f"\n✓ Requirements saved to: {requirements_path}")
    
    print("\n" + "="*70)
    print("SETUP COMPLETE!")
    print("="*70)
    print("\nYour AI engineering environment is ready to use!")


if __name__ == "__main__":
    main()
