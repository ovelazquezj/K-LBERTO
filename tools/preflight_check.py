#!/usr/bin/env python3
"""
Pre-flight Check: Verify everything is ready for WikiANN download and K-BERT training

This script verifies:
1. Required Python packages are installed
2. Output directory exists or can be created
3. Internet connectivity
4. K-BERT directories exist
5. WikidataES Knowledge Graph is properly registered
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Tuple, List

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_header(text: str):
    print(f"\n{Colors.BLUE}{'='*60}")
    print(f"{text:^60}")
    print(f"{'='*60}{Colors.END}\n")

def print_success(text: str):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text: str):
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_info(text: str):
    print(f"  {text}")

def check_python_version() -> bool:
    """Check if Python version is 3.8+"""
    print_header("Python Version Check")
    
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print_success(f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python {version.major}.{version.minor} detected (requires 3.8+)")
        return False

def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """Check if a Python package is installed"""
    import_name = import_name or package_name
    
    try:
        __import__(import_name)
        # Get version if available
        try:
            version = __import__(import_name).__version__
            return True, f"{package_name}=={version}"
        except:
            return True, package_name
    except ImportError:
        return False, None

def check_required_packages() -> bool:
    """Check if all required packages are installed"""
    print_header("Required Packages Check")
    
    packages = [
        ('datasets', 'datasets'),
        ('huggingface_hub', 'huggingface_hub'),
        ('torch', 'torch'),
        ('transformers', 'transformers'),
    ]
    
    all_ok = True
    for package_name, import_name in packages:
        ok, version = check_package(package_name, import_name)
        if ok:
            print_success(version)
        else:
            print_error(f"{package_name} not installed")
            print_info(f"Install with: pip install {package_name} --break-system-packages")
            all_ok = False
    
    return all_ok

def check_directories() -> bool:
    """Check if required directories exist"""
    print_header("Directory Structure Check")
    
    directories = {
        "K-BERT Project": "~/projects/K-BERT",
        "Brain (KGs)": "~/projects/K-BERT/brain/kgs",
        "Models": "~/projects/K-BERT/models",
        "Output": "./pipeline-dataset/outputs",
    }
    
    all_ok = True
    for name, path in directories.items():
        expanded_path = Path(path).expanduser()
        if expanded_path.exists():
            print_success(f"{name}: {expanded_path}")
        else:
            print_warning(f"{name} not found: {expanded_path}")
            # Check if it's critical
            if "K-BERT Project" in name:
                print_error("K-BERT project not found. Ensure it's cloned and configured.")
                all_ok = False

    return all_ok

def check_knowledge_graph() -> bool:
    """Check if WikidataES Knowledge Graph is available"""
    print_header("Knowledge Graph Check")
    
    kg_path = Path("~/projects/K-BERT/brain/kgs/WikidataES_CLEAN_v251109.spo").expanduser()
    
    if kg_path.exists():
        size_mb = kg_path.stat().st_size / (1024 * 1024)
        print_success(f"WikidataES_CLEAN_v251109.spo found ({size_mb:.1f} MB)")
        
        # Check if it's in config.py
        config_path = Path("~/projects/K-BERT/brain/config.py").expanduser()
        if config_path.exists():
            with open(config_path, 'r') as f:
                content = f.read()
                if 'WikidataES_CLEAN_v251109' in content:
                    print_success("Knowledge Graph registered in brain/config.py")
                    return True
                else:
                    print_warning("Knowledge Graph NOT registered in brain/config.py")
                    print_info("Add this line to KGS dictionary:")
                    print_info("  'WikidataES_CLEAN_v251109': os.path.join(FILE_DIR_PATH, 'kgs/WikidataES_CLEAN_v251109.spo'),")
                    return False
    else:
        print_error(f"WikidataES_CLEAN_v251109.spo not found: {kg_path}")
        print_info("Expected in: ~/projects/K-BERT/brain/kgs/")
        return False

def check_internet_connectivity() -> bool:
    """Check if we can reach Hugging Face"""
    print_header("Internet Connectivity Check")
    
    try:
        import urllib.request
        urllib.request.urlopen('https://huggingface.co', timeout=5)
        print_success("Can reach Hugging Face (huggingface.co)")
        return True
    except Exception as e:
        print_error(f"Cannot reach Hugging Face: {str(e)}")
        print_info("Ensure you have internet connectivity")
        print_info("If behind a proxy, configure urllib/requests accordingly")
        return False

def check_disk_space() -> bool:
    """Check if there's enough disk space"""
    print_header("Disk Space Check")
    
    output_dir = Path("./pipeline-dataset/outputs").expanduser()
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get disk usage
        import shutil
        disk_info = shutil.disk_usage(output_dir)
        free_gb = disk_info.free / (1024 ** 3)
        
        if free_gb >= 1:  # At least 1 GB
            print_success(f"Disk space: {free_gb:.1f} GB free")
            return True
        else:
            print_warning(f"Low disk space: {free_gb:.1f} GB free (recommend >1 GB)")
            return True  # Not critical
    except Exception as e:
        print_warning(f"Could not check disk space: {str(e)}")
        return True  # Not critical

def generate_summary(checks: dict) -> None:
    """Generate a summary of all checks"""
    print_header("Pre-flight Check Summary")
    
    total = len(checks)
    passed = sum(1 for v in checks.values() if v)
    
    print(f"Results: {passed}/{total} checks passed\n")
    
    for name, result in checks.items():
        status = f"{Colors.GREEN}PASS{Colors.END}" if result else f"{Colors.RED}FAIL{Colors.END}"
        print(f"  [{status}] {name}")
    
    print()
    
    if passed == total:
        print_success("All checks passed! Ready to run WikiANN downloader.")
        print_info("\nNext step:")
        print_info("  python download_prepare_wikiann_es.py")
        return True
    else:
        print_error(f"{total - passed} check(s) failed. Fix issues above before proceeding.")
        return False

def main():
    print_header("WikiANN Pre-flight Check")
    
    checks = {
        "Python Version (3.8+)": check_python_version(),
        "Required Packages": check_required_packages(),
        "Directory Structure": check_directories(),
        "Knowledge Graph (WikidataES)": check_knowledge_graph(),
        "Internet Connectivity": check_internet_connectivity(),
        "Disk Space (>1 GB)": check_disk_space(),
    }
    
    success = generate_summary(checks)
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
