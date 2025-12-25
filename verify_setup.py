#!/usr/bin/env python3
"""
Setup verification script for Stable Audio Open Modal deployment

Run this script to verify your Modal setup is correct before deploying.
"""

import sys
import subprocess
from pathlib import Path


def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_status(check, status, message=""):
    """Print a status line"""
    symbol = "✓" if status else "✗"
    status_text = "PASS" if status else "FAIL"
    print(f"{symbol} {check:<40} [{status_text}]")
    if message:
        print(f"  → {message}")


def check_python_version():
    """Check if Python version is 3.8+"""
    version = sys.version_info
    is_valid = version.major == 3 and version.minor >= 8
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print_status(
        "Python Version (3.8+)",
        is_valid,
        f"Found Python {version_str}"
    )
    return is_valid


def check_modal_installed():
    """Check if Modal is installed"""
    try:
        import modal
        version = modal.__version__
        print_status(
            "Modal Package",
            True,
            f"Version {version} installed"
        )
        return True
    except ImportError:
        print_status(
            "Modal Package",
            False,
            "Run: pip install modal"
        )
        return False


def check_modal_auth():
    """Check if Modal is authenticated"""
    try:
        result = subprocess.run(
            ["modal", "token", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        is_authed = result.returncode == 0
        print_status(
            "Modal Authentication",
            is_authed,
            "Authenticated" if is_authed else "Run: python3 -m modal setup"
        )
        return is_authed
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print_status(
            "Modal Authentication",
            False,
            "Run: python3 -m modal setup"
        )
        return False


def check_files_exist():
    """Check if all required files exist"""
    required_files = [
        "stable_audio_modal.py",
        "web_endpoint.py",
        "example_client.py",
        "requirements.txt",
        "README.md",
    ]
    
    all_exist = True
    for file in required_files:
        exists = Path(file).exists()
        if not exists:
            all_exist = False
        print_status(f"File: {file}", exists)
    
    return all_exist


def check_modal_credits():
    """Check Modal credits (if possible)"""
    try:
        result = subprocess.run(
            ["modal", "profile", "current"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print_status(
                "Modal Credits",
                True,
                "Check your dashboard at modal.com"
            )
            return True
        else:
            print_status(
                "Modal Credits",
                False,
                "Unable to check - verify at modal.com"
            )
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print_status(
            "Modal Credits",
            False,
            "Unable to check - verify at modal.com"
        )
        return False


def main():
    """Run all verification checks"""
    print_header("Stable Audio Open - Setup Verification")
    
    print("\n📋 Checking prerequisites...")
    
    checks = {
        "Python Version": check_python_version(),
        "Modal Package": check_modal_installed(),
        "Modal Auth": check_modal_auth(),
        "Required Files": check_files_exist(),
    }
    
    print("\n" + "-" * 70)
    
    # Summary
    passed = sum(checks.values())
    total = len(checks)
    
    print(f"\n📊 Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n✅ All checks passed! You're ready to deploy.")
        print("\nNext steps:")
        print("  1. Test imports:")
        print("     modal run stable_audio_modal.py::test_imports")
        print("\n  2. Generate your first audio:")
        print('     modal run stable_audio_modal.py --prompt "A peaceful piano melody"')
        print("\n  3. Read QUICKSTART.md for more examples")
        print("\n💰 Cost: ~$0.005-0.01 per generation (A10G GPU)")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        if not checks["Modal Package"]:
            print("  • Install Modal: pip install modal")
        if not checks["Modal Auth"]:
            print("  • Authenticate: python3 -m modal setup")
        if not checks["Required Files"]:
            print("  • Ensure you're in the correct directory")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Verification cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)

