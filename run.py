#!/usr/bin/env python3
"""
Entry point script for the Credit Card Churn Prediction application.
This script provides a convenient way to launch the Streamlit application.
"""

import sys
import subprocess
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed."""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'sklearn',
        'joblib'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("Missing required packages. Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        print("Or run: pip install -r requirements.txt")
        return False

    return True

def check_files():
    """Check if required files exist."""
    required_files = [
        'app/main.py',
        'config/config.py',
        'models/churn_model.pkl',
        'web_user/BankChurners.csv'
    ]

    missing_files = []

    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print("Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        return False

    return True

def run_application(port: int = 8501, headless: bool = True):
    """Run the Streamlit application."""
    try:
        app_path = Path('app/main.py')

        if not app_path.exists():
            print("Application file not found. Please ensure you're in the correct directory.")
            return False

        cmd = [sys.executable, '-m', 'streamlit', 'run', str(app_path)]

        if headless:
            cmd.extend(['--server.headless', 'true'])

        cmd.extend(['--server.port', str(port)])

        print(f"Starting Credit Card Churn Prediction application on port {port}...")
        print("Application will be available at: http://localhost:{port}")
        print("Press Ctrl+C to stop the application")

        # Run the application
        subprocess.run(cmd, check=True)

    except subprocess.CalledProcessError as e:
        print(f"Failed to start application: {e}")
        return False
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
        return True
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def main():
    """Main entry point."""
    print("Credit Card Churn Prediction System")
    print("=" * 50)

    # Check if we're in the right directory
    if not Path('app/main.py').exists():
        print("❌ Please run this script from the project root directory")
        print("   Example: python run.py")
        sys.exit(1)

    # Check requirements
    print("Checking requirements...")
    if not check_requirements():
        sys.exit(1)

    # Check files
    print("Checking files...")
    if not check_files():
        sys.exit(1)

    print("All checks passed!")

    # Get port from command line argument
    port = 8501
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port number: {sys.argv[1]}")
            sys.exit(1)

    # Run application
    success = run_application(port=port)

    if success:
        print("\nApplication completed successfully!")
    else:
        print("\nApplication failed to start.")
        sys.exit(1)

if __name__ == "__main__":
    main()