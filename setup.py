from setuptools import setup, find_packages

setup(
    name="biketrace",
    version="1.0.0",
    description="High-Speed CCTV Forensics & Stolen 2-Wheeler Suspect Identification Toolkit",
    author="Antigravity Forensics",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.2.0",
        "pillow>=9.5.0",
        "rich>=13.0.0",
        "pydantic>=2.0.0",
        "tqdm>=4.65.0",
        "streamlit>=1.25.0",
    ],
    entry_points={
        "console_scripts": [
            "biketrace=biketrace.cli:main",
        ],
    },
)
