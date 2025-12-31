"""
SynDX Setup Configuration
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="syndx",
    version="0.1.0",
    author="Chatchai Tritham, Chakkrit Snae Namahoot",
    author_email="chatchai.tritham@nu.ac.th",
    description="Explainable AI-Driven Synthetic Data Generation for Vestibular Disorders",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chatchai.tritham/SynDX",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.3",
        "scipy>=1.11.1",
        "pandas>=2.0.3",
        "scikit-learn>=1.3.0",
        "torch>=2.0.1",
        "shap>=0.42.1",
        "xgboost>=1.7.6",
        "diffprivlib>=0.6.0",
        "fhir.resources>=7.0.2",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "tqdm>=4.66.1",
        "pyyaml>=6.0.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.2",
            "pytest-cov>=4.1.0",
            "black>=23.9.1",
            "flake8>=6.1.0",
            "jupyter>=1.0.0",
        ],
        "docs": [
            "sphinx>=7.2.5",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "syndx-generate=syndx.cli:generate",
            "syndx-validate=syndx.cli:validate",
        ],
    },
)
