from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mfi-loan-risk-assessment",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="MFI Loan Risk Assessment using Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mfi-loan-risk-assessment",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.1.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "flask>=2.2.0",
        "flask-cors>=3.0.10",
        "joblib>=1.2.0",
        "scipy>=1.9.0",
        "plotly>=5.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "requests>=2.28.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mfi-train=ml_models:main",
            "mfi-generate-data=generate_mfi_data:main",
            "mfi-serve=app:main",
        ],
    },
)