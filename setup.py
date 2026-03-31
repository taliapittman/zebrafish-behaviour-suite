from setuptools import setup, find_packages

setup(
    name="zf-bhv-suite",
    version="0.1.0",
    author="Talia Pittman",
    author_email="your.email@ucl.ac.uk",
    description="Zebrafish behavioural analysis suite for FramebyFrame outputs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/zebrafish-behaviour-suite",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "dabest>=2023.2.14",
        "matplotlib>=3.3.0",
        "Pillow>=8.0.0",
        "scipy>=1.7.0",
        "numba>=0.54.0",
    ],
)
