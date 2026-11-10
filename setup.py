from setuptools import setup, find_packages

setup(
    name="cangio_graphnn",
    version="1.0.0",
    author="Can Gio Mangrove Remote Sensing Research Team",
    description="Graph Neural Networks and Bi-temporal Foundation Architectures for Mangrove Canopy Disturbance Mapping",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/cangio-earth/cangio_graphnn",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "rasterio>=1.3.0",
        "opencv-python>=4.7.0",
        "numpy>=1.23.0,<2.0.0",
        "scipy>=1.10.0",
        "matplotlib>=3.6.0",
        "scikit-learn>=1.2.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
