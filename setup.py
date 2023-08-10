from setuptools import setup, find_packages

setup(
    name="progressive-neural-architecture-search-torch",
    version="0.0.1", 
    description="A PyTorch implementation of Progressive Neural Architecture Search. https://arxiv.org/pdf/1712.00559.pdf",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    author="juvi21",
    author_email="juv121@skiff.com",
    url="https://github.com/juvi21/progressive-neural-architecture-search-pytorch",
    install_requires=[
        "torch",
        "triton"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
