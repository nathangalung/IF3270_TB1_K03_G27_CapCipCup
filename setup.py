from setuptools import setup, find_packages

setup(
    name="ffnn_from_scratch",
    version="0.1.0",
    author="CapCipCup Team",
    author_email="team.capcip@example.com",
    description="Feedforward Neural Network implementation from scratch",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/IF3270_TB1_K03_G27_CapCipCup",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "ipywidgets>=7.6.0",
        ],
    },
    project_urls={
        "Neural Networks Guide": "https://www.deeplearningbook.org/",
        "MNIST Dataset": "http://yann.lecun.com/exdb/mnist/",
        "Backpropagation Tutorial": "https://www.3blue1brown.com/topics/neural-networks",
        "Weight Initialization": "https://www.deeplearning.ai/ai-notes/initialization/",
        "Activation Functions": "https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/",
        "Loss Functions": "https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html",
    },
)