import setuptools


with open("README.md", "r") as f:
    long_description = f.read()


setuptools.setup(
    name="teal",
    version="0.0.3",
    author="Amit Yadav",
    author_email="amit.yadav.iitr@gmail.com",
    description="teal - TensorFlow Audio Layers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/am1tyadav/teal.git",
    packages=[
        "teal", "teal.augment",
        "teal.feature"
    ],
    install_requires=[
        "tensorflow>=2"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
