import setuptools


with open("README.md", "r") as f:
    long_description = f.read()


setuptools.setup(
    name="Teal",
    version="0.0.3",
    author="Amit Yadav",
    author_email="amit.yadav.iitr@gmail.com",
    description="Teal - TensorFlow Audio Layers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/am1tyadav/teal.git",
    packages=setuptools.find_packages(exclude=["tests"]),
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
