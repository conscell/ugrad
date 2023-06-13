import setuptools

with open("README.md", "r") as fi:
    long_description = fi.read()

setuptools.setup(
    name="ugrad",
    version="0.1.0",
    author="conscell",
    description="μGrad is a lightweight automatic differentiation engine.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/conscell/ugrad",
    packages=setuptools.find_packages(),
    install_requires=[         
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
