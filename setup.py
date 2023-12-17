import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="corpus-pruner",
    version="0.1.0",
    author="L.Beaudoux",
    description="A Python library for pruning the sentences of a corpus",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LBeaudoux/corpus-pruner",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
    python_requires=">=3.8",
    install_requires=[
        "iso639-lang>=2.2.2",
        "pandas>=2.0.3",
        "wordfreq[cjk]>=3.1.1",
    ],
)
