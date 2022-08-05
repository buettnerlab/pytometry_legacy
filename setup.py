import setuptools

#with open("README.md", "r") as fh:
#    long_description = fh.read()
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()
    requirements = [x for x in requirements if not x.startswith("#") and x != ""]


setuptools.setup(
    name="pytometrylegacy", # Replace with your own username
    version="0.0.1",
    author="Felix Hempel, Thomas Ryborz, Maren Buettner",
    author_email="maren.buettner@helmholtz-muenchen.de",
    description="Tools for Flow Cytometry Analysis using the Anndata-dataformat",
#    long_description=long_description,
#    long_description_content_type="text/markdown",
    url="https://github.com/buettnerlab/pytometry_legacy",
    packages = setuptools.find_packages(),
    install_requires = requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
