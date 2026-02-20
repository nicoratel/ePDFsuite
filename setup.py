from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ePDFsuite",
    version="0.1.0",
    author="Nicolas Ratel-Ramond",
    author_email="nicolas.ratel-ramond@insa-toulouse.fr",  # Update this
    description="Python library for SAED data processing and PDF extraction with interactive Streamlit GUI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nicoratel/ePDFsuite",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    keywords="SAED electron diffraction PDF RDF distribution function",
    project_urls={
        "Documentation": "https://github.com/nicoratel/ePDFsuite/wiki",
        "Source": "https://github.com/nicoratel/ePDFsuite",
        "Tracker": "https://github.com/nicoratel/ePDFsuite/issues",
    },
)
