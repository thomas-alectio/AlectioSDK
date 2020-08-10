import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="alectio_sdk",
    version="0.1.0",
    author="Alectio",
    author_email="admin@alectio.com",
    description="Integrate customer side ML application with the Alectio Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=["Programming Language :: Python :: 3", "Operating System :: Linux",],
    python_requires=">=3.6",
    package_data={"": ["config.json"]},
    include_package_data=True,
)
