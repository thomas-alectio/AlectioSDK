import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="alectio_sdk",
    version="0.0.1",
    author="Alectio",
    author_email="admin@alectio.com",
    description="Integrate customer side ML application with the Alectio Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
<<<<<<< HEAD
    packages=setuptools.find_packages(exclude=["examples.*", "examples", "test.*", "test", "docs.*", "docs"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Linux",
    ],
=======
    packages=setuptools.find_packages(),
    classifiers=["Programming Language :: Python :: 3", "Operating System :: Linux",],
>>>>>>> 1b0d26ad1ef39054439fa9dfda6bf1911b6b6ef1
    python_requires=">=3.6",
    package_data={"": ["config.json"]},
    include_package_data=True,
)
