import setuptools

def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

install_reqs = parse_requirements('./requirements.txt')

with open("README.md", "r") as fh:
    long_description = fh.read()

reqs = install_reqs
setuptools.setup(
    name="alectio_sdk",
    version="0.0.3",
    author="Alectio",
    author_email="admin@alectio.com",
    description="Integrate customer side ML application with the Alectio Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.6",
    install_requires=install_reqs,
    package_data={"": ["config.json"]},
    include_package_data=True,
)
