import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="release-deployment-dashboard",
    version="1.0.0",
    description="Dashboard website to display latest GitLab release statuses",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/myorg/release-deployment-dashboard",
    packages=setuptools.find_packages(),
    python_requires='>=3.10',
    install_requires=["flask ~= 2.0"]
)
