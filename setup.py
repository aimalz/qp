from setuptools import setup

setup(
    name="qp",
    version="0.1",
    author="Alex Malz, Phil Marshall",
    author_email="aimalz@nyu.edu, pjm@slac.stanford.edu",
    url = "https://github.com/aimalz/qp",
    packages=["qp"],
    description="Quantile parametrization of probability distribution functions",
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        ],
    install_requires=["matplotlib", "numpy", "pathos", "scipy", "sklearn", "astropy", "h5py", "flexcode"]
)
