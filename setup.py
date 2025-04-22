from setuptools import setup, find_packages

setup(
    name="employee_retention",
    version="1.0.0",
    packages=find_packages(where="employee_retention"),
    package_dir={"": "employee_retention"},
    install_requires=open("requirements.txt").read().splitlines(),
    author="Sonny Bigras-Dewan",
    description="A calibrated ML pipeline for predicting employee retention risk with SHAP explanations.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
