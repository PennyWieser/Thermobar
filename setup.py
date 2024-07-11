
#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path


this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, 'src', 'Thermobar', '_version.py'), encoding='utf-8') as f:
    exec(f.read())

with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name="Thermobar",
    version=__version__,
    author="Penny, Maurizio, Jordan, Eric, Sinan",
    author_email="penny.wieser@gmail.com",
    description="Thermobar",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PennyWieser/Thermobar",
    package_dir={'': 'src'},  # Optional
    packages=find_packages(where='src'),
    package_data={
        # Assuming 'Thermobar' is the package name
        # Specify both CSVs in the 'Cali_Datasets_CSVs' folder and pickle files
        "Thermobar": ["*.csv", "*.pkl"],
    },

    install_requires=[
            'pandas',
            'numpy<2',
            'python-ternary',
            'matplotlib',
            'scikit-learn>=1.3',
            'scipy',
            'tqdm',
            'statsmodels',
            'openpyxl',
            'pathlib'
            ],
            # PyPI doesnt allow this :(
    # extras_require={
    #     'onnx': ["Thermobar_onnx @ https://github.com/PennyWieser/Thermobar_onnx/archive/refs/tags/0.0.1dev.zip"]
    # },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
