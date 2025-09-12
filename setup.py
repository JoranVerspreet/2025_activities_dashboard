from setuptools import setup, find_packages

setup(
    name='supermarket_package',
    version='0.1.0',
    packages=find_packages(where='notebooks'),
    package_dir={'': 'notebooks'},
    install_requires=[],
    author='group_project',
    description='Supermarket case study package',
    python_requires='>=3.7',
)
