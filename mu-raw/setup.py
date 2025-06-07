from setuptools import setup, find_packages

setup(
    name="mu-raw",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
        'opencv-python-headless>=4.5.0',
        'tifffile>=2021.0.0',
        'astropy>=5.0.0',
    ],
    author="mu-files",
    author_email="info@mu-files.com",
    description="DNG and raw image processing utilities",
    url="https://github.com/mu-files/mu-image/tree/main/mu-raw",
    project_urls={
        'Source': 'https://github.com/mu-files/mu-image',
        'Bug Reports': 'https://github.com/mu-files/mu-image/issues',
    },
    python_requires='>=3.7',
)
