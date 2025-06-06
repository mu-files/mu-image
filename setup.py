from setuptools import setup, find_packages

setup(
    name="raw",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
        'opencv-python-headless>=4.5.0',
        'tifffile>=2021.0.0',
        'astropy>=5.0.0',
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="DNG and raw image processing utilities",
    python_requires='>=3.7',
)
