from setuptools import setup, find_packages

setup(
    name="muraw",  # Package name follows PEP 8 lowercase with no underscores
    version="0.1.0",
    packages=find_packages(where="."),
    package_dir={"": "."},
    py_modules=['dng'],
    install_requires=[
        'numpy>=1.20.0',
        'opencv-python-headless>=4.5.0',
        'tifffile>=2021.0.0',
        'astropy>=5.0.0',
    ],
    author="mu-files",
    author_email="info@mu-files.com",
    description="DNG and raw image processing utilities",
    python_requires='>=3.8',
    zip_safe=False,
)
