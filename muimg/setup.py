from setuptools import setup, find_packages

setup(
    name="muimg",
    version="0.1.0",
    description="DNG and raw image processing utilities",
    author="mu-files",
    author_email="mu-files@users.noreply.github.com",
    url="https://github.com/mu-files/mu-image",
    packages=find_packages(),
    python_requires=">=3.9",
    entry_points={
        'console_scripts': [
            'muimg=muimg.cli:main',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
