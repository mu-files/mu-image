## About

Welcome to my code repository. I'm an individual developer and `mu-files` is where I publish the custom software infrastructure, libraries, and desktop utilities I build to handle image-processing tasks.

Public packages here are released under the **BSD 3-Clause License**.

---

## Organization of the code

My software focuses heavily on **RAW image infrastructure, time-lapse compilation, and astrophotography data pipelines**.

| Repository / Module | Target Audience | Primary Focus |
| :--- | :--- | :--- |
| [**`muimg`**](https://github.com/mu-files/mu-image/tree/main/muimg) | Developers & Engineers | A Python library and CLI tool providing a comprehensive API for native Adobe DNG manipulation, custom RAW rendering pipelines, and parallel batch processing. |
| [**`mu-dng-converter`**](https://github.com/mu-files/mu-image/tree/main/mu-dng-converter) | Astrophotographers & Editors | A lightweight, cross-platform desktop application. Transcodes and processes DNGs (compression, metadata operations), renders DNG sequences into MP4 videos or image sequences (ideal for timelapse footage), and converts astronomical FITS files to DNGs. Both source code and built installer packages are available. |
| [**`mu-rasppi`**](https://github.com/mu-files/mu-rasppi) | Raspberry Pi camera users | Example code and benchmarks for Raspberry Pi setups. Includes scripts to capture raw frames from ZWO ASI or Raspberry Pi HQ Camera cameras directly to DNG, and performance benchmarks against PiDNG. |

---

## Licensing & Usage Terms

- **License**: [BSD 3-Clause](https://opensource.org/licenses/BSD-3-Clause). See each repository's `LICENSE` file for the full text.
- **AI/crawler preferences**: The intent of [`llms.txt`](https://github.com/mu-files/mu-image/blob/main/llms.txt) and [`robots.txt`](https://github.com/mu-files/mu-image/blob/main/robots.txt) is that the core `muimg` package (`muimg/muimg/` and `muimg/c-src/`) **NOT** be used for LLM or ML training. Documentation, tests, and examples may be used for learning the API.

---

## Feedback & Contributions

If this looks useful to you, your feedback, bug reports, and feature requests are incredibly valuable. Please feel free to open an issue!
