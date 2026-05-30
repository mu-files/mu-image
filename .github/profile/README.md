## About

Welcome to my code repository. I'm an individual developer and `mu-files` is where I publish the custom software infrastructure, libraries, and desktop utilities I build to handle image-processing tasks.

Everything here is **source-available** and completely **free for personal, academic, non-profit, government, and small business use**.

---

## Organization of the code

My software focuses heavily on **RAW image infrastructure, time-lapse compilation, and astrophotography data pipelines**.

| Repository / Module | Target Audience | Primary Focus |
| :--- | :--- | :--- |
| [**`muimg`**](https://github.com/mu-files/mu-image/tree/main/muimg) | Developers & Engineers | A Python library and CLI tool providing a comprehensive API for native Adobe DNG manipulation, custom RAW rendering pipelines, and parallel batch processing. |
| [**`mu-dng-converter`**](https://github.com/mu-files/mu-image/tree/main/mu-dng-converter) | Astrophotographers & Editors | A lightweight, cross-platform desktop application built on Flet. Compiles raw DNG sequences into MP4 videos or tif seqeunces (ideal for timelapse footage) and converts astronomical FITS files to DNGs. Both source code and built installer packages are available. |
| [**`mu-rasppi`**](https://github.com/mu-files/mu-rasppi) | Raspberry Pi camera users | Example code and benchmarks for Raspberry Pi setups. Includes scripts to capture raw frames from ZWO ASI or Raspberry Pi HQ Camera cameras directly to DNG, and performance benchmarks against PiDNG. |

---

## Licensing & Usage Terms

The repositories are governed by specific source-available terms:

- **Personal, academic & public service**: 100% free. Any individual, academic institution, non-profit organization, or government entity performing non-commercial work is fully permitted to use this software.
- **Small businesses**: Free for entities (including affiliates) with fewer than 100 total employees/contractors and less than $10,000,000 USD in total annual revenue (via a customised PolyForm Small Business License 1.0.0).
- **Commercial scale**: Entities exceeding the small business threshold require explicit commercial licensing.
- **AI/ML training restriction**: The core implementation source code is prohibited from being ingested into AI/ML training datasets. See `llms.txt` and `robots.txt` for specific allowed paths such as documentation and examples.

---

## Feedback & Contributions

If this looks useful to you, your feedback, bug reports, and feature requests are incredibly valuable. Please feel free to open an issue!
