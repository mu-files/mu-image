# License Information

## Main Package License

The muimg package is released under a modified PolyForm Small Business License.
See the main [LICENSE](../LICENSE) file for details.

## Third-Party Bundled Components

### Adobe DNG SDK (`c-src/raw_render/`)

- **License**: Adobe DNG SDK License (permissive, royalty-free)
- **Copyright**: 2006-2024 Adobe Systems Incorporated
- **Source**: https://www.adobe.com/support/downloads/dng/dng_sdk.html
- **Usage**: Pixel-level color processing operations extracted from DNG SDK 1.7.1
- **Modifications**: Python/NumPy bindings, standalone implementation

The Adobe DNG SDK License grants worldwide, royalty-free rights to use, modify,
distribute, and sublicense for any purpose, including commercial use. Copyright
notices must be retained in distributed code.

See [LICENSE-ADOBE-DNG-SDK](LICENSE-ADOBE-DNG-SDK) for full license text.

### VNG Demosaicing Algorithm (`c-src/demosaic/vng.c`)

- **License**: LGPL v2.1 / CDDL v1.0 (dual license)
- **Original Authors**: Dave Coffin, LibRaw LLC
- **Source**: https://github.com/LibRaw/LibRaw
- **Modifications**: Python/NumPy bindings and thread-safety fixes

See [LICENSE-LGPL-2.1](LICENSE-LGPL-2.1) for full license text.

## Optional Components (Not Distributed)

### RCD Demosaicing Algorithm (`c-src/demosaic/rcd.txt`)

- **License**: GNU General Public License v3.0 or later
- **Original Author**: Luis Sanz Rodríguez
- **Source**: https://github.com/LuisSR/RCD-Demosaicing
- **Status**: Available as source code (rcd.txt) but NOT built or distributed by default

The RCD algorithm is provided as optional source code. To use it, users must:
1. Manually rename `rcd.txt` to `rcd.c`
2. Rebuild the package
3. Accept the GPL v3 license terms

## License Compatibility

**Adobe DNG SDK** and **VNG** are permissive licenses (Adobe DNG SDK License and LGPL)
that allow redistribution in proprietary software. They are included and distributed
with muimg.

**RCD** (GPL v3) is optional and not distributed by default. Users who enable it must
comply with GPL v3 terms.
