"""FITS header to AVM (Astronomy Visualization Metadata) XMP mapping.

Maps standard FITS keywords to the AVM namespace for embedding in DNG files.
AVM standard: http://www.communicatingastronomy.org/avm/1.0/

Only includes fields that are present in the FITS header — no zeros for
missing WCS data.
"""

from __future__ import annotations

from typing import Any


def _parse_ra_dec(ra_str: str, dec_str: str) -> tuple[float, float] | None:
    """Parse RA/DEC strings to decimal degrees.

    Handles both decimal and sexagesimal (HH:MM:SS / DD:MM:SS) formats.
    RA is in hours (0-24) and converted to degrees (0-360).
    """
    try:
        ra_deg = float(ra_str)
        dec_deg = float(dec_str)
        return ra_deg, dec_deg
    except (ValueError, TypeError):
        pass

    # Try sexagesimal: "HH:MM:SS.s" or "HH MM SS.s"
    try:
        parts = ra_str.replace(":", " ").split()
        h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
        ra_deg = (h + m / 60.0 + s / 3600.0) * 15.0  # hours to degrees

        parts = dec_str.replace(":", " ").split()
        sign = -1 if parts[0].startswith("-") else 1
        d, m, s = abs(float(parts[0])), float(parts[1]), float(parts[2])
        dec_deg = sign * (d + m / 60.0 + s / 3600.0)

        return ra_deg, dec_deg
    except (ValueError, TypeError, IndexError):
        return None


def fits_header_to_avm_xmp(header) -> dict[str, Any]:
    """Extract AVM XMP properties from a FITS header.

    Args:
        header: astropy FITS header object

    Returns:
        Dict with 'avm:'-prefixed keys suitable for XmpMetadata.from_attributes().
        Only includes fields actually present in the header.
    """
    attrs: dict[str, Any] = {}

    # Object / subject name
    obj = header.get("OBJECT")
    if obj and str(obj).strip():
        attrs["avm:Subject.Name"] = str(obj).strip()

    # Instrument (camera)
    instrume = header.get("INSTRUME")
    if instrume and str(instrume).strip():
        attrs["avm:Instrument"] = str(instrume).strip()

    # Telescope / facility
    telescop = header.get("TELESCOP")
    if telescop and str(telescop).strip():
        attrs["avm:Facility"] = str(telescop).strip()

    # Coordinate frame from equinox
    equinox = header.get("EQUINOX")
    if equinox is not None:
        eq_val = float(equinox)
        if eq_val == 2000.0:
            attrs["avm:Spatial.CoordinateFrame"] = "ICRS"
        elif eq_val == 1950.0:
            attrs["avm:Spatial.CoordinateFrame"] = "FK4"
        else:
            attrs["avm:Spatial.CoordinateFrame"] = "FK5"

    # WCS reference value (RA, Dec in degrees)
    # Try CRVAL first, fall back to RA/DEC keywords
    crval1 = header.get("CRVAL1")
    crval2 = header.get("CRVAL2")
    if crval1 is not None and crval2 is not None:
        attrs["avm:Spatial.ReferenceValue"] = [str(float(crval1)), str(float(crval2))]
    else:
        ra = header.get("RA")
        dec = header.get("DEC")
        if ra is not None and dec is not None:
            coords = _parse_ra_dec(str(ra), str(dec))
            if coords:
                attrs["avm:Spatial.ReferenceValue"] = [str(coords[0]), str(coords[1])]

    # WCS reference pixel
    crpix1 = header.get("CRPIX1")
    crpix2 = header.get("CRPIX2")
    if crpix1 is not None and crpix2 is not None:
        attrs["avm:Spatial.ReferencePixel"] = [str(float(crpix1)), str(float(crpix2))]

    # Image dimensions
    naxis1 = header.get("NAXIS1")
    naxis2 = header.get("NAXIS2")
    if naxis1 is not None and naxis2 is not None:
        attrs["avm:Spatial.ReferenceDimension"] = [str(int(naxis1)), str(int(naxis2))]

    # Plate scale (deg/pixel)
    # Try CDELT first, then compute from FOCALLEN + pixel size
    cdelt1 = header.get("CDELT1")
    cdelt2 = header.get("CDELT2")
    if cdelt1 is not None and cdelt2 is not None:
        attrs["avm:Spatial.Scale"] = [str(float(cdelt1)), str(float(cdelt2))]
    else:
        focallen = header.get("FOCALLEN")
        xpixsz = header.get("XPIXSZ")
        ypixsz = header.get("YPIXSZ")
        if focallen and xpixsz and ypixsz:
            # plate scale = pixel_size_um / focal_length_mm * 206.265 (arcsec/pixel)
            # convert to deg/pixel for AVM
            fl_mm = float(focallen)
            if fl_mm > 0:
                scale_x = float(xpixsz) / fl_mm * 206.265 / 3600.0
                scale_y = float(ypixsz) / fl_mm * 206.265 / 3600.0
                attrs["avm:Spatial.Scale"] = [str(-scale_x), str(scale_y)]

    # Rotation
    crota2 = header.get("CROTA2")
    if crota2 is not None:
        attrs["avm:Spatial.Rotation"] = str(float(crota2))

    # CD matrix (alternative to CDELT+CROTA)
    cd1_1 = header.get("CD1_1")
    cd1_2 = header.get("CD1_2")
    cd2_1 = header.get("CD2_1")
    cd2_2 = header.get("CD2_2")
    if all(v is not None for v in (cd1_1, cd1_2, cd2_1, cd2_2)):
        attrs["avm:Spatial.CDMatrix"] = [
            str(float(cd1_1)), str(float(cd1_2)),
            str(float(cd2_1)), str(float(cd2_2)),
        ]

    # Projection (extract from CTYPE, e.g. "RA---TAN" → "TAN")
    ctype1 = header.get("CTYPE1")
    if ctype1 is not None:
        ctype_str = str(ctype1).strip()
        # Standard WCS convention: last 3 chars after "---"
        if "---" in ctype_str:
            proj = ctype_str.split("---")[-1]
            if proj:
                attrs["avm:Spatial.CoordsystemProjection"] = proj

    # Spectral
    filt = header.get("FILTER")
    if filt and str(filt).strip():
        attrs["avm:Spectral.Bandpass"] = str(filt).strip()

    # Temporal
    date_obs = header.get("DATE-OBS")
    if date_obs is not None:
        attrs["avm:TemporalStartTime"] = str(date_obs).strip()

    exptime = header.get("EXPTIME")
    if exptime is not None:
        attrs["avm:TemporalIntegrationTime"] = str(float(exptime))

    # Creator (try multiple keywords)
    observer = (header.get("OBSERVER") or header.get("AUTHOR")
                or header.get("CREATOR"))
    if observer and str(observer).strip():
        attrs["avm:Creator"] = str(observer).strip()

    # Type
    attrs["avm:Type"] = "Observation"
    attrs["avm:MetadataVersion"] = "1.1"

    return attrs
