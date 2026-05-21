# Released under a modified PolyForm Small Business License.
# Free for small businesses, individuals, and academics. See LICENSE for details.

"""DNG compression utilities for tiled and stripped image data."""

import numpy as np

from .deps import imagecodecs_proxy as imagecodecs, tifffile_proxy as tifffile


def swizzle_cfa_data(
    raw_data: np.ndarray,
    output_region: tuple[int, int, int, int] | None = None
) -> np.ndarray:
    """Swizzle 2x2 CFA data into a 2x2 grid of channel sub-images.
    
    Args:
        raw_data: Input CFA data in 2x2 Bayer pattern (H, W)
        output_region: Optional (y_start, y_end, x_start, x_end) in swizzled
            coordinate space. If specified, only returns that region of the
            swizzled output.
    
    Returns:
        Swizzled data array. If output_region is None, returns full (H, W) array.
        If output_region is specified, returns only the requested region.
    """
    h, w = raw_data.shape
    h_half, w_half = h // 2, w // 2
    
    # Default to full image if no region specified
    if output_region is None:
        output_region = (0, h, 0, w)
    
    y_start, y_end, x_start, x_end = output_region
    out_h = y_end - y_start
    out_w = x_end - x_start
    
    swizzled_data = np.empty((out_h, out_w), dtype=raw_data.dtype)
    
    # R pixels: top-left quadrant [0:h_half, 0:w_half]
    if y_start < h_half and x_start < w_half:
        r_y_start = max(0, y_start)
        r_y_end = min(h_half, y_end)
        r_x_start = max(0, x_start)
        r_x_end = min(w_half, x_end)
        swizzled_data[r_y_start-y_start:r_y_end-y_start, 
                      r_x_start-x_start:r_x_end-x_start] = \
            raw_data[r_y_start*2:r_y_end*2:2, r_x_start*2:r_x_end*2:2]
    
    # G1 pixels: top-right quadrant [0:h_half, w_half:w]
    if y_start < h_half and x_end > w_half:
        g1_y_start = max(0, y_start)
        g1_y_end = min(h_half, y_end)
        g1_x_start = max(w_half, x_start)
        g1_x_end = min(w, x_end)
        swizzled_data[g1_y_start-y_start:g1_y_end-y_start,
                      g1_x_start-x_start:g1_x_end-x_start] = \
            raw_data[g1_y_start*2:g1_y_end*2:2, 
                     (g1_x_start-w_half)*2+1:(g1_x_end-w_half)*2+1:2]
    
    # G2 pixels: bottom-left quadrant [h_half:h, 0:w_half]
    if y_end > h_half and x_start < w_half:
        g2_y_start = max(h_half, y_start)
        g2_y_end = min(h, y_end)
        g2_x_start = max(0, x_start)
        g2_x_end = min(w_half, x_end)
        swizzled_data[g2_y_start-y_start:g2_y_end-y_start,
                      g2_x_start-x_start:g2_x_end-x_start] = \
            raw_data[(g2_y_start-h_half)*2+1:(g2_y_end-h_half)*2+1:2,
                     g2_x_start*2:g2_x_end*2:2]
    
    # B pixels: bottom-right quadrant [h_half:h, w_half:w]
    if y_end > h_half and x_end > w_half:
        b_y_start = max(h_half, y_start)
        b_y_end = min(h, y_end)
        b_x_start = max(w_half, x_start)
        b_x_end = min(w, x_end)
        swizzled_data[b_y_start-y_start:b_y_end-y_start,
                      b_x_start-x_start:b_x_end-x_start] = \
            raw_data[(b_y_start-h_half)*2+1:(b_y_end-h_half)*2+1:2,
                     (b_x_start-w_half)*2+1:(b_x_end-w_half)*2+1:2]
    
    return swizzled_data


def deswizzle_cfa_data(swizzled_data: np.ndarray) -> np.ndarray:
    """Deswizzle CFA data from a 2x2 grid of R, G1, G2, B sub-images back to RGGB."""
    h_swizzled, w_swizzled = swizzled_data.shape
    if h_swizzled % 2 != 0 or w_swizzled % 2 != 0:
        raise ValueError("Swizzled data dimensions must be even.")

    # Calculate half dimensions for quadrant extraction
    h_half, w_half = h_swizzled // 2, w_swizzled // 2

    # Extract the four channels from the swizzled data
    # R is top-left quadrant
    r_channel = swizzled_data[0:h_half, 0:w_half]
    # G1 (first green) is top-right quadrant
    g1_channel = swizzled_data[0:h_half, w_half:w_swizzled]
    # G2 (second green) is bottom-left quadrant
    g2_channel = swizzled_data[h_half:h_swizzled, 0:w_half]
    # B is bottom-right quadrant
    b_channel = swizzled_data[h_half:h_swizzled, w_half:w_swizzled]

    # Create an empty array for the original interleaved data
    # Its dimensions will be the same as the swizzled_data because each sub-image
    # was H/2 x W/2, and they are re-interleaved into a H x W image.
    original_data = np.empty_like(swizzled_data)

    # Place the channels back into the original RGGB pattern
    original_data[0::2, 0::2] = r_channel  # R pixels
    original_data[0::2, 1::2] = g1_channel  # G1 pixels (top-right G)
    original_data[1::2, 0::2] = g2_channel  # G2 pixels (bottom-left G)
    original_data[1::2, 1::2] = b_channel  # B pixels

    return original_data


def compress_ifd(
    data: np.ndarray,
    compression: tifffile.COMPRESSION,
    compression_args: dict | None,
    bits_per_sample: int,
    photometric: str,
    target_byteorder: str,
    tile_size: tuple[int, int] | None = None,
    rows_per_strip: int | None = None,
    num_compression_workers: int = 1
) -> list[bytes]:
    """Segment image data and compress each segment in a single pass.
    
    Args:
        data: Image data array (H, W) or (H, W, C)
        compression: Compression type
        compression_args: Compression-specific arguments
        bits_per_sample: Bits per sample
        photometric: Photometric interpretation
        target_byteorder: Target byte order ('>' or '<')
        tile_size: Tile dimensions (height, width) for tiled layout
        rows_per_strip: Rows per strip for stripped layout
        num_compression_workers: Number of parallel compression workers (default: 1)
        
    Returns:
        List of encoded segments (bytes) in raster-scan order
    """
    from .tiff_metadata import normalize_array_to_target_byteorder
    
    # Track whether we need to swizzle CFA data per-tile
    needs_swizzle = (
        photometric == "CFA" and 
        compression in (tifffile.COMPRESSION.JPEGXL, tifffile.COMPRESSION.JPEGXL_DNG)
    )
    
    h, w = data.shape[:2]
    
    # Use sequential processing if num_workers=1 or single segment
    use_pipeline = (
        num_compression_workers > 1 and
        (tile_size is not None or rows_per_strip is not None)
    )
    
    # Helper: Extract and prepare tile data (with padding for tiles only)
    def extract_tile(ty: int, tx: int, tile_h: int, tile_w: int) -> np.ndarray:
        """Extract and prepare a tile segment with padding."""
        # Calculate actual bounds (don't go beyond image dimensions)
        y_end = min(ty + tile_h, h)
        x_end = min(tx + tile_w, w)
        
        # Apply per-tile swizzling if needed
        if needs_swizzle:
            tile = swizzle_cfa_data(data, (ty, y_end, tx, x_end))
        else:
            tile = data[ty:y_end, tx:x_end]
        
        # TIFF spec requires padding edge tiles to nominal tile size
        actual_h, actual_w = tile.shape[:2]
        if actual_h < tile_h or actual_w < tile_w:
            # Pad edge tiles with zeros
            if len(tile.shape) == 3:
                padded = np.zeros((tile_h, tile_w, tile.shape[2]), dtype=tile.dtype)
                padded[:actual_h, :actual_w, :] = tile
            else:
                padded = np.zeros((tile_h, tile_w), dtype=tile.dtype)
                padded[:actual_h, :actual_w] = tile
            tile = padded
        
        return tile
    
    # Helper: Extract strip data (no padding for strips)
    def extract_strip(y: int, strip_h: int) -> np.ndarray:
        """Extract a strip segment without padding."""
        # Calculate actual bounds
        y_end = min(y + strip_h, h)
        
        # Apply swizzling if needed
        if needs_swizzle:
            strip = swizzle_cfa_data(data, (y, y_end, 0, w))
        else:
            strip = data[y:y_end]
        
        return strip
    
    # Define compression function for a single segment
    def compress_segment(segment: np.ndarray) -> bytes:
        """Compress a single segment based on compression type."""
        if compression in (tifffile.COMPRESSION.JPEGXL, tifffile.COMPRESSION.JPEGXL_DNG):
            # JXL compression
            jxl_distance = compression_args.get('distance', 0.0) if compression_args else 0.0
            jxl_effort = compression_args.get('effort', 5) if compression_args else 5
            
            if jxl_distance == 0.0:
                return imagecodecs.jpegxl_encode(
                    segment, lossless=True, effort=jxl_effort, bitspersample=bits_per_sample
                )
            else:
                return imagecodecs.jpegxl_encode(
                    segment, distance=jxl_distance, effort=jxl_effort, bitspersample=bits_per_sample
                )
                
        elif compression == tifffile.COMPRESSION.JPEG:
            # JPEG compression
            lossless = compression_args.get('lossless', True) if compression_args else True
            seg_data = segment
            
            if photometric == "CFA":
                # Reshape CFA data to 2-component interleaved format
                h, w = seg_data.shape
                if w % 2 != 0:
                    raise ValueError(f"CFA width must be even for JPEG compression, got {w}")
                seg_data = seg_data.reshape(h, w // 2, 2)
                
                return imagecodecs.jpeg_encode(
                    seg_data,
                    lossless=lossless,
                    bitspersample=bits_per_sample
                )
            else:
                return imagecodecs.jpeg_encode(
                    seg_data,
                    lossless=lossless,
                    bitspersample=bits_per_sample,
                    colorspace='RGB',
                    outcolorspace='RGB'
                )
                
        elif compression == tifffile.COMPRESSION.NONE:
            # Uncompressed - handle byte order and bit packing
            # 1. Convert to target byte order
            seg = normalize_array_to_target_byteorder(segment, target_byteorder)
            
            # 2. Pack bits if needed (9-15 bit data in uint16 container)
            if 9 <= bits_per_sample <= 15:
                # Use imagecodecs.packints_encode like tifffile does
                runlen = seg.shape[-1]
                if len(seg.shape) > 1:
                    runlen *= seg.shape[-2]
                return imagecodecs.packints_encode(seg, bits_per_sample, runlen=runlen)
            else:
                # No packing needed - convert to bytes
                return seg.tobytes()
        else:
            # Other compression types (shouldn't happen for DNG files)
            return segment.tobytes()
    
    if not use_pipeline:
        # Sequential processing (original behavior)
        encoded_segments = []
        
        if tile_size is not None:
            # Tiled layout - split into 2D grid and compress each tile
            tile_h, tile_w = tile_size
            for ty in range(0, h, tile_h):
                for tx in range(0, w, tile_w):
                    tile = extract_tile(ty, tx, tile_h, tile_w)
                    encoded_segments.append(compress_segment(tile))
        elif rows_per_strip is not None:
            # Stripped layout - split into horizontal strips and compress each
            for y in range(0, h, rows_per_strip):
                strip = extract_strip(y, rows_per_strip)
                encoded_segments.append(compress_segment(strip))
        else:
            # Single segment (full image)
            if needs_swizzle:
                segment = swizzle_cfa_data(data)
            else:
                segment = data
            encoded_segments.append(compress_segment(segment))
        
        return encoded_segments
    
    else:
        # Parallel processing using ProcessingPipeline
        from .processing import ProcessingPipeline
        
        # Producer: Generate tile/strip metadata
        def tile_producer():
            tile_id = 0
            if tile_size is not None:
                tile_h, tile_w = tile_size
                for ty in range(0, h, tile_h):
                    for tx in range(0, w, tile_w):
                        yield (tile_id, 'tile', ty, tx, tile_h, tile_w)
                        tile_id += 1
            elif rows_per_strip is not None:
                for y in range(0, h, rows_per_strip):
                    yield (tile_id, 'strip', y, 0, rows_per_strip, 0)
                    tile_id += 1
        
        # Consumer: Extract and compress tile/strip
        def tile_consumer(task):
            tile_id, seg_type, ty, tx, seg_h, seg_w = task
            if seg_type == 'tile':
                segment = extract_tile(ty, tx, seg_h, seg_w)
            else:  # strip
                segment = extract_strip(ty, seg_h)
            compressed = compress_segment(segment)
            return (tile_id, compressed)
        
        # Writer: Buffer and assemble tiles in order
        tile_buffer = {}
        next_expected_id = 0
        result_segments = []
        
        def tile_writer(result):
            nonlocal next_expected_id
            if result is None:
                return
            
            tile_id, compressed = result
            tile_buffer[tile_id] = compressed
            
            # Write all consecutive tiles starting from next expected
            while next_expected_id in tile_buffer:
                result_segments.append(tile_buffer.pop(next_expected_id))
                next_expected_id += 1
        
        # Run pipeline
        pipeline = ProcessingPipeline(
            producer=tile_producer,
            consumer=tile_consumer,
            writer=tile_writer,
            num_workers=num_compression_workers,
            task_name="Tile Compression"
        )
        pipeline.run()
        
        return result_segments
