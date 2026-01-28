"""Video I/O utilities for encoding image sequences to video."""

import io
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Union

import av
import numpy as np

from .imgio import decode_image
from .processing import ProcessingPipeline

logger = logging.getLogger(__name__)


class SequenceEncodePipeline(ProcessingPipeline):
    """Pipeline for encoding a sequence of images to a video file.
    
    Extends ProcessingPipeline with video-specific functionality:
    - Default producer reads image files from a list of paths
    - Default consumer decodes images using decode_image
    - Default writer buffers frames and encodes them in order using PyAV
    
    Each of producer, consumer, writer can be overridden by passing custom
    callables to __init__ or by subclassing and overriding the default_* methods.
    
    Example usage:
        pipeline = SequenceEncodePipeline(
            source_files=["/path/to/img1.dng", "/path/to/img2.dng", ...],
            output_path="/path/to/output.mp4",
            resolution=(1920, 1080),
            config={"codec": "hevc", "crf": 20, "bit_depth": 8, "frame_rate": 30},
        )
        pipeline.run()
    """
    
    def __init__(
        self,
        source_files: list[Union[str, Path]] = None,
        output_path: Union[str, Path] = None,
        resolution: tuple[int, int] = None,
        config: dict[str, Any] = None,
        use_temp_file: bool = True,
        producer: Callable[[], Iterable[Any]] = None,
        consumer: Callable[[Any], Any] = None,
        writer: Callable[[Any], None] = None,
        num_workers: int = 4,
        queue_size: int = None,
        task_name: str = "Video Encoding",
    ):
        """Initialize the video encoding pipeline.
        
        Args:
            source_files: List of image file paths to encode. Required if using
                default producer.
            output_path: Path to output video file. Required if using default writer.
            resolution: Output video resolution as (width, height). If None and using
                default consumer/writer, will be determined from first decoded image.
            config: Video encoding configuration with keys:
                - codec: Video codec (default: 'hevc', options: 'hevc', 'h264', 'vp9')
                - crf: Constant Rate Factor for quality (default: 20, lower=better)
                - bit_depth: Bit depth (default: 8, options: 8, 10)
                - frame_rate: Output frame rate in fps (default: 30)
            use_temp_file: If True (default), encode to a local temp file first,
                then copy to output_path. Helps avoid issues with network drives.
            producer: Custom producer callable. If None, uses default_producer.
            consumer: Custom consumer callable. If None, uses default_consumer.
            writer: Custom writer callable. If None, uses default_writer.
            num_workers: Number of parallel consumer threads.
            queue_size: Maximum size of processing queues.
            task_name: Descriptive name for logging.
        """
        self.source_files = [Path(f) for f in source_files] if source_files else []
        self.output_path = Path(output_path) if output_path else None
        self.resolution = resolution
        self.use_temp_file = use_temp_file
        
        # Temp file path (set during run if use_temp_file is True)
        self._temp_path: Path | None = None
        
        # Container metadata (set via set_metadata(), applied before container close)
        self._container_metadata: dict[str, str] = {}
        
        # Parse config with defaults
        config = config or {}
        self.codec = config.get("codec", "hevc")
        self.crf = config.get("crf", 20)
        self.bit_depth = config.get("bit_depth", 8)
        self.frame_rate = config.get("frame_rate", 30)
        
        # Validate bit depth
        if self.bit_depth not in (8, 10):
            raise ValueError(f"bit_depth must be 8 or 10, got {self.bit_depth}")
        
        # Determine output dtype and pixel format based on bit depth
        if self.bit_depth == 10:
            self.output_dtype = np.uint16
            self.pix_fmt = "yuv420p10le"
            self.input_format = "rgb48le"
        else:
            self.output_dtype = np.uint8
            self.pix_fmt = "yuv420p"
            self.input_format = "rgb24"
        
        # PyAV container and stream (initialized in _setup_encoder)
        self._container = None
        self._stream = None
        
        # Buffered writer state
        self._next_expected_index = 0
        self._frame_buffer = {}
        self._failed_frames = set()
        
        # Use provided callables or default methods
        actual_producer = producer if producer is not None else self.default_producer
        actual_consumer = consumer if consumer is not None else self.default_consumer
        actual_writer = writer if writer is not None else self.default_writer
        
        super().__init__(
            producer=actual_producer,
            consumer=actual_consumer,
            writer=actual_writer,
            num_workers=num_workers,
            queue_size=queue_size,
            task_name=task_name,
        )
    
    def _setup_encoder(self):
        """Initialize PyAV container and video stream."""
        if self.output_path is None:
            raise ValueError("output_path is required for video encoding")
        if self.resolution is None:
            raise ValueError("resolution must be set before encoding")
        
        width, height = self.resolution
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine encoding path (temp file or direct)
        if self.use_temp_file:
            temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4', prefix='video_encode_')
            os.close(temp_fd)
            self._temp_path = Path(temp_path)
            encode_path = self._temp_path
            logger.info(f"Encoding to temporary file: {self._temp_path}")
        else:
            encode_path = self.output_path
        
        self._container = av.open(str(encode_path), mode="w")
        self._stream = self._container.add_stream(self.codec, rate=self.frame_rate)
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = self.pix_fmt
        
        options = {"crf": str(self.crf)}
        if self.bit_depth == 10:
            options["profile"] = "main10"
        self._stream.options = options
        
        logger.info(
            f"Video encoder initialized: {width}x{height} @ {self.frame_rate}fps "
            f"(codec={self.codec}, crf={self.crf}, {self.bit_depth}-bit)"
        )
    
    def _finalize_encoder(self):
        """Flush and close the PyAV container."""
        if self._container is not None:
            # Flush encoder
            for packet in self._stream.encode():
                self._container.mux(packet)
            
            # Apply container metadata before closing
            for key, value in self._container_metadata.items():
                self._container.metadata[key] = value
                logger.debug(f"Set container metadata {key}={value}")
            
            self._container.close()
            self._container = None
        
        # Copy from temp file to final destination if needed
        if self._temp_path is not None:
            try:
                logger.info(f"Copying video to final destination: {self.output_path}")
                shutil.copy2(self._temp_path, self.output_path)
                self._temp_path.unlink()
                logger.info(f"Cleaned up temporary file")
            except Exception as e:
                logger.error(f"Failed to copy video to final destination: {e}")
                logger.info(f"Video remains at temporary location: {self._temp_path}")
                raise
            finally:
                self._temp_path = None
        
        logger.info(f"Video saved to {self.output_path}")
    
    @property
    def failed_frames(self) -> set[int]:
        """Set of frame indices that failed to decode."""
        return self._failed_frames
    
    def set_metadata(self, key: str, value: str) -> None:
        """Set a metadata key-value pair on the output video container.
        
        Args:
            key: Metadata key (e.g., 'creation_time')
            value: Metadata value
        """
        self._container_metadata[key] = value
    
    def default_producer(self) -> Iterable[tuple[int, str, bytes]]:
        """Default producer: reads image files and yields (index, path, blob).
        
        Yields:
            Tuples of (index, file_path_str, file_bytes)
        """
        for index, file_path in enumerate(self.source_files):
            try:
                with open(file_path, "rb") as f:
                    blob = f.read()
                yield (index, str(file_path), blob)
            except OSError as e:
                logger.warning(f"Skipping file {file_path} due to I/O error: {e}")
                continue
    
    def default_consumer(self, task: tuple[int, str, bytes]) -> tuple[int, np.ndarray | None]:
        """Default consumer: decodes image blob using decode_image.
        
        Args:
            task: Tuple of (index, file_path, blob)
            
        Returns:
            Tuple of (index, decoded_image) or (index, None) on failure
        """
        index, file_path, blob = task
        try:
            img = decode_image(io.BytesIO(blob), output_dtype=self.output_dtype)
            
            if img is None:
                logger.warning(f"Frame {index}: Failed to decode {Path(file_path).name}")
                self._failed_frames.add(index)
                return (index, None)
            
            # Resize if resolution is set and image doesn't match
            if self.resolution is not None:
                current_height, current_width = img.shape[:2]
                if (current_width, current_height) != self.resolution:
                    import cv2
                    img = cv2.resize(img, self.resolution, interpolation=cv2.INTER_LINEAR)
            
            return (index, img)
        except Exception as e:
            logger.warning(f"Frame {index}: Error processing {Path(file_path).name}: {e}")
            self._failed_frames.add(index)
            return (index, None)
    
    def default_writer(self, result: tuple[int, np.ndarray | None]) -> None:
        """Default writer: buffers frames and encodes them in order.
        
        Args:
            result: Tuple of (index, image) from consumer
        """
        if result is None:
            return
        
        index, img = result
        
        # Add to buffer
        self._frame_buffer[index] = img
        
        # Encode all consecutive frames starting from next expected index
        while self._next_expected_index in self._frame_buffer:
            current_index = self._next_expected_index
            img = self._frame_buffer.pop(current_index)
            
            # Skip failed frames
            if img is None:
                self._next_expected_index += 1
                continue
            
            # Encode frame
            frame = av.VideoFrame.from_ndarray(img, format=self.input_format)
            for packet in self._stream.encode(frame):
                self._container.mux(packet)
            
            self._next_expected_index += 1
    
    def run(self):
        """Run the video encoding pipeline.
        
        Sets up the encoder before processing and finalizes it after.
        """
        logger.info(
            f"Starting video encode: {len(self.source_files)} images -> {self.output_path}"
        )
        
        # Set up encoder
        self._setup_encoder()
        
        try:
            # Run the base pipeline
            super().run()
        finally:
            # Always finalize encoder
            self._finalize_encoder()
        
        # Report stats
        stats = self.get_queue_stats()
        failed_count = len(self._failed_frames)
        success_count = len(self.source_files) - failed_count
        logger.info(
            f"Encoding complete: {success_count}/{len(self.source_files)} frames "
            f"in {stats.get('processing_time', 0):.1f}s"
        )
        if failed_count > 0:
            logger.warning(f"Failed frames: {sorted(self._failed_frames)}")
