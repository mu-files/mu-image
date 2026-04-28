# Released under a modified PolyForm Small Business License.
# Free for small businesses, individuals, and academics. See LICENSE for details.

import csv
import logging
import threading

from dataclasses import asdict, fields
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, get_args, get_type_hints

logger = logging.getLogger(__name__)


class CsvWriter:
    """A generic class to handle CSV logging."""

    def __init__(
        self,
        csv_filepath: Path,
        header_data: list[tuple[str, Any]],
        data_type: type,
    ):
        self.csv_filepath = csv_filepath
        # If a single type is provided, wrap it in a list for consistent processing.
        data_types = data_type if isinstance(data_type, list) else [data_type]

        # Combine column titles from all provided data types.
        self.column_titles = []
        for dt in data_types:
            self.column_titles.extend([f.name for f in fields(dt)])

        # Open file and create writer
        self.csv_filepath.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing CSV data to: {self.csv_filepath}")
        self.file_handle = open(self.csv_filepath, "w", newline="")
        self.writer = csv.writer(self.file_handle)

        # Write header and column titles
        self._write_header(header_data)
        self._write_column_titles()
        self.file_handle.flush()

    def _format_value(self, value: Any) -> Any:
        """Formats a simple value for CSV writing, converting None to an empty string."""
        if value is None:
            return ''
        if isinstance(value, (str, int, float, bool, datetime, timedelta)):
            return value
        raise TypeError(
            f"Value must be a simple type (str, int, float, bool, datetime, timedelta), but has type {type(value)}."
        )

    def _write_header(self, header_data: list[tuple[str, Any]]):
        """Writes the header data to the CSV file, handling various data structures."""
        for key, value in header_data:
            if isinstance(value, dict):
                # Handle dictionary values
                formatted_items = [f"{k}: {self._format_value(v)}" for k, v in value.items()]
                self.writer.writerow([key] + formatted_items)

            elif isinstance(value, list):
                # Handle list of dictionaries
                if not all(isinstance(i, dict) for i in value):
                    raise TypeError(
                        f"Header list for key '{key}' must contain only dictionaries, but found other types."
                    )

                if not value:
                    self.writer.writerow([key])
                    continue

                # Write the first item on the same line as the key
                first_item = value[0]
                first_row = [key] + [f"{k}: {self._format_value(v)}" for k, v in first_item.items()]
                self.writer.writerow(first_row)

                # Write subsequent items on new lines, indented
                for item_dict in value[1:]:
                    row = [""] + [f"{k}: {self._format_value(v)}" for k, v in item_dict.items()]
                    self.writer.writerow(row)
            else:
                # Handle simple values (str, int, float, bool, datetime, timedelta)
                self.writer.writerow([key, self._format_value(value)])

    def _write_column_titles(self):
        self.writer.writerow([])  # Blank line for separation
        self.writer.writerow(self.column_titles)

    def write_row(self, row_data: dict[str, Any] | list[Any] | Any):
        """
        Writes a single row of data from a dictionary, a dataclass, or a list of dataclasses.

        - If a dictionary is provided, keys must be a subset of column_titles.
        - If a dataclass or list of dataclasses is provided, they are converted to a dictionary.
        - Missing keys will be written as empty strings.
        - Floats and datetimes are auto-formatted.
        """
        row_dict = {}
        if isinstance(row_data, dict):
            row_dict = row_data
        elif isinstance(row_data, list):
            for item in row_data:
                row_dict.update(asdict(item))
        else:  # Assume a single dataclass object
            row_dict = asdict(row_data)

        extra = set(row_dict.keys()) - set(self.column_titles)
        if extra:
            raise ValueError(f"Row data contains extra keys not in column titles: {extra}")

        # Create the row in the correct order, substituting "" for missing keys
        # and formatting values.
        ordered_row = []
        for col in self.column_titles:
            value = row_dict.get(col, "")
            if isinstance(value, float):
                ordered_row.append(f"{value:.6g}")
            elif isinstance(value, datetime):
                ordered_row.append(value.isoformat())
            else:
                ordered_row.append(value)
        self.writer.writerow(ordered_row)
        self.file_handle.flush()

    def close(self):
        """Closes the CSV file."""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class CsvOrderedWriter(CsvWriter):
    """
    A generic, thread-safe writer that ensures rows are written to a CSV file
    in sequential order, even when results arrive from parallel workers out of order.
    """

    def __init__(
        self,
        csv_filepath: Path,
        header_data: list[tuple[str, Any]],
        data_type: type,
    ):
        """Initializes the writer."""
        super().__init__(csv_filepath, header_data, data_type)
        self._lock = threading.Lock()
        self._buffer: dict[int, dict[str, Any]] = {}
        self._next_index = 0

    def _write_buffered_items(self):
        """
        Writes any consecutive items from the buffer that are now ready.
        This method MUST be called within a lock.
        """
        while self._next_index in self._buffer:
            row_data = self._buffer.pop(self._next_index)
            super().write_row(row_data)
            self._next_index += 1

    def write_row(self, index: int, row_data: dict[str, Any] | list[Any] | Any):
        """
        Writes a data row. Buffers if the index is not the next one expected.
        Dataclasses are converted to dictionaries before buffering.
        """
        row_dict = {}
        if isinstance(row_data, dict):
            row_dict = row_data
        elif isinstance(row_data, list):
            for item in row_data:
                row_dict.update(asdict(item))
        else:  # Assume a single dataclass object
            row_dict = asdict(row_data)

        with self._lock:
            if index == self._next_index:
                super().write_row(row_dict)
                self._next_index += 1
                self._write_buffered_items()
            elif index > self._next_index:
                self._buffer[index] = row_dict

    def close(self):
        """
        Flushes any remaining items in the buffer and closes the underlying writer.
        """
        with self._lock:
            self._write_buffered_items()
            if self._buffer:
                logger.warning(
                    f"Flushing remaining {len(self._buffer)} items from log buffer. Gaps in sequence may exist."
                )
                sorted_keys = sorted(self._buffer.keys())
                for key in sorted_keys:
                    row_data = self._buffer.pop(key)
                    super().write_row(row_data)
        super().close()


class CsvReader:
    """Reads a CSV file with a key-value header and yields dataclass objects for each row."""

    def __init__(
        self,
        csv_filepath: str | Path,
        header_schema: list[tuple[str, Any]],
        data_type: type | list[type],
    ):
        self.csv_filepath = Path(csv_filepath)
        self.header_schema = header_schema
        self._single_type_requested = not isinstance(data_type, list)
        self.data_types = data_type if isinstance(data_type, list) else [data_type]

        self._file: TextIO | None = None
        self._reader: csv.DictReader | None = None
        self.header: list[tuple[str, Any]] | None = None
        self.row_dict: dict[str, str] | None = None
        self.fields_to_load: set[str] | None = None

        # Open the file and prepare the reader immediately upon initialization.
        self._file = open(self.csv_filepath, "r", newline="")
        try:
            self._read_header_and_first_row()
        except Exception:
            self._file.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Closes the file handle. Required if not using a 'with' statement."""
        if self._file:
            self._file.close()
            self._file = None
            self._reader = None

    def _read_header_and_first_row(self):
        """Reads the header and the first data row from the open file."""
        raw_header = {}
        last_key = None
        schema_map = {k.lower(): v for k, v in self.header_schema}
        data_column_headers = None
        self._rows = []  # Cache for data rows to allow re-iteration

        # Create a reader to parse the file line-by-line.
        line_reader = csv.reader(self._file)

        for row in line_reader:
            if not row:
                continue

            key = row[0].strip()

            # The data section starts when the first column is not a recognized header key.
            if key and key.lower() not in schema_map:
                data_column_headers = row
                # The rest of the file is data. We create our own dicts.
                self._rows = [dict(zip(data_column_headers, data_row)) for data_row in line_reader]
                break  # Stop parsing for metadata.

            # --- Standard Header Parsing Logic ---
            # Strip empty cells (trailing commas from Excel create empty strings)
            values = [v.strip() for v in row[1:] if v.strip()]
            def parse_as_dict(items):
                d = {}
                for item in items:
                    if ':' not in item:
                        raise ValueError(f"Invalid dictionary item format for key '{key}': '{item}'. Expected 'key: value'.")
                    k, v = item.split(':', 1)
                    d[k.strip()] = v.strip()
                return d

            if key:
                last_key = key.lower()
                requested_type = schema_map.get(last_key)
                is_container_schema = isinstance(requested_type, tuple)

                if is_container_schema:
                    raw_header[last_key] = parse_as_dict(values)
                else:
                    if len(values) > 1:
                        raise ValueError(f"Header key '{key}' expects a single value but got {len(values)}.")
                    raw_header[last_key] = values[0].strip() if values else ""
            elif not key and last_key:
                current_value = raw_header.get(last_key)
                if not values:
                    continue
                new_dict = parse_as_dict(values)
                if isinstance(current_value, dict):
                    raw_header[last_key] = [current_value, new_dict]
                elif isinstance(current_value, list):
                    current_value.append(new_dict)

        # Set the first row for inspection, if it exists.
        self.row_dict = self._rows[0] if self._rows else None

        # The main reader for iteration is now just an iterator over our cached list.
        self._reader = iter(self._rows)

        self.header = self._cast_and_validate(raw_header, self.header_schema)
        # The fields to load are determined from all provided data_type dataclasses.
        self.fields_to_load = set()
        for dt in self.data_types:
            self.fields_to_load.update({f.name for f in fields(dt)})

    def _cast_and_validate(self, raw_dict: dict[str, Any], schema: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
        """Recursively validates and casts a dictionary to conform to an ordered schema, returning a list of tuples."""
        typed_list = []
        for key, requested_type in schema:
            if key.lower() not in raw_dict:
                raise KeyError(f"Required header key '{key}' not found in CSV header.")

            raw_value = raw_dict[key.lower()]
            
            try:
                if isinstance(requested_type, tuple):
                    expected_container_type, sub_schema = requested_type
                    if expected_container_type == list:
                        if not isinstance(raw_value, list):
                            if isinstance(raw_value, dict):
                                raw_value = [raw_value]
                            else:
                                raise TypeError(f"Expected a list for key '{key}', but got {type(raw_value)}.")
                        typed_value = [self._cast_and_validate_dict(item, sub_schema) for item in raw_value]
                        typed_list.append((key, typed_value))

                    elif expected_container_type == dict:
                        if not isinstance(raw_value, dict):
                            raise TypeError(f"Expected a dict for '{key}', but got {type(raw_value)}.")
                        typed_value = self._cast_and_validate_dict(raw_value, sub_schema)
                        typed_list.append((key, typed_value))
                else:  # Simple types
                    if not isinstance(raw_value, str):
                        raise TypeError(f"Expected a string for simple type conversion for key '{key}', but got {type(raw_value)}.")
                    # Treat an empty string value in the header as None.
                    if raw_value == '':
                        typed_value = None
                    else:
                        typed_value = requested_type(raw_value)
                    typed_list.append((key, typed_value))

            except (ValueError, TypeError, KeyError) as e:
                raise type(e)(f"Error processing key '{key}': {e}") from e
        return typed_list

    def _cast_and_validate_dict(self, raw_dict: dict[str, Any], schema: list[tuple[str, Any]]) -> dict[str, Any]:
        """Helper to cast a simple dictionary against an ordered list-of-tuples schema."""
        typed_dict = {}
        for key, requested_type in schema:
            if key.lower() not in raw_dict:
                raise KeyError(f"Required header key '{key}' not found in nested dictionary.")
            
            raw_value = raw_dict[key.lower()]
            try:
                typed_dict[key] = requested_type(raw_value)
            except (ValueError, TypeError) as e:
                raise type(e)(f"Error processing nested key '{key}': {e}") from e
        return typed_dict

    def _process_row(self, row_dict: dict[str, str]) -> list[Any]:
        """Processes a single row dictionary into a list of dataclass instances."""
        processed_objects = []
        for data_type in self.data_types:
            typed_params = {}
            type_hints = get_type_hints(data_type)
            field_names = {f.name for f in fields(data_type)}

            for field_name in field_names:
                # Only process fields that are actually present in the CSV row.
                if field_name not in row_dict:
                    continue

                field_type = type_hints.get(field_name)
                value_str = row_dict.get(field_name)

                if value_str is None or (not value_str and field_name != "note"):
                    typed_params[field_name] = None
                    continue

                try:
                    if field_type is bool:
                        typed_params[field_name] = value_str.lower() == "true"
                    elif field_type is int:
                        typed_params[field_name] = int(float(value_str))
                    elif field_type is float:
                        typed_params[field_name] = float(value_str)
                    elif field_type is datetime:
                        typed_params[field_name] = datetime.fromisoformat(value_str)
                    elif field_type is timedelta:
                        typed_params[field_name] = timedelta(seconds=float(value_str))
                    elif field_type is str:
                        typed_params[field_name] = value_str
                    else:
                        # Try direct constructor call (works for ToneCurve and other types)
                        # If that fails, handle Optional[T] by attempting to cast to the inner type
                        try:
                            typed_params[field_name] = field_type(value_str)
                        except (ValueError, TypeError):
                            # This case handles Optional[T] by attempting to cast to the inner type.
                            # It's a simplification and might not cover all edge cases.
                            inner_type = get_args(field_type)[0]
                            typed_params[field_name] = inner_type(value_str)
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse value '{value_str}' for field '{field_name}' as {field_type}. Setting to None.")
                    typed_params[field_name] = None

            # Only create an instance if at least one of its fields has a non-None value.
            if typed_params and any(value is not None for value in typed_params.values()):
                processed_objects.append(data_type(**typed_params))

        return processed_objects

    def __iter__(self):
        if self._file is None or self._reader is None:
            raise RuntimeError("I/O operation on a closed file. The CsvReader may have been closed or not opened correctly.")

        # The file pointer was reset in the init, so we can iterate from the beginning.
        for row_dict in self._reader:
            processed_objects = self._process_row(row_dict)
            
            # Skip empty rows (e.g., rows with only commas) silently
            if not processed_objects:
                continue
                
            if self._single_type_requested:
                # If a single type was requested, return the first object.
                yield processed_objects[0]
            else:
                # Otherwise, return the full list of objects.
                yield processed_objects

