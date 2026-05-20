# Released under a modified PolyForm Small Business License.
# Free for small businesses, individuals, and academics. See LICENSE for details.

"""Common utilities for muimg."""
from __future__ import annotations

import logging
import sys
import threading
import time
import weakref

from enum import Enum
from typing import Type

logger = logging.getLogger(__name__)

_thread_local = threading.local()


def enum_display_name(enum_class: Type[Enum], value: int, suffix: str = "") -> str:
    """
    Get display name for an enum value.
    
    Converts enum member name to display format (e.g., MAIN_IMAGE -> MainImage).
    Returns formatted value string if enum member not found.
    
    Args:
        enum_class: The enum class to look up
        value: The numeric value to find
        suffix: Optional suffix to append (e.g., " compression")
        
    Returns:
        Display name string (e.g., "MainImage") or "Type{value}" if not found
    """
    try:
        member = enum_class(value)
        display = ''.join(word.capitalize() for word in member.name.split('_'))
        return f"{display}{suffix}" if suffix else display
    except ValueError:
        return f"Type{value}{suffix}" if suffix else f"Type{value}"


def enum_from_value(enum_class: Type[Enum], value: int) -> Enum | None:
    """
    Get enum member from numeric value.
    
    Args:
        enum_class: The enum class to look up
        value: The numeric value to find
        
    Returns:
        Enum member or None if not found
    """
    try:
        return enum_class(value)
    except ValueError:
        return None


def enum_from_string(enum_class: Type[Enum], value: str) -> Enum:
    """
    Get enum member from string value.
    
    For string enums (inheriting from str and Enum), this looks up the member
    by its string value. Raises KeyError if not found.
    
    Args:
        enum_class: The enum class to look up
        value: The string value to find
        
    Returns:
        Enum member
        
    Raises:
        KeyError: If value not found in enum
    """
    # For str enums, we can iterate and compare values
    for member in enum_class:
        if member.value == value:
            return member
    raise KeyError(f"'{value}' is not a valid {enum_class.__name__}")


def setup_logging(verbosity: int = 0) -> None:
    """
    Set up logging configuration based on verbosity level.
    
    Args:
        verbosity: Verbosity level (0=ERROR, 1=WARNING, 2=INFO, 3+=DEBUG)
    """
    if verbosity == 0:
        level = logging.ERROR
    elif verbosity == 1:
        level = logging.WARNING
    elif verbosity == 2:
        level = logging.INFO
    else:
        level = logging.DEBUG
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stderr
    )
    
    # Set specific logger levels
    logger = logging.getLogger('muimg')
    logger.setLevel(level)


class PerfTimer:
    """Hierarchical performance timer for tracking code execution.

    Every node in the timing tree is a PerfTimer. Root nodes (depth=-1) are
    created directly via PerfTimer(name) and additionally manage thread-local
    registration and report generation. Child nodes are created via
    start_step() and share the same API.

    Thread-safe: each root timer instance stores per-thread state in its own
    threading.local(), isolated from all other PerfTimer instances.
    Use naming convention "(c++)" suffix to indicate C++ processing steps.

    Usage:
        timer = PerfTimer("my_operation")
        step = timer.start_step("decode")
        # ... work ...
        step.close()                 # end this specific step
        timer.start_step("process")  # auto-closes previous sibling
        # ... work ...
        timer.end_step()             # end the most recent child of timer
        timer.log_report(logger)
        timer.close()                # end any open child + deactivate thread-local
    """

    def __init__(
        self,
        name: str,
        _parent: "PerfTimer | None" = None,
        _depth: int = -1,
    ):
        self.name = name
        self.parent = _parent
        self.depth = _depth
        self.children: list[PerfTimer] = []
        self.start_time = time.perf_counter()
        self.end_time: float | None = None
        self._active_child: PerfTimer | None = None

        if self._is_root:
            self._local = threading.local()
            existing_ref = getattr(_thread_local, 'active_timer_node', None)
            existing = existing_ref() if existing_ref is not None else None
            if existing is not None:
                logger.warning(
                    f"PerfTimer '{name}' activated while '{existing.name}' "
                    f"is already active on this thread"
                )
            _thread_local.active_timer_node = weakref.ref(self)

    @property
    def _is_root(self) -> bool:
        return self.depth == -1

    def start_step(self, name: str) -> "PerfTimer":
        """Start a new child step, auto-closing the previous sibling if open."""
        if self._active_child is not None and self._active_child.end_time is None:
            self._active_child.close()

        child = PerfTimer(name, _parent=self, _depth=self.depth + 1)
        self.children.append(child)
        self._active_child = child
        return child

    def __del__(self):
        if self.end_time is None:
            self.close()

    def close(self):
        """End this specific node, auto-closing any open children.

        On root nodes, also deactivates this timer as the thread-local context.
        """
        if self.end_time is not None:
            logger.warning(f"close() called twice on '{self.name}'")
            return

        if self._active_child is not None:
            self._active_child.close()

        self.end_time = time.perf_counter()

        if self.parent and self.parent._active_child is self:
            self.parent._active_child = None

        if self._is_root:
            current_ref = getattr(_thread_local, 'active_timer_node', None)
            current = current_ref() if current_ref is not None else None
            if current is self:
                _thread_local.active_timer_node = None
            else:
                logger.warning(
                    f"PerfTimer '{self.name}' closed but active_timer_node is "
                    f"'{current.name if current is not None else None}' - another timer may have stomped the slot"
                )

    def end_step(self):
        """End the most recently started child step of this node."""
        if self._active_child is None:
            logger.warning(f"end_step() called on '{self.name}' but no active child")
            return
        self._active_child.close()

    def get_elapsed_ms(self) -> float:
        """Return elapsed time in milliseconds, using current time if not yet ended."""
        end = self.end_time if self.end_time is not None else time.perf_counter()
        return (end - self.start_time) * 1000

    def log_report(self, logger_instance, level=logging.INFO):
        """Log the timing report if the logger is enabled at the given level."""
        if logger_instance.isEnabledFor(level):
            logger_instance.log(level, "\n" + self.get_report())

    def get_report(self) -> str:
        """Generate a formatted hierarchical timing report for this node's subtree.

        When called on the root, wall-clock time covers the full pipeline.
        When called on a child node, wall-clock time is that node's elapsed time.
        """
        report_root = self
        children = report_root.children

        if not children:
            return "No timing data recorded"

        # Wall-clock baseline: elapsed time of this node (root or child)
        wall_clock_time = self.get_elapsed_ms()

        if wall_clock_time == 0:
            return "No timing data recorded"

        # Depth offset so child reports start indentation at 0
        depth_offset = report_root.depth + 1

        lines = []
        lines.append(f"Rendering Performance ({report_root.name}):" if not self._is_root
                     else "Rendering Performance:")

        def get_max_width(node: PerfTimer, current_max: int = 0) -> int:
            indent = "  " * (node.depth - depth_offset)
            current_max = max(current_max, len(indent + node.name))
            for child in node.children:
                current_max = get_max_width(child, current_max)
            return current_max

        name_width = max(25, get_max_width(report_root))
        header = f"{'Step':<{name_width}}  {'Total':>10}  {'%':>6}"
        lines.append(header)
        lines.append("─" * len(header))

        root_depth = report_root.depth  # depth of the report root (children are root_depth+1)

        def add_step_rows(node: PerfTimer, prev_end_time: float | None):
            # Gap detection only at the immediate children of report_root
            if node.depth == root_depth + 1 and prev_end_time is not None:
                gap_ms = (node.start_time - prev_end_time) * 1000
                if gap_ms > 4.0:
                    gap_pct = (gap_ms / wall_clock_time * 100) if wall_clock_time > 0 else 0
                    lines.append(
                        f"{'unallocated':<{name_width}}  {gap_ms:>9.1f}ms  {gap_pct:>5.1f}%"
                    )

            indent = "  " * (node.depth - depth_offset)
            elapsed_ms = node.get_elapsed_ms()
            pct = (elapsed_ms / wall_clock_time * 100) if wall_clock_time > 0 else 0
            lines.append(
                f"{indent + node.name:<{name_width}}  {elapsed_ms:>9.1f}ms  {pct:>5.1f}%"
            )

            for child in node.children:
                add_step_rows(child, prev_end_time)

            return node.end_time if node.depth == root_depth + 1 else prev_end_time

        prev_end = report_root.start_time
        for child in children:
            prev_end = add_step_rows(child, prev_end) or prev_end

        # Final gap after last step (up to when the root was closed, or now if still running)
        root_end = report_root.end_time if report_root.end_time is not None else time.perf_counter()
        if prev_end is not None:
            final_gap_ms = (root_end - prev_end) * 1000
            if final_gap_ms > 4.0:
                gap_pct = (final_gap_ms / wall_clock_time * 100) if wall_clock_time > 0 else 0
                lines.append(
                    f"{'unallocated':<{name_width}}  {final_gap_ms:>9.1f}ms  {gap_pct:>5.1f}%"
                )

        lines.append("─" * len(header))
        lines.append(f"{'TOTAL':<{name_width}}  {wall_clock_time:>9.1f}ms  100.0%")

        return "\n".join(lines)


def get_active_timer() -> PerfTimer:
    """Get the currently active timer node for this thread.

    Returns an orphan node if no timer is active, so callers never need to
    check for None.
    """
    ref = getattr(_thread_local, 'active_timer_node', None)
    root = ref() if ref is not None else None
    if root is None:
        return PerfTimer("__orphan__", _depth=0)
    node = root
    while node._active_child is not None and node._active_child.end_time is None:
        node = node._active_child
    return node


class ScopedPerfTimer:
    """Context manager that provides scoped performance timing.
    
    Uses existing active timer if available, otherwise creates new root timer.
    Only logs report for root timers (depth == -1) unless auto_log is disabled.
    
    Usage:
        with scoped_perf_timer("my_operation", logger) as timer:
            # ... work ...
    """
    
    def __init__(self, name: str, logger_instance, auto_log: bool = True):
        self.name = name
        self.logger = logger_instance
        self.timer: PerfTimer | None = None
        self.auto_log = auto_log
        
    def __enter__(self) -> PerfTimer:
        # Repeat get_active_timer logic to avoid creating orphan timer
        ref = getattr(_thread_local, 'active_timer_node', None)
        root = ref() if ref is not None else None
        if root is not None:
            # Find the active child
            node = root
            while node._active_child is not None and node._active_child.end_time is None:
                node = node._active_child
            self.timer = node
        else:
            # No active timer, create new root timer
            self.timer = PerfTimer(self.name)
        return self.timer
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.timer is not None:
            # Only close timers we created (root timers with depth == -1)
            if self.timer.depth == -1:
                self.timer.close()
                # Only log if auto_log is True
                if self.auto_log:
                    self.timer.log_report(self.logger)
    
    def enter(self) -> PerfTimer:
        """Start the timer manually (equivalent to __enter__)."""
        return self.__enter__()
    
    def close(self, exc_type=None, exc_val=None, exc_tb=None):
        """End the timer manually (equivalent to __exit__)."""
        return self.__exit__(exc_type, exc_val, exc_tb)


def scoped_perf_timer(name: str, logger_instance, auto_log: bool = True) -> ScopedPerfTimer:
    """Create a ScopedPerfTimer context manager."""
    return ScopedPerfTimer(name, logger_instance, auto_log)

