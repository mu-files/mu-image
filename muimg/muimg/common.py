# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files

"""Common utilities for muimg."""
from __future__ import annotations

import logging
import sys
import threading
import time
import weakref

from enum import Enum
from typing import Sequence, TypeVar

logger = logging.getLogger(__name__)

_thread_local = threading.local()

_EnumT = TypeVar("_EnumT", bound=Enum)


def enum_display_name(enum_class: type[Enum], value: int, suffix: str = "") -> str:
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


def enum_from_value(enum_class: type[_EnumT], value: int) -> _EnumT | None:
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


def enum_from_string(enum_class: type[_EnumT], value: str) -> _EnumT:
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
    try:
        return enum_class(value)
    except ValueError as e:
        raise KeyError(f"'{value}' is not a valid {enum_class.__name__}") from e


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
    """Hierarchical wall-clock timer (optional instrumentation).

        timer = PerfTimer("render_raw", log=logger)
        PerfTimer.step("decode_cfa")   # under current (timer here)
        ...
        setup = PerfTimer.step("render_setup")
        ...
        setup.close()
        timer.close()  # logs if log= was set

    ``step(name)`` starts a child under ``current()``. When nothing is timing it
    returns a no-op handle so callers can always ``close()`` without checks.
    Nested work (e.g. ``graph.compute``) records under the current stage via the
    stack. An inconsistent stack yields ``broken stack``.

    ``with PerfTimer(...)`` still works and auto-nests, but is not required.
    ``PerfTimer.current()`` is the deepest open timer on this thread.
    """

    class _NoopStep:
        """Returned by ``step`` when nothing is timing — ``close()`` is a no-op."""

        __slots__ = ()
        end_time: float | None = 0.0  # already "closed" for end_time checks

        def close(self) -> None:
            return None

    #: No-op step handle (``close()`` does nothing). Use when a step is optional.
    inactive = _NoopStep()

    @classmethod
    def _stack(cls) -> list["PerfTimer"]:
        stack = getattr(_thread_local, "timer_stack", None)
        if stack is None:
            stack = []
            _thread_local.timer_stack = stack
        return stack

    @classmethod
    def current(cls) -> "PerfTimer | None":
        """Deepest open PerfTimer on this thread, or None."""
        stack = cls._stack()
        return stack[-1] if stack else None

    @classmethod
    def root(cls) -> "PerfTimer | None":
        """Outermost open PerfTimer on this thread, or None."""
        stack = cls._stack()
        if not stack:
            return None
        t = stack[0]
        while t.parent is not None:
            t = t.parent
        return t

    @classmethod
    def step(cls, name: str) -> "PerfTimer | PerfTimer._NoopStep":
        """Start a host stage under the current timer.

        When idle, returns a no-op handle whose ``close()`` does nothing.
        """
        top = cls.current()
        if top is None:
            return cls.inactive
        return top.start_step(name)

    @classmethod
    def _push(cls, timer: "PerfTimer") -> None:
        stack = cls._stack()
        if stack and stack[-1] is timer:
            return
        stack.append(timer)

    @classmethod
    def _pop(cls, timer: "PerfTimer") -> None:
        stack = cls._stack()
        if stack and stack[-1] is timer:
            stack.pop()
            return
        for i in range(len(stack) - 1, -1, -1):
            if stack[i] is timer:
                root = cls.root()
                if root is not None:
                    root._broken = True
                del stack[i]
                break

    def __init__(
        self,
        name: str,
        _parent: "PerfTimer | None" = None,
        _depth: int | None = None,
        start_time: float | None = None,
        *,
        log: logging.Logger | None = None,
        _register_root: bool = True,
    ):
        self.name = name
        self.parent = _parent
        self.children: list[PerfTimer] = []
        self.start_time = start_time if start_time is not None else time.perf_counter()
        self.end_time: float | None = None
        self._active_child: PerfTimer | None = None
        self._log = log
        self._on_stack = False
        self._broken = False

        if _parent is not None:
            self.depth = _parent.depth + 1 if _depth is None else _depth
        elif _depth is not None:
            self.depth = _depth
        else:
            self.depth = -1

        if self._is_root and _register_root:
            self._local = threading.local()
            top = PerfTimer.current()
            if top is None:
                _thread_local.active_timer_node = weakref.ref(self)
                PerfTimer._push(self)
                self._on_stack = True
            # else: another timer is open — __enter__ will nest under it

    @property
    def _is_root(self) -> bool:
        return self.depth == -1 and self.parent is None

    def __enter__(self) -> "PerfTimer":
        top = PerfTimer.current()
        if self.parent is None and top is not None and top is not self:
            # Constructed as a root/standalone while another timer is open → nest.
            if self._on_stack:
                PerfTimer._pop(self)
                self._on_stack = False
            if top._active_child is not None and top._active_child.end_time is None:
                top._active_child.close()
            self.parent = top
            self.depth = top.depth + 1
            top.children.append(self)
            top._active_child = self
        if not self._on_stack:
            PerfTimer._push(self)
            self._on_stack = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def start_step(self, name: str) -> "PerfTimer":
        """Start a new child step, auto-closing the previous sibling if open."""
        if self._active_child is not None and self._active_child.end_time is None:
            self._active_child.close()

        child = PerfTimer(
            name, _parent=self, _depth=self.depth + 1, _register_root=False
        )
        self.children.append(child)
        self._active_child = child
        PerfTimer._push(child)
        child._on_stack = True
        return child

    def add_completed_step(
        self,
        name: str,
        duration_s: float,
        *,
        end_time: float | None = None,
    ) -> "PerfTimer":
        """Append a finished child with a known duration (e.g. native op timings).

        Places the interval ending at ``end_time`` (default: ``perf_counter()``).
        Does not become the active child. For several ops from one native segment,
        use ``add_completed_steps`` so intervals are laid out end-to-end.
        """
        if self._active_child is not None and self._active_child.end_time is None:
            self._active_child.close()

        duration_s = max(0.0, float(duration_s))
        end = time.perf_counter() if end_time is None else float(end_time)
        child = PerfTimer(
            name,
            _parent=self,
            _depth=self.depth + 1,
            start_time=end - duration_s,
            _register_root=False,
        )
        child.end_time = end
        self.children.append(child)
        self._active_child = None
        return child

    def add_completed_steps(
        self,
        steps: Sequence[tuple[str, float]],
    ) -> list["PerfTimer"]:
        """Append finished children laid out end-to-end (no overlapping intervals).

        For a single op use ``add_completed_step``. This places several durations
        sequentially ending at ``perf_counter()`` (a bare loop of
        ``add_completed_step`` would pin every end to ``now`` and overlap).
        """
        step_list = list(steps)
        if not step_list:
            raise ValueError(
                "add_completed_steps() requires at least one (name, duration_s)"
            )

        total_s = sum(max(0.0, float(d)) for _, d in step_list)
        cursor = time.perf_counter() - total_s
        children: list[PerfTimer] = []
        for name, duration_s in step_list:
            duration_s = max(0.0, float(duration_s))
            end = cursor + duration_s
            children.append(
                self.add_completed_step(name, duration_s, end_time=end)
            )
            cursor = end
        return children

    def __del__(self):
        if self.end_time is None:
            self.close()

    def close(self):
        """End this specific node, auto-closing any open children.

        On root nodes, also deactivates this timer as the thread-local context.
        If the stack was left inconsistent, the report is ``broken stack``.
        """
        if self.end_time is not None:
            logger.warning(f"close() called twice on '{self.name}'")
            return

        if self._active_child is not None:
            self._active_child.close()

        self.end_time = time.perf_counter()

        if self.parent and self.parent._active_child is self:
            self.parent._active_child = None

        if self._on_stack:
            PerfTimer._pop(self)
            self._on_stack = False

        if self._is_root:
            stack = PerfTimer._stack()
            if stack:
                self._broken = True
                stack.clear()
            current_ref = getattr(_thread_local, "active_timer_node", None)
            current = current_ref() if current_ref is not None else None
            if current is self:
                _thread_local.active_timer_node = None
            elif current is not None:
                self._broken = True
                _thread_local.active_timer_node = None
            if self._log is not None:
                self.log_report(self._log)

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

        The report root is printed as the top row; children are indented under it.
        Wall-clock TOTAL is this node's elapsed time.
        Returns ``broken stack`` if the timer stack was left inconsistent.
        """
        if self._broken:
            return "broken stack"

        report_root = self
        children = report_root.children

        if not children:
            return "No timing data recorded"

        # Wall-clock baseline: elapsed time of this node (root or child)
        wall_clock_time = self.get_elapsed_ms()

        if wall_clock_time == 0:
            return "No timing data recorded"

        # Indent so the report root is at column 0
        depth_offset = report_root.depth

        lines = []
        lines.append("Performance:")

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

        root_depth = report_root.depth  # children are root_depth+1
        child_indent = "  "

        def add_step_rows(node: PerfTimer, prev_end_time: float | None):
            # Gap detection only at the immediate children of report_root
            if node.depth == root_depth + 1 and prev_end_time is not None:
                gap_ms = (node.start_time - prev_end_time) * 1000
                if gap_ms > 4.0:
                    gap_pct = (gap_ms / wall_clock_time * 100) if wall_clock_time > 0 else 0
                    lines.append(
                        f"{child_indent + 'unallocated':<{name_width}}  "
                        f"{gap_ms:>9.1f}ms  {gap_pct:>5.1f}%"
                    )

            indent = "  " * (node.depth - depth_offset)
            elapsed_ms = node.get_elapsed_ms()
            pct = (elapsed_ms / wall_clock_time * 100) if wall_clock_time > 0 else 0
            lines.append(
                f"{indent + node.name:<{name_width}}  {elapsed_ms:>9.1f}ms  {pct:>5.1f}%"
            )

            if node.depth == root_depth:
                sibling_prev = node.start_time
                for child in node.children:
                    sibling_prev = add_step_rows(child, sibling_prev) or sibling_prev
                return sibling_prev

            for child in node.children:
                add_step_rows(child, prev_end_time)

            return node.end_time if node.depth == root_depth + 1 else prev_end_time

        prev_end = add_step_rows(report_root, None)

        # Final gap after last step (up to when the root was closed, or now if still running)
        root_end = report_root.end_time if report_root.end_time is not None else time.perf_counter()
        if prev_end is not None:
            final_gap_ms = (root_end - prev_end) * 1000
            if final_gap_ms > 4.0:
                gap_pct = (final_gap_ms / wall_clock_time * 100) if wall_clock_time > 0 else 0
                lines.append(
                    f"{child_indent + 'unallocated':<{name_width}}  "
                    f"{final_gap_ms:>9.1f}ms  {gap_pct:>5.1f}%"
                )

        lines.append("─" * len(header))
        lines.append(f"{'TOTAL':<{name_width}}  {wall_clock_time:>9.1f}ms  100.0%")

        return "\n".join(lines)
