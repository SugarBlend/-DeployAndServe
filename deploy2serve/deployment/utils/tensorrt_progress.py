import os
import sys
from typing import Any

import tensorrt as trt


class ProgressMonitor(trt.IProgressMonitor):
    def __init__(self) -> None:
        super().__init__()
        self._active_phases: dict[str, dict[str, Any]] = {}
        self._step_result: bool = True

    def phase_start(self, phase_name: str, parent_phase: str, num_steps: int) -> None:
        try:
            nb_indents = 1 + self._active_phases[parent_phase]["nbIndents"] if parent_phase is not None else 0
            self._active_phases[phase_name] = {
                "title": phase_name,
                "steps": 0,
                "num_steps": num_steps,
                "nbIndents": nb_indents,
            }
            self._redraw()
        except KeyboardInterrupt:
            # The phase_start callback cannot directly cancel the build, so request the cancellation from
            # within step_complete.
            _step_result = False

    def phase_finish(self, phase_name: str) -> None:
        try:
            del self._active_phases[phase_name]
            self._redraw(blank_lines=1)  # Clear the removed phase.
        except KeyboardInterrupt:
            _step_result = False

    def step_complete(self, phase_name: str, step: int) -> bool:
        try:
            self._active_phases[phase_name]["steps"] = step
            self._redraw()
        except KeyboardInterrupt:
            # There is no need to propagate this exception to TensorRT. We can simply cancel the build.
            return False
        else:
            return self._step_result

    def _redraw(self, *, blank_lines: int = 0) -> None:
        # The Python curses module is not widely available on Windows platforms.
        # Instead, this function uses raw terminal escape sequences. See the sample documentation for references.
        def clear_line() -> None:
            print("\x1b[2K", end="")

        def move_to_start_of_line() -> None:
            print("\x1b[0G", end="")

        def move_cursor_up(lines: str) -> None:
            print(f"\x1b[{lines}A", end="")

        def progress_bar(steps: int, num_steps: int) -> str:
            inner_width = 10
            completed_bar_chars = int(inner_width * steps / float(num_steps))
            return "[{}{}]".format("=" * completed_bar_chars, "-" * (inner_width - completed_bar_chars))

        # Set max_cols to a default of 200 if not run in interactive mode.
        max_cols = os.get_terminal_size().columns if sys.stdout.isatty() else 200

        move_to_start_of_line()
        for phase in self._active_phases.values():
            phase_prefix = "{indent}{bar} {title}".format(
                indent=" " * phase["nbIndents"],
                bar=progress_bar(phase["steps"], phase["num_steps"]),
                title=phase["title"],
            )
            phase_suffix = "{steps}/{num_steps}".format(**phase)
            allowable_prefix_chars = max_cols - len(phase_suffix) - 2
            if allowable_prefix_chars < len(phase_prefix):
                phase_prefix = phase_prefix[0 : allowable_prefix_chars - 3] + "..."
            clear_line()
            print(phase_prefix, phase_suffix)
        for _ in range(blank_lines):
            clear_line()
            print()
        move_cursor_up(len(self._active_phases) + blank_lines)
        sys.stdout.flush()
