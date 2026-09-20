"""Use tqdm counters in terminals and widget-free HTML inside notebooks."""

from __future__ import annotations

import sys
from html import escape
from typing import Any
from weakref import WeakSet

from IPython import get_ipython
from IPython.display import DisplayHandle
from tqdm.std import tqdm as TerminalProgress


def in_notebook() -> bool:
    return getattr(get_ipython(), "kernel", None) is not None


class NotebookProgress(TerminalProgress):  # type: ignore[misc]
    """Render nested tqdm counters without terminal cursor controls or widgets."""

    # tqdm's monitor refreshes bars from a background thread. IPython display handles
    # belong to the notebook cell's context, so publishing from that thread can raise
    # ``LookupError: parent_header`` while training itself is still healthy.
    monitor_interval = 0
    # tqdm otherwise shares one instance registry across subclasses. Keeping the
    # notebook bars separate also prevents a monitor started by terminal tqdm from
    # discovering and refreshing an IPython display handle in its worker thread.
    _instances: WeakSet[Any] = WeakSet()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.handle: Any = None
        kwargs["gui"] = True
        kwargs["disable"] = bool(kwargs.get("disable", False))
        kwargs.setdefault("mininterval", 0.5)
        super().__init__(*args, **kwargs)
        if not self.disable:
            self.display()

    def display(self, msg: str | None = None, pos: int | None = None) -> bool:
        if self.disable:
            return False
        if self.handle is None:
            self.handle = DisplayHandle()  # type: ignore[no-untyped-call]
            publish = self.handle.display
        else:
            publish = self.handle.update
        values = self.format_dict
        total = values["total"]
        attributes = f'max="{total}" value="{self.n}"' if total else ""
        summary = msg if msg is not None else self.format_meter(**{**values, "ncols": 0})
        label = escape(str(self.desc or "Progress"), quote=True)
        publish(
            {
                "text/plain": summary,
                "text/html": (
                    '<div class="training-progress" style="margin:6px 0;max-width:900px">'
                    f'<progress aria-label="{label}" {attributes} style="width:100%"></progress>'
                    '<div style="font-family:monospace;white-space:pre-wrap">'
                    f"{escape(summary)}</div>"
                    "</div>"
                ),
            },
            raw=True,
        )
        return True

    def clear(self, *args: Any, **kwargs: Any) -> None:
        if self.handle is not None:
            self.handle.update({"text/html": "", "text/plain": ""}, raw=True)

    def close(self) -> None:
        if getattr(self, "disable", True):
            return
        if self.leave or self.total is None or self.n < self.total:
            self.display()
        else:
            self.clear()
        super().close()


def tqdm(*args: Any, **kwargs: Any) -> Any:
    """Create a consistent progress bar for training, evaluation, and analysis."""

    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("mininterval", 0.5)
    backend = NotebookProgress if in_notebook() else TerminalProgress
    return backend(*args, **kwargs)


def status(message: str, *, enabled: bool = True) -> None:
    """Print a durable status line independently of machine-readable artifacts."""

    if enabled:
        print(message, file=sys.stdout if in_notebook() else sys.stderr, flush=True)
