# Copyright (c) Meta Platforms, Inc. and affiliates.

"""The demo app's look and its run chrome: theme, header, credits, Run/Stop, status lines.

Taken from the GenerativeReconstruction framework's ``gradio/_gradio_common.py``, whose demo
app this one is laid out like (the look is ``style.py``'s, after 3DView). The status line and
the step table read the run itself (its log banners, its ``config.yaml``) instead of the
framework's launchers. Nothing here names a method: the startup markers and the block list
come from ``demo/method.py``.
"""
import contextlib
import functools
import html
import json
import re
import signal
import subprocess
from pathlib import Path

import gradio as gr

from . import style


def theme():
    """The app theme: Gradio's base with 3DView's palette and font (``style.TOKENS``)."""
    return gr.themes.Base(font=style.FONT, font_mono=style.FONT_MONO,
                          text_size=gr.themes.Size(**style.TEXT_SIZE)).set(**style.TOKENS)


# Dark is the app's only mode (3DView's dark theme is the one it is styled after): Gradio 5
# resolves the mode client-side, so the class it keys off is set on load, and the URL stamped
# so a refresh comes back dark.
_DARK_ONLY = """() => {
  document.body.classList.add('dark');
  const url = new URL(window.location);
  if (url.searchParams.get('__theme') !== 'dark') {
    url.searchParams.set('__theme', 'dark');
    history.replaceState(null, '', url);
  }
}"""


def startup_js() -> str:
    """Everything the page does on load, as the one function ``gr.Blocks(js=…)`` takes."""
    return _DARK_ONLY


# The Stop button is red the hard way (Gradio aliases `button_cancel_*` to the secondary
# style), and Gradio's streaming indicators are silenced: a run streams for minutes, and the
# status line already says what is happening.
CSS = style.CSS + """
.stop-btn { background: #dc2626 !important; border-color: #dc2626 !important;
            color: #fff !important; }
.stop-btn:hover { background: #b91c1c !important; border-color: #b91c1c !important; }

.generating { border: none !important; animation: none !important;
              box-shadow: none !important; }
.progress-bar, .progress-bar-wrap, .progress-level, .progress-level-inner,
.progress-text, .eta-bar { display: none !important; }
"""


@contextlib.contextmanager
def header(title: str):
    """The title bar, a 3DView glass panel; what is built inside lands on its right."""
    with gr.Row(elem_classes=["app-header"]):
        gr.HTML(
            '<div class="app-header-brand">'
            f'<span class="app-header-title">{html.escape(title)}</span>'
            '</div>',
            padding=False,
        )
        with gr.Row(elem_classes=["app-header-nav"]):
            yield


def credits():
    """The footer: the 3D view, and the look the page takes from it, are 3DView's."""
    return gr.HTML('<div class="app-credits">3D view and look: <a href="https://github.com/'
                   's-esposito/3DView" target="_blank" rel="noopener noreferrer">3DView</a></div>',
                   padding=False)


def run_stop_buttons(label: str):
    """The Run button and the red Stop that takes its place while a run is in flight."""
    run = gr.Button(label, variant="primary")
    stop = gr.Button("⏹ Stop", variant="stop", visible=False, elem_classes=["stop-btn"])
    return run, stop


def wire_run(run, stop, handler, inputs, outputs, halt, lock=()):
    """Swap Run for Stop, stream ``handler`` into ``outputs``, swap back (also on failure).

    The swaps and ``halt`` are ``queue=False``: the app runs one job at a time, so a queued
    Stop would wait behind the very run it is meant to stop. ``halt`` reports into the
    status line, ``outputs[1]``. ``lock`` goes non-interactive for the run's duration.
    """
    def swap(running: bool):
        return (gr.update(visible=not running), gr.update(visible=running),
                *(gr.update(interactive=not running) for _ in lock))

    swapped = [run, stop, *lock]
    ev = run.click(lambda: swap(True), outputs=swapped, queue=False)
    ev = ev.then(handler, inputs=inputs, outputs=outputs)
    ev.then(lambda: swap(False), outputs=swapped, queue=False)
    stop.click(halt, outputs=outputs[1], queue=False)


@functools.lru_cache(maxsize=1)
def gpu_error() -> str | None:
    """A message if this machine has no usable GPU, else None (``nvidia-smi``: no import)."""
    try:
        probe = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        if probe.returncode == 0 and probe.stdout.strip():
            return None
        why = f"`nvidia-smi -L` exited {probe.returncode}"
    except OSError:
        why = "`nvidia-smi` not found"
    return (f"No usable GPU here ({why}): reconstruction needs CUDA. "
            "Launch this app on a GPU machine.")


def gpu_memory_gib() -> tuple[float, float] | None:
    """``(total, free)`` GiB of the first GPU, or None (``nvidia-smi``, so this process
    opens no CUDA context of its own)."""
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free",
             "--format=csv,noheader,nounits"], capture_output=True, text=True)
        total, free = (float(v) / 1024 for v in probe.stdout.splitlines()[0].split(","))
        return total, free
    except (OSError, IndexError, ValueError):
        return None


def mmss(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


# --- what the run is doing, read off its own log and config ---------------------------

#: A block starts with a rule naming it, e.g. ``──── SHAPE_INIT: Shape Init (…) ────``;
#: FINAL starts with the ``EVALUATION`` one.
_BANNER = re.compile(r"─ ([A-Z][A-Z0-9_]+): (\S*)")
_DONE = "Pipeline complete!"
_STATUS_MARK = {"done": "✅", "running": "🔄", "pending": "⚪"}


def _steps(run_dir: Path, blocks) -> list[str] | None:
    """The run's steps in order: preprocessing, each enabled block of ``blocks`` (``(block,
    config section)`` in run order), final; None before its ``config.yaml`` lands (it is
    written before the first block)."""
    cfg_path = Path(run_dir) / "config.yaml"
    if not cfg_path.is_file():
        return None
    import yaml
    try:
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    except yaml.YAMLError:
        return None   # caught mid-write
    enabled = [b for b, section in blocks if (cfg.get(section) or {}).get("enabled")]
    return ["preprocessing", *enabled, "final"]


def starts_block(line: str) -> bool:
    """Whether ``line`` is the banner a block starts with."""
    return bool(_BANNER.search(line))


def _running_step(log: str) -> str:
    # A block's own evaluation ("EVALUATION: After <block>") is still that block's step;
    # only "EVALUATION: Final state" is FINAL's.
    banners = [name for name, word in _BANNER.findall(log)
               if name != "EVALUATION" or word == "Final"]
    if not banners:
        return "preprocessing"
    last = banners[-1].lower()
    return "final" if last == "evaluation" else last


def status_text(finished: bool, returncode, log: str, elapsed: float, phases=()) -> str:
    """The one-line run status, with an elapsed clock as the liveness cue. ``phases`` are
    ``(log marker, label)`` pairs naming the startup work before the first block."""
    if not finished:
        phase = "Warming up…"
        for marker, label in phases:
            if marker in log:
                phase = label
        if _BANNER.search(log):
            phase = f"Running `{_running_step(log)}`…"
        return f"⏳ {phase} · {mmss(elapsed)}"
    if returncode == 0:
        return f"✅ Done in {mmss(elapsed)}"
    if returncode in (-signal.SIGTERM, -signal.SIGKILL):
        return "⏹ Stopped"
    return f"❌ Failed (exit code {returncode}); the log says why"


#: A line a method prints to report a loop's progress (GenIA: ``core/utils/progress.py``):
#: the prefix, then JSON ``{"task", "n", "total"}``, plus ``"stopped": true`` for a loop that
#: ended short of its total. A method that prints none gets no bars.
PROGRESS_PREFIX = "@@progress "


def read_progress(line: str):
    """``(task, n, total, stopped)`` from a progress line, or None for any other line."""
    if not line.startswith(PROGRESS_PREFIX):
        return None
    try:
        msg = json.loads(line[len(PROGRESS_PREFIX):])
        return str(msg["task"]), int(msg["n"]), msg.get("total"), bool(msg.get("stopped"))
    except (ValueError, KeyError, TypeError):
        return None


def task_bars(tasks: dict) -> str:
    """One bar per task the running block has reported (``{task: (n, total, stopped)}``),
    after 3DView's slider tracks; empty when there is none. A loop that stopped early shows
    full, its count saying so."""
    if not tasks:
        return ""
    rows = []
    for task, (n, total, stopped) in tasks.items():
        if stopped:
            width = 100.0
        elif total:
            width = 100.0 * min(n / total, 1.0)
        else:
            width = 0.0
        count = f"{n} / {total if total else '?'}" + (" · stopped early" if stopped else "")
        rows.append(
            f'<div class="task-row"><span>{html.escape(task)}</span>'
            f'<span class="task-count">{count}</span>'
            f'<div class="task-track"><div class="task-fill" style="width:{width:.1f}%">'
            '</div></div></div>')
    return '<div class="task-list">' + "".join(rows) + "</div>"


def block_progress(run_dir: Path, log: str, name: str, blocks) -> str:
    """The step table as Markdown: a progress line, then one row per step."""
    steps = _steps(run_dir, blocks)
    if steps is None:
        return f"**{name}** · starting…"
    complete = _DONE in log
    current = None if complete else _running_step(log)
    index = len(steps) if complete else (steps.index(current) if current in steps else 0)
    lines = [f"**{name}** · {index}/{len(steps)} steps"]
    for i, name in enumerate(steps):
        status = "done" if i < index else "running" if i == index else "pending"
        label = f"**`{name}`**" if status == "running" else f"`{name}`"
        lines.append(f"{_STATUS_MARK[status]}&nbsp; {label}")
    return "  \n".join(lines)
