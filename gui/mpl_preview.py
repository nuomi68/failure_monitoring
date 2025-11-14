from __future__ import annotations

import matplotlib.pyplot as plt
from typing import Callable, Optional

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QSizePolicy,
    QDialog,
)


class InteractiveMplCanvas(FigureCanvas):
    """Reusable Matplotlib canvas with wheel zoom and drag-pan support."""

    def __init__(self, parent: QWidget | None = None):
        self.fig, self.ax = plt.subplots()
        super().__init__(self.fig)
        self.setParent(parent)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._home_xlim = None
        self._home_ylim = None
        self.mpl_connect("draw_event", self._on_first_draw)

        self.mpl_connect("scroll_event", self._on_scroll)
        self._is_panning = False
        self._pan_axes = None
        self._pan_last_xdata = None
        self._pan_last_ydata = None
        self._pan_last_xlim = None
        self._pan_last_ylim = None
        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("motion_notify_event", self._on_motion)

    def clear(self) -> None:
        self.ax.clear()
        self.draw_idle()

    def clear(self) -> None:
        """Mimic PlotCanvas API so callers can reuse plotting code."""
        self.ax.clear()
        self.draw_idle()

    # ---------------- Navigation helpers -----------------
    def _on_first_draw(self, event):
        if self._home_xlim is None and event.canvas is self:
            self._home_xlim = self.ax.get_xlim()
            self._home_ylim = self.ax.get_ylim()

    def reset_home(self):
        if self._home_xlim is None:
            return
        self.ax.set_xlim(self._home_xlim)
        self.ax.set_ylim(self._home_ylim)
        self.draw_idle()

    # ---------------- Wheel zoom -----------------
    def _on_scroll(self, event):
        if event.inaxes is None:
            return

        if getattr(self, "toolbar", None) is not None:
            if getattr(self.toolbar, "mode", ""):
                return

        base_scale = 1.2
        scale = 1 / base_scale if event.button == "up" else base_scale
        ax = event.inaxes
        xdata, ydata = event.xdata, event.ydata
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        new_xlim = [
            xdata - (xdata - xlim[0]) * scale,
            xdata + (xlim[1] - xdata) * scale,
        ]
        new_ylim = [
            ydata - (ydata - ylim[0]) * scale,
            ydata + (ylim[1] - ydata) * scale,
        ]
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        self.draw_idle()

    # ---------------- Drag pan -----------------
    def _on_press(self, event):
        if event.button != 1 or event.inaxes is None:
            return
        if getattr(self, "toolbar", None) is not None:
            if getattr(self.toolbar, "mode", ""):
                return
        self._is_panning = True
        self._pan_axes = event.inaxes
        self._pan_last_xdata = event.xdata
        self._pan_last_ydata = event.ydata
        self._pan_last_xlim = self._pan_axes.get_xlim()
        self._pan_last_ylim = self._pan_axes.get_ylim()

    def _on_motion(self, event):
        if not self._is_panning:
            return
        if event.inaxes is not self._pan_axes:
            return
        if event.xdata is None or event.ydata is None:
            return
        ax = self._pan_axes
        dx = event.xdata - self._pan_last_xdata
        dy = event.ydata - self._pan_last_ydata
        ax.set_xlim(self._pan_last_xlim[0] - dx, self._pan_last_xlim[1] - dx)
        ax.set_ylim(self._pan_last_ylim[0] - dy, self._pan_last_ylim[1] - dy)
        self.draw_idle()
        self._pan_last_xdata = event.xdata
        self._pan_last_ydata = event.ydata
        self._pan_last_xlim = ax.get_xlim()
        self._pan_last_ylim = ax.get_ylim()

    def _on_release(self, event):
        if event.button != 1:
            return
        self._is_panning = False
        self._pan_axes = None


class InteractiveToolbar(NavigationToolbar2QT):
    def home(self, *args):
        if hasattr(self.canvas, "reset_home"):
            self.canvas.reset_home()
        else:
            super().home(*args)


class MatplotlibPreviewDialog(QDialog):
    """Reusable dialog containing the interactive canvas plus toolbar."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        canvas_factory: Optional[Callable[[QWidget], FigureCanvas]] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("图表放大预览")
        self.resize(900, 720)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        if canvas_factory:
            self.canvas = canvas_factory(self)
        else:
            self.canvas = InteractiveMplCanvas(self)
        toolbar = InteractiveToolbar(self.canvas, self)
        self.canvas.toolbar = toolbar

        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)


__all__ = [
    "InteractiveMplCanvas",
    "InteractiveToolbar",
    "MatplotlibPreviewDialog",
]
