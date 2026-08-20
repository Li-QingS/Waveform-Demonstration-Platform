# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import time
from collections import deque

import numpy as np
import pyqtgraph as pg
try:
    import pyqtgraph.opengl as gl
    _PG_OPENGL_AVAILABLE = True
except Exception:
    gl = None
    _PG_OPENGL_AVAILABLE = False
from PyQt5.QtCore import Qt, QTimer, QSignalBlocker, pyqtSignal, QEvent, QSize
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QPushButton,
    QLabel, QComboBox, QDoubleSpinBox, QSpinBox, QTextEdit, QSplitter,
    QScrollArea, QSizePolicy, QCheckBox, QFileDialog, QDialog, QStackedLayout,
)


MATLAB_BLUE = (0, 114, 189)
MATLAB_ORANGE = (217, 83, 25)
MATLAB_PURPLE = (126, 47, 142)
LIGHT_BG = (250, 250, 250)
AXIS_COLOR = (60, 60, 60)
BORDER_COLOR = (225, 225, 225)




class _PlotGridCell(QWidget):
    """A fixed-behaviour plot cell used by the right-side 2x2 grid.

    PyQtGraph/OpenGL widgets often report a large sizeHint.  If the grid uses
    that hint directly, the left column can become wider than the right column.
    This lightweight wrapper deliberately gives every plot cell the same small
    hint and a near-zero minimum hint, so the QGridLayout row/column stretch
    rules dominate and the four plots stay visually balanced.
    """

    def __init__(self, child: QWidget, parent=None):
        super().__init__(parent)
        self.setObjectName("PlotGridCell")
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMinimumSize(80, 80)
        self.setStyleSheet("QWidget#PlotGridCell { background: #fafafa; border: 1px solid #dedede; border-radius: 7px; }")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(0)
        child.setParent(self)
        child.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        child.setMinimumSize(0, 0)
        layout.addWidget(child)

    def sizeHint(self):
        return QSize(360, 260)

    def minimumSizeHint(self):
        return QSize(80, 80)

class _AlphaBetaSurfaceCanvas(QWidget):
    """Reusable alpha/beta performance view.

    The small embedded view and the standalone presentation window use the same
    renderer, so z-axis metric changes, view-plane changes, and measured-point
    filtering stay logically identical in both places.
    """

    openRequested = pyqtSignal()
    viewModeChanged = pyqtSignal(str)

    def __init__(self, show_open_button: bool = True, large: bool = False, parent=None):
        super().__init__(parent)
        self._large = bool(large)
        self._allow_open_click = bool(show_open_button)
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Expanding if self._large else QSizePolicy.Ignored,
                           QSizePolicy.Expanding if self._large else QSizePolicy.Ignored)
        self._points = []
        self._metric = "evm_average_percent"
        self._metric_label = "EVM平均(%)"
        self._metric_direction = "lower"
        self._best_idx = -1
        self._current_alpha = float("nan")
        self._current_beta = float("nan")
        self._click_press_pos = None
        self._click_press_obj = None
        self._gl_dynamic_items = []
        self._gl_static_items = []
        self._gl_axis_text_items = []
        self._projection_dynamic_items = []
        self._camera_initialized = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3 if not self._large else 6)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(4)
        self.title_label = QLabel("α-β 性能面")
        # 嵌入式四宫格中的标题必须允许收缩；完整说明放入 tooltip。
        self.title_label.setWordWrap(bool(self._large))
        self.title_label.setMinimumWidth(0)
        self.title_label.setMinimumHeight(20 if not self._large else 42)
        self.title_label.setMaximumHeight(26 if not self._large else 84)
        self.title_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.title_label.setStyleSheet("font-weight: 650; color: #2f2f2f; padding-left: 4px;")
        top.addWidget(self.title_label, 1)

        legend_html = (
            '<span style="color:#0072BD; font-weight:800;">■测</span>&nbsp;'
            '<span style="color:#D95319; font-weight:800;">★优</span>&nbsp;'
            '<span style="color:#7E2F8E; font-weight:800;">●现</span>'
            if not self._large else
            '<span style="color:#0072BD; font-weight:800;">■ 已测柱体</span>&nbsp;&nbsp;'
            '<span style="color:#D95319; font-weight:800;">★ 最优点</span>&nbsp;&nbsp;'
            '<span style="color:#7E2F8E; font-weight:800;">● 当前α/β</span>'
        )
        self.legend_label = QLabel(legend_html)
        self.legend_label.setTextFormat(Qt.RichText)
        self.legend_label.setMinimumHeight(18 if not self._large else 24)
        self.legend_label.setMaximumHeight(26 if not self._large else 34)
        self.legend_label.setMaximumWidth(126 if not self._large else 360)
        self.legend_label.setSizePolicy(QSizePolicy.Fixed if not self._large else QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.legend_label.setStyleSheet("padding-right: 2px;")
        top.addWidget(self.legend_label)

        self.view_combo = QComboBox()
        self._add_view_combo_items()
        self.view_combo.setMinimumContentsLength(4 if not self._large else 8)
        self.view_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.view_combo.setMaximumWidth(86 if not self._large else 128)
        self.view_combo.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        view_label = QLabel("视角")
        view_label.setMaximumWidth(30)
        top.addWidget(view_label)
        top.addWidget(self.view_combo)

        self.reset_view_button = QPushButton("↺" if not self._large else "重置视角")
        self.reset_view_button.setToolTip("恢复 MATLAB 风格默认三维视角；独立窗口中可用鼠标滚轮缩放、左键拖拽旋转、右键拖拽平移。")
        self.reset_view_button.setMaximumWidth(34 if not self._large else 86)
        self.reset_view_button.clicked.connect(lambda: self._reset_3d_view(force=True))
        top.addWidget(self.reset_view_button)

        self.open_button = None
        if show_open_button:
            self.open_button = QPushButton("↗")
            self.open_button.setToolTip("单击图面或点击此按钮，在独立窗口中放大展示 α-β 性能面。")
            self.open_button.setMaximumWidth(34)
            self.open_button.clicked.connect(self.openRequested.emit)
            top.addWidget(self.open_button)
        layout.addLayout(top)

        self.stack_holder = QWidget()
        self.stack_holder.setMinimumSize(0, 0)
        self.stack_holder.setSizePolicy(QSizePolicy.Expanding if self._large else QSizePolicy.Ignored,
                                        QSizePolicy.Expanding if self._large else QSizePolicy.Ignored)
        self.stack = QStackedLayout(self.stack_holder)
        self.stack.setContentsMargins(0, 0, 0, 0)
        self.gl_view = None
        self._gl_stack_index = -1
        if _PG_OPENGL_AVAILABLE and gl is not None:
            self.gl_view = gl.GLViewWidget()
            self.gl_view.setSizePolicy(QSizePolicy.Expanding if self._large else QSizePolicy.Ignored,
                                       QSizePolicy.Expanding if self._large else QSizePolicy.Ignored)
            self.gl_view.setMinimumSize(0, 0)
            self.gl_view.setMinimumHeight(96 if not self._large else 540)
            try:
                self.gl_view.setBackgroundColor(LIGHT_BG)
            except Exception:
                pass
            self._init_gl_scene()
            self._gl_stack_index = self.stack.addWidget(self.gl_view)

        self.projection_plot = pg.PlotWidget(title="α-β性能面二维投影")
        self.projection_plot.setSizePolicy(QSizePolicy.Expanding if self._large else QSizePolicy.Ignored,
                                           QSizePolicy.Expanding if self._large else QSizePolicy.Ignored)
        self.projection_plot.setMinimumSize(0, 0)
        self.projection_plot.setBackground(LIGHT_BG)
        self.projection_plot.showGrid(x=True, y=True, alpha=0.35)
        for axis_name in ("left", "bottom"):
            axis = self.projection_plot.getAxis(axis_name)
            axis.setPen(pg.mkPen(AXIS_COLOR, width=1.0))
            axis.setTextPen(pg.mkPen(AXIS_COLOR, width=1.0))
        try:
            self.projection_plot.getPlotItem().getViewBox().setBorder(pg.mkPen(BORDER_COLOR, width=1.0))
        except Exception:
            pass
        self.projection_plot.setMinimumHeight(96 if not self._large else 540)
        self.projection_scatter = pg.ScatterPlotItem(size=13 if not self._large else 18,
                                                     pen=pg.mkPen(AXIS_COLOR, width=1),
                                                     brush=pg.mkBrush(0, 114, 189, 150))
        self.projection_best = pg.ScatterPlotItem(size=18 if not self._large else 26,
                                                  symbol="star",
                                                  pen=pg.mkPen(MATLAB_ORANGE, width=2),
                                                  brush=pg.mkBrush(217, 83, 25, 190))
        self.projection_current = pg.ScatterPlotItem(size=18 if not self._large else 26,
                                                     symbol="o",
                                                     pen=pg.mkPen(MATLAB_PURPLE, width=2.4),
                                                     brush=pg.mkBrush(126, 47, 142, 55))
        self.projection_plot.addItem(self.projection_scatter)
        self.projection_plot.addItem(self.projection_best)
        self.projection_plot.addItem(self.projection_current)
        self._projection_stack_index = self.stack.addWidget(self.projection_plot)
        layout.addWidget(self.stack_holder, 1)

        self.axis_label = QLabel("x=α，y=β，z=平均性能指标；未测点留空。")
        self.axis_label.setWordWrap(bool(self._large))
        self.axis_label.setMinimumWidth(0)
        self.axis_label.setMinimumHeight(16 if not self._large else 28)
        self.axis_label.setMaximumHeight(24 if not self._large else 64)
        self.axis_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.axis_label.setStyleSheet("color: #555555; padding-left: 4px;")
        layout.addWidget(self.axis_label)

        self.view_combo.currentIndexChanged.connect(self._on_view_combo_changed)
        self._install_click_sources()
        self._apply_view_mode(self.view_mode(), emit_signal=False)

    def _add_view_combo_items(self):
        if _PG_OPENGL_AVAILABLE and gl is not None:
            self.view_combo.addItem("三维", "3d")
        self.view_combo.addItem("XY俯视", "xy")
        self.view_combo.addItem("XZ平面", "xz")
        self.view_combo.addItem("YZ平面", "yz")

    def _install_click_sources(self):
        self._click_sources = []
        for obj in (self.title_label, self.axis_label, self.gl_view, self.projection_plot):
            if obj is not None:
                obj.installEventFilter(self)
                self._click_sources.append(obj)
                if self._allow_open_click:
                    try:
                        obj.setCursor(Qt.PointingHandCursor)
                    except Exception:
                        pass
        for plot_obj in (self.projection_plot, self.gl_view):
            try:
                vp = plot_obj.viewport()
                vp.installEventFilter(self)
                self._click_sources.append(vp)
                if self._allow_open_click:
                    vp.setCursor(Qt.PointingHandCursor)
            except Exception:
                pass

    def eventFilter(self, obj, event):
        if self._allow_open_click and obj in getattr(self, "_click_sources", []):
            et = event.type()
            if et == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                try:
                    self._click_press_pos = event.globalPos()
                    self._click_press_obj = obj
                except Exception:
                    self._click_press_pos = None
                    self._click_press_obj = None
            elif et == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                try:
                    if self._click_press_obj is obj and self._click_press_pos is not None:
                        delta = event.globalPos() - self._click_press_pos
                        if abs(delta.x()) + abs(delta.y()) <= 5:
                            self.openRequested.emit()
                except Exception:
                    pass
                finally:
                    self._click_press_pos = None
                    self._click_press_obj = None
            elif et == QEvent.MouseButtonDblClick and event.button() == Qt.LeftButton:
                self.openRequested.emit()
        return super().eventFilter(obj, event)

    @staticmethod
    def _rgba(rgb, alpha=1.0):
        return (float(rgb[0]) / 255.0, float(rgb[1]) / 255.0, float(rgb[2]) / 255.0, float(alpha))

    def _short_title(self, text: str) -> str:
        if self._large:
            return str(text)
        pts = len(getattr(self, "_points", []) or [])
        return f"α-β性能面  z={self._metric_label}  点={pts}"

    def _set_title_text(self, text: str):
        full = str(text or "")
        self.title_label.setText(self._short_title(full))
        self.title_label.setToolTip(full)

    def _set_axis_text(self, text: str):
        full = str(text or "")
        if self._large:
            shown = full
        else:
            shown = f"x=α，y=β，z={self._metric_label}；冻结点不再改写，未测点留空。"
        self.axis_label.setText(shown)
        self.axis_label.setToolTip(full)

    def _reset_3d_view(self, force: bool = True):
        if self.gl_view is None or gl is None:
            return
        if (not force) and bool(getattr(self, "_camera_initialized", False)):
            return
        try:
            # MATLAB-like default: x/y grid is not stretched, bars are readable,
            # and the distance leaves margin for the independent presentation window.
            self.gl_view.setCameraPosition(distance=7.6 if not self._large else 7.2,
                                           elevation=28.0, azimuth=-45.0)
            try:
                self.gl_view.opts["fov"] = 55.0
            except Exception:
                pass
        except Exception:
            try:
                self.gl_view.opts["distance"] = 7.6 if not self._large else 7.2
                self.gl_view.opts["elevation"] = 28.0
                self.gl_view.opts["azimuth"] = -45.0
                self.gl_view.opts["fov"] = 55.0
            except Exception:
                pass
        self._camera_initialized = True

    def _init_gl_scene(self):
        if self.gl_view is None or gl is None:
            return
        self._reset_3d_view(force=True)
        self._add_gl_grids()
        self._add_gl_axes()
        self._set_gl_axis_labels("z: 指标")

    def _add_gl_grids(self):
        if self.gl_view is None or gl is None:
            return
        grid_specs = [
            ("xy", (4.4, 4.4, 1.0), (0.5, 0.5, 1.0), None, (0.0, 0.0, 0.0)),
            ("xz", (4.4, 2.4, 1.0), (0.5, 0.4, 1.0), (90.0, 1.0, 0.0, 0.0), (0.0, -2.2, 1.2)),
            ("yz", (4.4, 2.4, 1.0), (0.5, 0.4, 1.0), (90.0, 0.0, 1.0, 0.0), (-2.2, 0.0, 1.2)),
        ]
        for _name, size, spacing, rotation, translation in grid_specs:
            try:
                item = gl.GLGridItem()
                item.setSize(*size)
                item.setSpacing(*spacing)
                if rotation is not None:
                    item.rotate(*rotation)
                if translation is not None:
                    item.translate(*translation)
                item.setDepthValue(20)
                self.gl_view.addItem(item)
                self._gl_static_items.append(item)
            except Exception:
                pass

    def _add_gl_axes(self):
        if self.gl_view is None or gl is None:
            return
        axis_specs = [
            (np.array([[-2.25, -2.25, 0.0], [2.25, -2.25, 0.0]], dtype=np.float32), MATLAB_BLUE),
            (np.array([[-2.25, -2.25, 0.0], [-2.25, 2.25, 0.0]], dtype=np.float32), MATLAB_ORANGE),
            (np.array([[-2.25, -2.25, 0.0], [-2.25, -2.25, 2.55]], dtype=np.float32), MATLAB_PURPLE),
        ]
        for pts, color in axis_specs:
            try:
                item = gl.GLLinePlotItem(pos=pts, color=self._rgba(color, 1.0), width=2.5, antialias=True)
                self.gl_view.addItem(item)
                self._gl_static_items.append(item)
            except Exception:
                pass

    def _set_gl_axis_labels(self, z_label: str):
        if self.gl_view is None or gl is None:
            return
        for item in list(getattr(self, "_gl_axis_text_items", [])):
            try:
                self.gl_view.removeItem(item)
            except Exception:
                pass
        self._gl_axis_text_items = []
        if not hasattr(gl, "GLTextItem"):
            return
        specs = [
            ((2.45, -2.25, 0.0), "x: α"),
            ((-2.25, 2.45, 0.0), "y: β"),
            ((-2.25, -2.25, 2.72), str(z_label)),
        ]
        for pos, text in specs:
            try:
                item = gl.GLTextItem(pos=pos, text=text, color=self._rgba(AXIS_COLOR, 1.0))
            except Exception:
                try:
                    item = gl.GLTextItem(pos=pos, text=text)
                except Exception:
                    item = None
            if item is not None:
                try:
                    self.gl_view.addItem(item)
                    self._gl_axis_text_items.append(item)
                except Exception:
                    pass

    def view_mode(self) -> str:
        data = self.view_combo.currentData()
        return str(data if data is not None else "3d")

    def set_view_mode(self, mode: str):
        mode = str(mode or "3d")
        for i in range(self.view_combo.count()):
            if str(self.view_combo.itemData(i)) == mode:
                if self.view_combo.currentIndex() != i:
                    self.view_combo.setCurrentIndex(i)
                else:
                    self._apply_view_mode(mode, emit_signal=False)
                return
        # OpenGL unavailable: a request for 3D falls back to XY rather than failing.
        if self.view_combo.count() > 0:
            self.view_combo.setCurrentIndex(0)

    def _on_view_combo_changed(self, *_args):
        self._apply_view_mode(self.view_mode(), emit_signal=True)

    def _apply_view_mode(self, mode: str, emit_signal: bool = True):
        mode = str(mode or "3d")
        if mode == "3d" and self.gl_view is not None and self._gl_stack_index >= 0:
            self.stack.setCurrentIndex(self._gl_stack_index)
            # Do not reset the camera on every refresh.  Otherwise mouse-wheel
            # zoom and drag rotation appear to be broken while the link is running.
            self._reset_3d_view(force=False)
            self._render_gl()
        else:
            self.stack.setCurrentIndex(self._projection_stack_index)
            self._render_projection()
        if emit_signal:
            self.viewModeChanged.emit(mode)

    def clear(self, title_text: str = None, axis_text: str = None):
        self._points = []
        self._best_idx = -1
        self._current_alpha = float("nan")
        self._current_beta = float("nan")
        title = title_text or "α-β 性能三维网格柱状图：等待实测帧"
        axis = axis_text or "x=α，y=β，z=所选实测性能指标；未测量的 α/β 不会被补值。"
        self._set_title_text(title)
        self._set_axis_text(axis)
        self._clear_gl_dynamic_items()
        self._clear_projection_dynamic_items()
        self.projection_scatter.setData([])
        self.projection_best.setData([])
        self.projection_current.setData([])
        self._set_gl_axis_labels(f"z: {self._metric_label}")

    def update_surface(self, points, metric: str, metric_label: str, direction: str,
                       best_idx: int, current_alpha: float, current_beta: float,
                       title_text: str, axis_text: str):
        self._points = list(points or [])
        self._metric = str(metric or "evm_average_percent")
        self._metric_label = str(metric_label or self._metric)
        self._metric_direction = str(direction or "lower")
        self._best_idx = int(best_idx) if best_idx is not None else -1
        try:
            self._current_alpha = float(current_alpha)
            self._current_beta = float(current_beta)
        except Exception:
            self._current_alpha = float("nan")
            self._current_beta = float("nan")
        title = str(title_text)
        axis = str(axis_text)
        self._set_title_text(title)
        self._set_axis_text(axis)
        self._set_gl_axis_labels(f"z: {self._metric_label}")
        if self.view_mode() == "3d" and self.gl_view is not None:
            self._render_gl()
        else:
            self._render_projection()

    @staticmethod
    def _surface_bar_width(values, default=0.18):
        vals = np.unique(np.round(np.asarray(values, dtype=np.float64), 6))
        vals = vals[np.isfinite(vals)]
        if vals.size >= 2:
            diffs = np.diff(np.sort(vals))
            diffs = diffs[diffs > 1e-9]
            if diffs.size:
                return float(max(0.06, min(0.45, 0.72 * float(np.min(diffs)))))
        return float(default)

    @staticmethod
    def _surface_display_heights(z_values):
        z = np.asarray(z_values, dtype=np.float64)
        z = np.where(np.isfinite(z) & (z > 0.0), z, 0.0)
        zmax = float(np.max(z)) if z.size else 0.0
        if zmax <= 1e-15:
            return np.zeros_like(z), 1.0
        return (2.4 * z / zmax).astype(np.float64), zmax / 2.4

    def _clear_gl_dynamic_items(self):
        if self.gl_view is None:
            return
        for item in list(getattr(self, "_gl_dynamic_items", [])):
            try:
                self.gl_view.removeItem(item)
            except Exception:
                pass
        self._gl_dynamic_items = []

    def _make_manual_gl_bars(self, pos, size, color):
        if gl is None:
            return None
        pos = np.asarray(pos, dtype=np.float32).reshape(-1, 3)
        size = np.asarray(size, dtype=np.float32).reshape(-1, 3)
        if pos.size == 0 or size.size == 0:
            return None
        vertices = []
        faces = []
        tri_faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3],
            [3, 7, 4], [3, 4, 0],
        ], dtype=np.int32)
        for p0, sz in zip(pos, size):
            x0, y0, z0 = map(float, p0)
            dx, dy, dz = map(float, sz)
            dz = max(dz, 1e-5)
            base = len(vertices)
            vertices.extend([
                [x0, y0, z0], [x0 + dx, y0, z0], [x0 + dx, y0 + dy, z0], [x0, y0 + dy, z0],
                [x0, y0, z0 + dz], [x0 + dx, y0, z0 + dz], [x0 + dx, y0 + dy, z0 + dz], [x0, y0 + dy, z0 + dz],
            ])
            faces.extend((tri_faces + base).tolist())
        vertices = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int32)
        try:
            return gl.GLMeshItem(vertexes=vertices, faces=faces, color=color, smooth=False, shader="shaded", glOptions="opaque")
        except Exception:
            try:
                return gl.GLMeshItem(vertexes=vertices, faces=faces, color=color, smooth=False)
            except Exception:
                return None

    def _render_gl(self):
        if self.gl_view is None or gl is None:
            return
        self._clear_gl_dynamic_items()
        points = list(self._points)
        if not points:
            return
        x = np.asarray([p["alpha"] for p in points], dtype=np.float64)
        y = np.asarray([p["beta"] for p in points], dtype=np.float64)
        z = np.asarray([p["z"] for p in points], dtype=np.float64)
        h, _scale = self._surface_display_heights(z)
        # Use one common footprint width so α and β cells have equal grid length
        # in the rendered body.  This avoids a visually stretched x/y lattice.
        step_x = self._surface_bar_width(x)
        step_y = self._surface_bar_width(y)
        footprint = float(max(0.06, min(step_x, step_y, 0.38)))
        dx = dy = footprint
        pos = np.column_stack((x - dx / 2.0, y - dy / 2.0, np.zeros_like(h))).astype(np.float32)
        size = np.column_stack((np.full_like(h, dx), np.full_like(h, dy), h)).astype(np.float32)
        # GLBarGraphItem behaves differently across pyqtgraph/OpenGL versions and
        # was the source of several “standalone 3D does not display correctly”
        # symptoms.  A single shaded GLMeshItem is deterministic and works in
        # both the embedded view and the independent presentation dialog.
        bars = self._make_manual_gl_bars(pos, size, self._rgba(MATLAB_BLUE, 0.92))
        if bars is not None:
            self.gl_view.addItem(bars)
            self._gl_dynamic_items.append(bars)
        else:
            try:
                bars = gl.GLBarGraphItem(pos=pos, size=size, color=self._rgba(MATLAB_BLUE, 0.92))
                self.gl_view.addItem(bars)
                self._gl_dynamic_items.append(bars)
            except Exception:
                pass
        marker_z = np.maximum(h, 0.03) + 0.05
        try:
            caps = gl.GLScatterPlotItem(
                pos=np.column_stack((x, y, marker_z)).astype(np.float32),
                size=5.5 if not self._large else 7.5,
                color=self._rgba(MATLAB_BLUE, 0.95), pxMode=True
            )
            self.gl_view.addItem(caps)
            self._gl_dynamic_items.append(caps)
        except Exception:
            pass
        if 0 <= int(self._best_idx) < len(points):
            try:
                best_marker = gl.GLScatterPlotItem(
                    pos=np.array([[x[self._best_idx], y[self._best_idx], marker_z[self._best_idx] + 0.10]], dtype=np.float32),
                    size=13.0 if not self._large else 18.0,
                    color=self._rgba(MATLAB_ORANGE, 1.0), pxMode=True
                )
                self.gl_view.addItem(best_marker)
                self._gl_dynamic_items.append(best_marker)
            except Exception:
                pass
        if np.isfinite(self._current_alpha) and np.isfinite(self._current_beta):
            current_height = 0.08
            dist = np.abs(x - self._current_alpha) + np.abs(y - self._current_beta)
            if dist.size and float(np.min(dist)) < 1e-6:
                current_height = float(marker_z[int(np.argmin(dist))] + 0.18)
            try:
                cur_marker = gl.GLScatterPlotItem(
                    pos=np.array([[self._current_alpha, self._current_beta, current_height]], dtype=np.float32),
                    size=10.0 if not self._large else 15.0,
                    color=self._rgba(MATLAB_PURPLE, 1.0), pxMode=True
                )
                self.gl_view.addItem(cur_marker)
                self._gl_dynamic_items.append(cur_marker)
            except Exception:
                pass
        # Camera is intentionally not reset here.  Runtime refresh must preserve
        # the user's current zoom/rotation/pan.

    def _clear_projection_dynamic_items(self):
        for item in list(getattr(self, "_projection_dynamic_items", [])):
            try:
                self.projection_plot.removeItem(item)
            except Exception:
                pass
        self._projection_dynamic_items = []

    def _projection_normalized_spots(self, x, y, z):
        zmin = float(np.nanmin(z)) if z.size else 0.0
        zmax = float(np.nanmax(z)) if z.size else 1.0
        denom = max(zmax - zmin, 1e-12)
        norm = (z - zmin) / denom
        spots = []
        for xi, yi, ni in zip(x, y, norm):
            alpha = int(80 + 150 * float(ni))
            spots.append({"pos": (float(xi), float(yi)),
                          "brush": pg.mkBrush(0, 114, 189, alpha),
                          "size": 14 if not self._large else 20})
        return spots

    def _render_projection(self):
        self._clear_projection_dynamic_items()
        points = list(self._points)
        if not points:
            self.projection_scatter.setData([])
            self.projection_best.setData([])
            self.projection_current.setData([])
            mode = self.view_mode()
            self.projection_plot.setTitle(f"α-β性能面二维投影[{mode}]：等待实测点")
            return
        x = np.asarray([p["alpha"] for p in points], dtype=np.float64)
        y = np.asarray([p["beta"] for p in points], dtype=np.float64)
        z = np.asarray([p["z"] for p in points], dtype=np.float64)
        mode = self.view_mode()
        if mode == "xy" or (mode == "3d" and self.gl_view is None):
            self.projection_plot.setLabel("bottom", "x: α")
            self.projection_plot.setLabel("left", "y: β")
            self.projection_plot.setTitle(f"XY俯视：颜色/透明度表示 z={self._metric_label}")
            self.projection_scatter.setData(self._projection_normalized_spots(x, y, z))
            if 0 <= int(self._best_idx) < len(points):
                self.projection_best.setData([{"pos": (float(x[self._best_idx]), float(y[self._best_idx]))}])
            else:
                self.projection_best.setData([])
            if np.isfinite(self._current_alpha) and np.isfinite(self._current_beta):
                self.projection_current.setData([{"pos": (float(self._current_alpha), float(self._current_beta))}])
            else:
                self.projection_current.setData([])
            self.projection_plot.setXRange(-2.1, 2.1, padding=0)
            self.projection_plot.setYRange(-2.1, 2.1, padding=0)
            return

        axis_values = x if mode == "xz" else y
        axis_name = "x: α" if mode == "xz" else "y: β"
        self.projection_plot.setLabel("bottom", axis_name)
        self.projection_plot.setLabel("left", f"z: {self._metric_label}")
        self.projection_plot.setTitle(f"{mode.upper()} 平面：二维展示 {axis_name} 与 z={self._metric_label}")
        width = self._surface_bar_width(axis_values, default=0.12)
        z = np.where(np.isfinite(z), z, np.nan)
        finite = np.isfinite(axis_values) & np.isfinite(z)
        if not np.any(finite):
            self.projection_scatter.setData([])
            self.projection_best.setData([])
            self.projection_current.setData([])
            return
        xv = axis_values[finite]
        zv = z[finite]
        y0 = 0.0 if float(np.nanmin(zv)) >= 0.0 else float(np.nanmin(zv))
        try:
            bar = pg.BarGraphItem(x=xv, height=zv - y0, width=width, y0=y0,
                                  brush=pg.mkBrush(0, 114, 189, 120),
                                  pen=pg.mkPen(MATLAB_BLUE, width=1))
            self.projection_plot.addItem(bar)
            self._projection_dynamic_items.append(bar)
        except Exception:
            pass
        spots = [{"pos": (float(xi), float(zi)),
                  "brush": pg.mkBrush(0, 114, 189, 180),
                  "size": 10 if not self._large else 15}
                 for xi, zi in zip(xv, zv)]
        self.projection_scatter.setData(spots)
        if 0 <= int(self._best_idx) < len(points):
            bx = float(axis_values[self._best_idx])
            bz = float(z[self._best_idx])
            if np.isfinite(bx) and np.isfinite(bz):
                self.projection_best.setData([{"pos": (bx, bz)}])
            else:
                self.projection_best.setData([])
        else:
            self.projection_best.setData([])
        if np.isfinite(self._current_alpha) and np.isfinite(self._current_beta):
            cx = self._current_alpha if mode == "xz" else self._current_beta
            cz = y0
            dist = np.abs(x - self._current_alpha) + np.abs(y - self._current_beta)
            if dist.size and float(np.min(dist)) < 1e-6:
                cz = float(z[int(np.argmin(dist))])
            self.projection_current.setData([{"pos": (float(cx), float(cz))}])
        else:
            self.projection_current.setData([])
        xmin = float(np.nanmin(xv)); xmax = float(np.nanmax(xv))
        if abs(xmax - xmin) < 1e-9:
            xmin -= 0.5; xmax += 0.5
        zmin = float(np.nanmin(zv)); zmax = float(np.nanmax(zv))
        if abs(zmax - zmin) < 1e-15:
            pad = max(abs(zmax) * 0.2, 1e-3)
            zmin -= pad; zmax += pad
        else:
            pad = 0.12 * (zmax - zmin)
            zmin -= pad; zmax += pad
        self.projection_plot.setXRange(xmin - width, xmax + width, padding=0)
        self.projection_plot.setYRange(zmin, zmax, padding=0)
