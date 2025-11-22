"""Interactive QGraphicsView displaying points and computed segments."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor, QMouseEvent, QPainter, QPen
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsScene,
    QGraphicsView,
    QMessageBox,
    QWidget,
)

from engine import LINE_WIDTH, POINT_RADIUS, SCENE_SIZE, Engine, EngineError


class Canvas(QGraphicsView):
    """Graphics view responsible for rendering points and lines."""

    def __init__(self, engine: Engine, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._engine = engine
        self._scene = QGraphicsScene(0.0, 0.0, float(SCENE_SIZE), float(SCENE_SIZE), self)
        self._scene.setBackgroundBrush(QBrush(Qt.GlobalColor.white))
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

        self._point_items: list[QGraphicsEllipseItem] = []
        self._line_items: list[QGraphicsLineItem] = []

        self._engine.register_points_listener(self._on_points_changed)
        self._engine.register_lines_listener(self._on_lines_changed)

        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    # ------------------------------------------------------------------
    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            try:
                self._engine.add_point(scene_pos.x(), scene_pos.y())
            except EngineError as exc:
                QMessageBox.critical(self, "Error", str(exc))
                event.accept()
                return
            event.accept()
            return
        super().mousePressEvent(event)

    # ------------------------------------------------------------------
    def _on_points_changed(self, points: np.ndarray) -> None:
        for item in self._point_items:
            self._scene.removeItem(item)
        self._point_items.clear()

        pen = QPen(Qt.GlobalColor.black)
        pen.setWidthF(1.0)
        brush = QBrush(Qt.GlobalColor.black)

        radius = float(POINT_RADIUS)
        diameter = radius * 2.0
        for x, y in points:
            ellipse = self._scene.addEllipse(x - radius, y - radius, diameter, diameter, pen, brush)
            self._point_items.append(ellipse)

    def _on_lines_changed(self, lines: np.ndarray) -> None:
        for item in self._line_items:
            self._scene.removeItem(item)
        self._line_items.clear()

        if lines.size == 0:
            return

        default_pen = QPen(Qt.GlobalColor.red)
        default_pen.setWidthF(float(LINE_WIDTH))

        if lines.shape[1] == 4:
            for x1, y1, x2, y2 in lines:
                line = self._scene.addLine(x1, y1, x2, y2, default_pen)
                self._line_items.append(line)
            return

        if lines.shape[1] != 7:
            raise ValueError("Unexpected line format received from engine.")

        for x1, y1, x2, y2, r, g, b in lines:
            color = QColor.fromRgbF(float(r), float(g), float(b))
            pen = QPen(color)
            pen.setWidthF(float(LINE_WIDTH))
            line = self._scene.addLine(x1, y1, x2, y2, pen)
            self._line_items.append(line)
