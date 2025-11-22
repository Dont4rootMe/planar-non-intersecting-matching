"""Control panel widget providing file and computation actions."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from engine import Engine, EngineError
from examples.centric_circles import get_centric_circles
from examples.geometric_composition import get_nested_polygons, get_pinwheel, get_spirals
from examples.randomized_circles import get_randomized_circles
from examples.swiss_roll import get_swiss_roll
from examples.triple_clusters import get_triple_clusters
from examples.two_moons import get_two_moons


class ControlPanel(QWidget):
    """Vertical panel with controls for managing points and lines."""

    def __init__(self, engine: Engine, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._engine = engine

        self._points_label = QLabel("number of points: 0", self)
        self._points_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self._reset_button = QPushButton("Clear board", self)
        self._load_button = QPushButton("Load points from file…", self)
        self._compute_button = QPushButton("Compute levels", self)
        self._save_button = QPushButton("Save point selection…", self)

        self._samples_label = QLabel(self)
        self._noise_label = QLabel(self)
        self._layers_label = QLabel("Layers: 0", self)
        self._points_per_layer_title = QLabel("Points per layer:", self)
        self._points_per_layer_title.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._points_per_layer_container = QWidget(self)
        self._points_per_layer_layout = QVBoxLayout()
        self._points_per_layer_layout.setContentsMargins(0, 0, 0, 0)
        self._points_per_layer_layout.setSpacing(2)
        self._points_per_layer_container.setLayout(self._points_per_layer_layout)
        self._points_per_layer_area = QScrollArea(self)
        self._points_per_layer_area.setWidgetResizable(True)
        self._points_per_layer_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._points_per_layer_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._points_per_layer_area.setWidget(self._points_per_layer_container)
        self._show_placeholder_points_per_layer()
        self._duration_label = QLabel("Computation time: 0.00 ms", self)

        self._samples_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._samples_slider.setRange(250, 5000)
        self._samples_slider.setSingleStep(50)
        self._samples_slider.setPageStep(250)
        self._samples_slider.setValue(500)

        self._noise_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._noise_slider.setRange(0, 100)
        self._noise_slider.setSingleStep(1)
        self._noise_slider.setPageStep(5)
        self._noise_slider.setValue(10)

        self._preset_group = self._create_preset_group()

        self._build_layout()
        self._connect_signals()
        self._engine.register_points_listener(self._on_points_changed)
        self._engine.register_layers_info_listener(self._on_layers_info_changed)
        self._update_samples_label(self._samples_slider.value())
        self._update_noise_label(self._noise_slider.value())

    def _build_layout(self) -> None:
        layout = QVBoxLayout()
        layout.addWidget(self._reset_button)
        layout.addWidget(self._load_button)
        layout.addWidget(self._points_label)
        layout.addWidget(self._preset_group)
        layout.addWidget(self._samples_label)
        layout.addWidget(self._samples_slider)
        layout.addWidget(self._noise_label)
        layout.addWidget(self._noise_slider)
        layout.addStretch()
        layout.addWidget(self._layers_label)
        layout.addWidget(self._points_per_layer_title)
        layout.addWidget(self._points_per_layer_area)
        layout.addWidget(self._duration_label)
        layout.addWidget(self._compute_button)
        layout.addWidget(self._save_button)
        self.setLayout(layout)

    def _connect_signals(self) -> None:
        self._reset_button.clicked.connect(self._handle_reset)
        self._load_button.clicked.connect(self._handle_load)
        self._compute_button.clicked.connect(self._handle_compute)
        self._save_button.clicked.connect(self._handle_save)
        self._samples_slider.valueChanged.connect(self._update_samples_label)
        self._noise_slider.valueChanged.connect(self._update_noise_label)

    # ------------------------------------------------------------------
    def _create_preset_group(self) -> QGroupBox:
        presets: list[tuple[str, Callable[..., np.ndarray]]] = [
            ("Concentric circles", get_centric_circles),
            ("Randomized circles", get_randomized_circles),
            ("Two moons", get_two_moons),
            ("Swiss roll", get_swiss_roll),
            ("Triple clusters", get_triple_clusters),
            ("Spirals", get_spirals),
            ("Nested polygons", get_nested_polygons),
            ("Pinwheel", get_pinwheel),
        ]

        group = QGroupBox("Default point sets", self)
        grid = QGridLayout()
        grid.setSpacing(6)

        for index, (title, generator) in enumerate(presets):
            button = QPushButton(title, group)
            button.clicked.connect(lambda _checked=False, gen=generator: self._handle_preset(gen))
            row = index // 2
            column = index % 2
            grid.addWidget(button, row, column)

        group.setLayout(grid)
        return group

    def _update_samples_label(self, value: int) -> None:
        self._samples_label.setText(f"Preset size: {value} points")

    def _update_noise_label(self, value: int) -> None:
        noise_value = value / 100.0
        self._noise_label.setText(f"Noise: {noise_value:.2f}")

    def _on_layers_info_changed(self, info: dict[str, object]) -> None:
        layer_count = int(info.get("layer_count", 0))
        points_per_layer = info.get("points_per_layer", [])
        duration = float(info.get("duration", 0.0))

        self._layers_label.setText(f"Layers: {layer_count}")
        self._update_points_per_layer(points_per_layer)
        self._duration_label.setText(f"Computation time: {duration * 1000:.2f} ms")

    def _handle_reset(self) -> None:
        try:
            self._engine.reset()
        except EngineError as exc:
            self._show_error(str(exc))

    def _handle_load(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load points",
            "",
            "Text files (*.txt);;All files (*)",
        )
        if not file_path:
            return
        try:
            self._engine.load_points(file_path)
        except EngineError as exc:
            self._show_error(str(exc))

    def _handle_compute(self) -> None:
        current_points = self._engine.points()
        if current_points.shape[0] < 2:
            QMessageBox.information(self, "Not enough points", "Add at least two points to calculate levels.")
            return
        try:
            self._engine.compute_levels()
        except EngineError as exc:
            self._show_error(str(exc))

    def _handle_preset(self, generator: Callable[..., np.ndarray]) -> None:
        n_samples = self._samples_slider.value()
        noise = self._noise_slider.value() / 100.0

        try:
            points = generator(n_samples=n_samples, noise=noise)
        except Exception as exc:  # pragma: no cover - safeguard around external code
            self._show_error(f"Failed to generate preset: {exc}")
            return

        if isinstance(points, tuple):
            points = points[0]

        try:
            self._engine.bulk_set_points(points, fit_to_scene=True)
        except EngineError as exc:
            self._show_error(str(exc))

    def _handle_save(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save points",
            "point_set.txt",
            "Text files (*.txt);;All files (*)",
        )
        if not file_path:
            return

        path_obj = Path(file_path)
        if path_obj.exists():
            answer = QMessageBox.question(
                self,
                "Overwrite file?",
                "File already exists. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            self._engine.save_points(file_path, overwrite=True)
        except EngineError as exc:
            self._show_error(str(exc))

    # ------------------------------------------------------------------
    def _on_points_changed(self, points: np.ndarray) -> None:
        self._points_label.setText(f"number of points: {points.shape[0]}")

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Error", message)

    def _update_points_per_layer(self, points_per_layer: object) -> None:
        if (
            isinstance(points_per_layer, Sequence)
            and not isinstance(points_per_layer, (str, bytes))
            and points_per_layer
        ):
            self._populate_points_per_layer(list(points_per_layer))
        else:
            self._show_placeholder_points_per_layer()

    def _populate_points_per_layer(self, points_per_layer: list[object]) -> None:
        self._clear_points_per_layer_entries()

        for index, value in enumerate(points_per_layer, start=1):
            label = QLabel(f"M({index}) = {int(value)}", self._points_per_layer_container)
            label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self._points_per_layer_layout.addWidget(label)

        self._adjust_points_per_layer_height(len(points_per_layer))

    def _show_placeholder_points_per_layer(self) -> None:
        self._clear_points_per_layer_entries()

        placeholder = QLabel("–", self._points_per_layer_container)
        placeholder.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._points_per_layer_layout.addWidget(placeholder)
        self._adjust_points_per_layer_height(1)

    def _clear_points_per_layer_entries(self) -> None:
        while self._points_per_layer_layout.count():
            item = self._points_per_layer_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _adjust_points_per_layer_height(self, item_count: int) -> None:
        metrics = self.fontMetrics()
        row_height = metrics.height() + 6
        visible_rows = min(max(item_count, 1), 10)
        frame_height = self._points_per_layer_area.frameWidth() * 2
        total_height = visible_rows * row_height + frame_height
        self._points_per_layer_area.setFixedHeight(total_height)
        policy = (
            Qt.ScrollBarPolicy.ScrollBarAlwaysOn
            if item_count > 10
            else Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._points_per_layer_area.setVerticalScrollBarPolicy(policy)
