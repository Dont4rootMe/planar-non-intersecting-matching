"""Left-side control panel for the planar matching app."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..engine import Engine, EngineError


class ControlPanel(QWidget):
    """Panel with file operations, matching trigger, and active set chooser."""

    def __init__(self, engine: Engine, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._engine = engine

        self._counts_label = QLabel("A: 0    B: 0", self)
        self._matching_label = QLabel("Паросочетание: отсутствует", self)

        self._btn_a = QPushButton("A (крестик)", self)
        self._btn_b = QPushButton("B (кружок)", self)
        self._btn_a.setCheckable(True)
        self._btn_b.setCheckable(True)
        self._btn_a.setChecked(True)

        self._load_button = QPushButton("Загрузить точки…", self)
        self._save_button = QPushButton("Сохранить точки…", self)
        self._clear_button = QPushButton("Очистить доску", self)
        self._compute_button = QPushButton("Вычислить паросочетание", self)

        self._notice_label = QLabel(
            "ЛКМ на правой доске добавляет точку в активное множество.\n"
            "Для расчёта паросочетания множества A и B должны быть одинакового размера.",
            self,
        )
        self._notice_label.setWordWrap(True)

        self._build_layout()
        self._connect_signals()

        self._engine.add_refresh_action(self._on_engine_refresh)
        self._engine.add_reset_action(self._on_engine_refresh)
        self._engine.set_active_set(0)
        self._on_engine_refresh()

    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        active_group = QGroupBox("Активное множество", self)
        active_layout = QHBoxLayout()
        active_layout.setSpacing(6)
        active_layout.addWidget(self._btn_a)
        active_layout.addWidget(self._btn_b)
        active_group.setLayout(active_layout)

        layout = QVBoxLayout()
        layout.addWidget(self._counts_label)
        layout.addWidget(active_group)
        layout.addWidget(self._notice_label)
        layout.addWidget(self._compute_button)
        layout.addWidget(self._matching_label)
        layout.addSpacing(10)
        layout.addWidget(self._clear_button)
        layout.addWidget(self._load_button)
        layout.addWidget(self._save_button)
        layout.addStretch()
        self.setLayout(layout)

    def _connect_signals(self) -> None:
        self._btn_a.clicked.connect(lambda: self._handle_set_toggle(0))
        self._btn_b.clicked.connect(lambda: self._handle_set_toggle(1))
        self._load_button.clicked.connect(self._handle_load)
        self._save_button.clicked.connect(self._handle_save)
        self._clear_button.clicked.connect(self._handle_clear)
        self._compute_button.clicked.connect(self._handle_compute)

    # ------------------------------------------------------------------
    def _on_engine_refresh(self) -> None:
        count_a, count_b = self._engine.get_point_counts()
        self._counts_label.setText(f"A: {count_a}    B: {count_b}")
        if self._engine.has_valid_matching():
            self._matching_label.setText("Паросочетание: готово")
        else:
            self._matching_label.setText("Паросочетание: отсутствует")

    def _handle_set_toggle(self, kind: int) -> None:
        try:
            self._engine.set_active_set(kind)
        except ValueError as exc:
            QMessageBox.warning(self, "Ошибка", str(exc))
            return

        # Keep buttons mutually exclusive manually
        if kind == 0:
            self._btn_a.setChecked(True)
            self._btn_b.setChecked(False)
        else:
            self._btn_a.setChecked(False)
            self._btn_b.setChecked(True)

    def _handle_clear(self) -> None:
        self._engine.clear_board()

    def _handle_compute(self) -> None:
        try:
            self._engine.compute_planar_matching()
        except EngineError as exc:
            QMessageBox.warning(self, "Невозможно вычислить", str(exc))

    def _handle_load(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Открыть точки",
            "",
            "Text files (*.txt);;All files (*)",
        )
        if not file_path:
            return
        try:
            self._engine.load_points_from_file(file_path)
        except ValueError as exc:
            QMessageBox.critical(self, "Некорректные данные", str(exc))
        except EngineError as exc:
            QMessageBox.critical(self, "Ошибка загрузки", str(exc))

    def _handle_save(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить точки",
            "points.txt",
            "Text files (*.txt);;All files (*)",
        )
        if not file_path:
            return

        path_obj = Path(file_path)
        if not path_obj.suffix:
            path_obj = path_obj.with_suffix(".txt")

        try:
            self._engine.save_points_to_file(str(path_obj))
        except EngineError as exc:
            QMessageBox.critical(self, "Ошибка сохранения", str(exc))
