import sys
import calendar
from math import ceil
from PyQt6.QtWidgets import QApplication, QWidget, QGridLayout, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt
from manager import TaskManager
from dash import deadline_heatmap

LEVEL_COLORS = [
    "#313244",  # 0
    "#1f3b2d",  # 1
    "#255238",  # 2
    "#2b6943",  # 3
    "#317f4f",  # 4
    "#379a5b",  # 5
    "#3db467",  # 6
    "#46c574",  # 7
    "#56d682",  # 8
    "#6fe293",  # 9
    "#8be9a8",  # 10
]


def make_label(text: str, bold: bool = False) -> QLabel:
    label = QLabel(text)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    if bold:
        label.setStyleSheet("font-weight: bold;")
    return label


class HeatmapView(QWidget):
    def __init__(self, tasks, year: int, month: int):
        super().__init__()
        self.setWindowTitle(f"Deadline {month}/{year}")

        counts = deadline_heatmap(tasks, year, month)
        max_count = int(counts.max()) if counts.size else 0

        layout = QVBoxLayout(self)
        layout.addWidget(make_label(f"Tháng {month}/{year}", bold=True))

        grid = QGridLayout()
        layout.addLayout(grid)

        # Hàng tiêu đề các thứ, monthcalendar bắt đầu từ thứ 2
        for col, name in enumerate(["T2", "T3", "T4", "T5", "T6", "T7", "CN"]):
            grid.addWidget(make_label(name, bold=True), 0, col)

        # Mỗi tuần là một hàng, số 0 là ô trống của ngày thuộc tháng khác
        for row, week in enumerate(calendar.monthcalendar(year, month), start=1):
            for col, day in enumerate(week):
                if day == 0:
                    continue

                n = int(counts[day - 1])
                level = min(n, 4)
                text_color = "#1e1e2e" if level >= 3 else "#cdd6f4"

                cell = make_label(str(day))
                cell.setMinimumSize(20, 20)
                cell.setStyleSheet(
                    f"background-color: {LEVEL_COLORS[level]};"
                    f"color: {text_color}; border-radius: 3px;"
                )
                cell.setToolTip(f"{n} deadline")
                grid.addWidget(cell, row, col)

