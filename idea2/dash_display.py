import sys
import calendar
from math import ceil
from PyQt6.QtWidgets import QApplication, QWidget, QGridLayout, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt
from manager import TaskManager
from dash import deadline_heatmap, completed_heatmap

DEADLINE_COLORS = [
    "#26344a",  # 0
    "#ffd166",  # 1
    "#ffb347",  # 2
    "#ff8c42",  # 3
    "#ff5d5d",  # 4
    "#ff477e",  # 5
    "#f72585",  # 6
    "#d61c7a",  # 7
    "#b5179e",  # 8
    "#7209b7",  # 9
    "#560bad",  # 10
]

COMPLETED_COLORS = [
    "#26344a",  # 0
    "#b8f2e6",  # 1
    "#7ae582",  # 2
    "#4cd97b",  # 3
    "#2dd881",  # 4
    "#00c853",  # 5
    "#00bfa5",  # 6
    "#00a896",  # 7
    "#008f8c",  # 8
    "#007f7b",  # 9
    "#006d77",  # 10
]


def make_label(text: str, bold: bool = False) -> QLabel:
    label = QLabel(text)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    if bold:
        label.setStyleSheet("font-weight: bold;")
    return label


class HeatmapDeadline(QWidget):
    def __init__(self, tasks, year: int, month: int):
        super().__init__()
        self.setWindowTitle(f"Deadline {month}/{year}")

        counts = deadline_heatmap(tasks, year, month)
        max_count = int(counts.max()) if counts.size else 0

        layout = QVBoxLayout(self)
        layout.addWidget(make_label(f"Tháng {month}/{year}", bold=True))

        grid = QGridLayout()
        layout.addLayout(grid)

        # tiêu đề thứ, start: thứ 2
        for col, name in enumerate(["T2", "T3", "T4", "T5", "T6", "T7", "CN"]):
            grid.addWidget(make_label(name, bold=True), 0, col)

        for row, week in enumerate(calendar.monthcalendar(year, month), start=1):
            for col, day in enumerate(week):
                if day == 0:
                    continue

                n = int(counts[day - 1])
                level = min(n, 4)
                text_color = "#1e1e2e" if level != 0 else "#cdd6f4"

                cell = make_label(str(day))
                cell.setMinimumSize(20, 20)
                cell.setStyleSheet(
                    f"background-color: {DEADLINE_COLORS[level]};"
                    f"color: {text_color}; border-radius: 3px;"
                    )
                cell.setToolTip(f"{n} deadline")
                grid.addWidget(cell, row, col)

class HeatmapCompleted(QWidget):
    def __init__(self, tasks, year: int, month: int):
        super().__init__()
        self.setWindowTitle(f"Completed {month}/{year}")

        counts = completed_heatmap(tasks, year, month)
        max_count = int(counts.max()) if counts.size else 0

        layout = QVBoxLayout(self)
        layout.addWidget(make_label(f"Tháng {month}/{year}", bold=True))

        grid = QGridLayout()
        layout.addLayout(grid)

        for col, name in enumerate(["T2", "T3", "T4", "T5", "T6", "T7", "CN"]):
            grid.addWidget(make_label(name, bold=True), 0, col)

        # Mỗi tuần là một hàng, số 0 là ô trống của ngày thuộc tháng khác
        for row, week in enumerate(calendar.monthcalendar(year, month), start=1):
            for col, day in enumerate(week):
                if day == 0:
                    continue

                n = int(counts[day - 1])
                level = min(n, 4)
                text_color = "#1e1e2e" if level != 0 else "#cdd6f4"

                cell = make_label(str(day))
                cell.setMinimumSize(20, 20)
                cell.setStyleSheet(
                    f"background-color: {COMPLETED_COLORS[level]};"
                    f"color: {text_color}; border-radius: 4px;"
                    )
                cell.setToolTip(f"{n} completed")
                grid.addWidget(cell, row, col)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    m = TaskManager()
    m.load_from_csv("C:/Users/vuong/school/2026-2027/editeh/idea2/csvs/tst.csv")

    view = HeatmapDeadline(m.get_all_task(), 2026, 11)
    view.setFixedSize(200,200)
    view.show()
    sys.exit(app.exec())
