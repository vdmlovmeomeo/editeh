from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from PyQt6.QtCore import QDate, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication, QDateEdit, QFileDialog, QFormLayout, QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QStackedWidget, QVBoxLayout, QWidget

from dash_display import HeatmapCompleted, HeatmapDeadline
from manager import TaskManager

from PyQt6.QtGui import QFontDatabase, QFont

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "csvs" / "tst.csv"

class ImportWorker(QThread):
    finished = pyqtSignal(int) # pyright: ignore[reportCallIssue]
    failed = pyqtSignal(str) # pyright: ignore[reportCallIssue] # pyright: ignore[reportCallIssue]
    def __init__(self, manager, image_path):
        super().__init__(); self.manager, self.image_path = manager, image_path
    def run(self):
        try:
            from ai_input import import_from_image
            self.finished.emit(import_from_image(self.manager, self.image_path))
        except Exception as exc:
            self.failed.emit(str(exc))

class TaskEditor(QFrame):
    saved = pyqtSignal() # pyright: ignore[reportCallIssue]
    def __init__(self, manager):
        super().__init__(); self.manager, self.task_id = manager, None; self.setObjectName("card")
        form = QFormLayout(self)
        self.title = QLineEdit(); self.title.setPlaceholderText("eg: go to sleep")
        self.content = QPlainTextEdit(); self.content.setPlaceholderText("eg: lie down, close your eyes, sleep")
        self.deadline = QDateEdit(QDate.currentDate()); self.deadline.setCalendarPopup(True); self.deadline.setDisplayFormat("dd/MM/yyyy")
        form.addRow("title", self.title); form.addRow("body", self.content); form.addRow("deadline", self.deadline)
        buttons = QHBoxLayout(); save = QPushButton("save to csv"); save.clicked.connect(self.save); clear = QPushButton("clear"); clear.clicked.connect(self.clear)
        buttons.addWidget(save); buttons.addWidget(clear); form.addRow(buttons)
    def load_task(self, task):
        self.task_id = task.id; self.title.setText(task.title); self.content.setPlainText(task.content); self.deadline.setDate(QDate(task.deadline.year, task.deadline.month, task.deadline.day))
    def clear(self):
        self.task_id = None; self.title.clear(); self.content.clear(); self.deadline.setDate(QDate.currentDate())
    def save(self):
        if not self.title.text().strip(): QMessageBox.warning(self,"", "no title"); return
        d = self.deadline.date()
        if self.task_id is None: self.manager.add_task(self.title.text(), self.content.toPlainText(), {"dd": d.day(), "mm": d.month(), "yy": d.year()})
        else:
            task = self.manager.find(self.task_id)
            if task: task.title = self.title.text().strip(); self.manager.edit_content(task.id, self.content.toPlainText() or " "); self.manager.edit_deadline(task.id, d.day(), d.month(), d.year())
        self.clear(); self.saved.emit()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.manager = TaskManager(); self.manager.load_from_csv(str(DATA_FILE)); self.setWindowTitle("tba"); self.resize(1120, 720); self.worker = None; self.compact_mode = False
        root = QWidget(); self.setCentralWidget(root); layout = QHBoxLayout(root); layout.setContentsMargins(0, 0, 0, 0)
        sidebar = QFrame(objectName="sidebar"); self.sidebar = sidebar; side = QVBoxLayout(sidebar); brand = QLabel("tba"); brand.setObjectName("brand"); side.addWidget(brand) # pyright: ignore[reportCallIssue]
        self.nav = QListWidget(); self.nav.addItems(["tasks", "dashboard"]); self.nav.currentRowChanged.connect(self.pages_switch); side.addWidget(self.nav); side.addStretch(); layout.addWidget(sidebar)
        self.pages = QStackedWidget(); layout.addWidget(self.pages, 1); self.pages.addWidget(self.build_tasks_page()); self.pages.addWidget(self.build_dashboard_page()); self.nav.setCurrentRow(0); self.setStyleSheet(STYLE); self.refresh_tasks()
    def build_tasks_page(self):
        page = QWidget(); outer = QVBoxLayout(page); header = QHBoxLayout(); heading = QLabel("tasks"); heading.setObjectName("heading"); header.addWidget(heading); header.addStretch()
        add = QPushButton("add task"); add.clicked.connect(self.start_new_task)
        refresh = QPushButton("refresh task"); refresh.clicked.connect(self.reload_csv)
        compact = QPushButton("compact mode"); compact.setCheckable(True); compact.clicked.connect(self.toggle_compact); self.compact_button = compact
        image = QPushButton("import from image"); image.clicked.connect(self.import_image); save = QPushButton("save to csv"); save.clicked.connect(self.save_csv); header.addWidget(add); header.addWidget(refresh); header.addWidget(image); header.addWidget(save); header.addWidget(compact); outer.addLayout(header)
        self.full_header_buttons = (add, refresh, image, save)
        body = QHBoxLayout(); left = QVBoxLayout(); self.summary = QLabel(); self.summary.setObjectName("muted"); left.addWidget(self.summary); self.task_list = QListWidget(); self.task_list.setWordWrap(True); self.task_list.setTextElideMode(Qt.TextElideMode.ElideNone); self.task_list.currentRowChanged.connect(self.select_task); left.addWidget(self.task_list)
        actions = QHBoxLayout()
        for text, slot in (("done", self.toggle_selected), ("del", self.delete_selected)):
            button = QPushButton(text); button.clicked.connect(slot); actions.addWidget(button)
        left.addLayout(actions); body.addLayout(left, 3); self.editor = TaskEditor(self.manager); self.editor.saved.connect(self.refresh_tasks); body.addWidget(self.editor, 2); outer.addLayout(body); return page
    def build_dashboard_page(self):
        page = QWidget(); lay = QVBoxLayout(page); heading = QLabel("dashboard"); heading.setObjectName("heading"); lay.addWidget(heading); info = QLabel(""); info.setObjectName("muted"); lay.addWidget(info); row = QHBoxLayout()
        for text, view_class in (("deadline heatmap", HeatmapDeadline), ("completed heatmap", HeatmapCompleted)):
            button = QPushButton(text); button.clicked.connect(lambda _=False, cls=view_class: self.show_heatmap(cls)); row.addWidget(button)
        lay.addLayout(row); lay.addStretch(); return page
    def pages_switch(self, index): self.pages.setCurrentIndex(index)
    def refresh_tasks(self):
        self.task_list.clear(); tasks = self.manager.get_all_task(); self.summary.setText(f"done: {sum(t.done for t in tasks)}/{len(tasks)}")
        for task in tasks:
            item = QListWidgetItem(f"{'✓' if task.done else '○'}  {task.title}  ·  {task.deadline:%d/%m/%Y}\n    {task.content}"); item.setData(Qt.ItemDataRole.UserRole, task.id); self.task_list.addItem(item)
    def selected_task(self):
        item = self.task_list.currentItem(); return self.manager.find(item.data(Qt.ItemDataRole.UserRole)) if item else None
    def select_task(self, _row):
        task = self.selected_task()
        if task: self.editor.load_task(task)
    def toggle_selected(self):
        task = self.selected_task()
        if task: self.manager.toggle_task(task.id); self.refresh_tasks()
    def delete_selected(self):
        task = self.selected_task()
        if task and QMessageBox.question(self, "del", f"del '{task.title}'?") == QMessageBox.StandardButton.Yes: self.manager.delete_task(task.id); self.refresh_tasks()
    def save_csv(self):
        DATA_FILE.parent.mkdir(exist_ok=True); self.manager.save_to_csv(str(DATA_FILE)); QMessageBox.information(self, "saved", f"saved {len(self.manager.get_all_task())} task.")
    def reload_csv(self):
        self.manager.load_from_csv(str(DATA_FILE)); self.editor.clear(); self.refresh_tasks(); self.statusBar().showMessage("refreshed task from CSV.") # pyright: ignore[reportOptionalMemberAccess]
    def start_new_task(self):
        self.editor.clear()
        self.task_list.clearSelection()
        self.editor.title.setFocus()
        self.statusBar().showMessage("creating new task.") # pyright: ignore[reportOptionalMemberAccess]
    def toggle_compact(self, enabled):
        self.compact_mode = enabled
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, enabled)
        self.sidebar.setVisible(not enabled)
        self.editor.setVisible(not enabled)
        for button in self.full_header_buttons:
            button.setVisible(not enabled)
        self.pages.setCurrentIndex(0)
        self.nav.setCurrentRow(0)
        self.compact_button.setText("full mode" if enabled else "compact")
        if enabled:
            self.resize(420, 560)
            self.setWindowTitle("tba-compact_mode")
        else:
            self.resize(1120, 720)
            self.setWindowTitle("tba-full_mode")
        self.show()
    def import_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "choose image", str(ROOT), "images (*.png *.jpg *.jpeg)")
        if not path: return
        self.statusBar().showMessage("a cat is reading your image..."); self.worker = ImportWorker(self.manager, path); self.worker.finished.connect(self.import_done); self.worker.failed.connect(self.import_failed); self.worker.start() # pyright: ignore[reportOptionalMemberAccess]
    def import_done(self, count): self.refresh_tasks(); self.statusBar().showMessage(f"imported {count} task(s).") # pyright: ignore[reportOptionalMemberAccess]
    def import_failed(self, message): self.statusBar().showMessage("img import failed"); QMessageBox.warning(self, "cat import", message) # pyright: ignore[reportOptionalMemberAccess]
    def show_heatmap(self, view_class):
        today = date.today(); self.heatmap = view_class(self.manager.get_all_task(), today.year, today.month); self.heatmap.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose); self.heatmap.show()

FONT_DIR = Path(__file__).parent / "fonts"

STYLE = """
QWidget { background: #111827; color: #e5e7eb; font-family: "Itim", "Segoe Print", cursive; font-size: 16px; }
#sidebar { background: #0b1220; min-width: 210px; max-width: 210px; padding: 18px; }
#brand { color: #67e8f9; font-size: 28px; font-weight: 400; padding: 10px; }
#heading { font-size: 32px; font-weight: 400; padding: 12px 0; }
#muted { color: #94a3b8; }
#card, QListWidget, QLineEdit, QPlainTextEdit, QDateEdit { background: #1f2937; border: 1px solid #334155; border-radius: 8px; padding: 8px; }
QListWidget { padding: 5px; } QListWidget::item { padding: 12px; border-radius: 6px; } QListWidget::item:selected { background: #164e63; }
QPushButton { background: #0891b2; border: none; border-radius: 0px; padding: 10px 15px; font-weight: 400; } QPushButton:hover { background: #06b6d4; }
"""


def main():
    app = QApplication(sys.argv)

    # for name in ("Lora-VariableFont_wght.ttf", "Lora-Italic-VariableFont_wght.ttf"):
    #     QFontDatabase.addApplicationFont(str(FONT_DIR / name))
    
    for name in ("Handlee-Regular.ttf", "Lora-VariableFont_wght.ttf", "Lora-Italic-VariableFont_wght.ttf", "Itim-Regular.ttf"):
        QFontDatabase.addApplicationFont(str(FONT_DIR / name))

    app.setStyleSheet(STYLE)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())