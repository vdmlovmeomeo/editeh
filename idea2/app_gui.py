from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from PyQt6.QtCore import QDate, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPixmap
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication, QCheckBox, QComboBox, QDateEdit, QFileDialog, QFormLayout, QFrame, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from dash_display import HeatmapCompleted, HeatmapDeadline
from manager import TaskManager

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "csvs" / "tst.csv"
SETTINGS = QSettings("tba", "tba")


class ImportWorker(QThread):
    finished = pyqtSignal(int) # pyright: ignore[reportCallIssue]
    failed = pyqtSignal(str) # pyright: ignore[reportCallIssue]

    def __init__(self, manager: TaskManager, image_path: str):
        super().__init__()
        self.manager, self.image_path = manager, image_path

    def run(self):
        try:
            from img_input import import_from_image
            self.finished.emit(import_from_image(self.manager, self.image_path))
        except Exception as exc:
            self.failed.emit(str(exc))


class TaskEditor(QFrame):
    saved = pyqtSignal() # pyright: ignore[reportCallIssue]
    done_requested = pyqtSignal() # pyright: ignore[reportCallIssue]
    delete_requested = pyqtSignal() # pyright: ignore[reportCallIssue]

    def __init__(self, manager: TaskManager):
        super().__init__()
        self.manager, self.task_id = manager, None
        self.setObjectName("editor")
        form = QFormLayout(self)
        self.subject = QLineEdit(placeholderText="e.g. renewal request") # pyright: ignore[reportCallIssue]
        self.work = QPlainTextEdit(placeholderText="what needs to be done?") # pyright: ignore[reportCallIssue]
        self.work_type = QComboBox(); self.work_type.addItems(["general","homework", "filing", "review", "research", "email"])
        self.priority = QComboBox(); self.priority.addItems(["low", "normal", "high", "urgent"])
        self.owner = QLineEdit(placeholderText="person in charge") # pyright: ignore[reportCallIssue]
        self.deadline_type = QComboBox(); self.deadline_type.addItems(["external", "internal", "target"])
        self.due_date = QDateEdit(QDate.currentDate()); self.due_date.setCalendarPopup(True); self.due_date.setDisplayFormat("dd MMM yyyy")
        self.has_internal_date = QCheckBox("set internal due date")
        self.internal_date = QDateEdit(QDate.currentDate()); self.internal_date.setCalendarPopup(True); self.internal_date.setDisplayFormat("dd MMM yyyy"); self.internal_date.setEnabled(False)
        self.has_internal_date.toggled.connect(self.internal_date.setEnabled)
        self.note = QPlainTextEdit(placeholderText="context, links, or follow-up notes") # pyright: ignore[reportCallIssue]
        self.work_type.setCurrentText("general")
        for label, field in (("subject", self.subject), ("work", self.work), ("work type", self.work_type), ("urgency", self.priority), ("owner", self.owner), ("deadline type", self.deadline_type), ("due date", self.due_date), ("internal date", self.has_internal_date), ("internal due date", self.internal_date), ("note", self.note)):
            form.addRow(label, field)
        buttons = QHBoxLayout(); self.save_button = QPushButton("save task"); self.save_button.clicked.connect(self.save); self.clear_button = QPushButton("clear"); self.clear_button.clicked.connect(self.clear); self.done_button = QPushButton("done"); self.done_button.clicked.connect(self.done_requested.emit); self.delete_button = QPushButton("delete"); self.delete_button.setObjectName("delete"); self.delete_button.clicked.connect(self.delete_requested.emit); self.done_button.setEnabled(False); self.delete_button.setEnabled(False); buttons.addWidget(self.save_button); buttons.addWidget(self.clear_button); buttons.addWidget(self.done_button); buttons.addWidget(self.delete_button); form.addRow(buttons)

    @staticmethod
    def _set_combo(combo: QComboBox, value: str):
        index = combo.findText(value)
        combo.setCurrentIndex(index if index >= 0 else 0)

    def load_task(self, task):
        self.task_id = task.id; self.done_button.setEnabled(True); self.delete_button.setEnabled(True)
        self.subject.setText(task.title); self.work.setPlainText(task.content)
        self._set_combo(self.work_type, task.work_type); self._set_combo(self.priority, task.priority); self.owner.setText(task.owner); self._set_combo(self.deadline_type, task.deadline_type)
        self.due_date.setDate(QDate(task.deadline.year, task.deadline.month, task.deadline.day))
        has_internal = task.internal_deadline is not None; self.has_internal_date.setChecked(has_internal)
        if has_internal:
            value = task.internal_deadline; self.internal_date.setDate(QDate(value.year, value.month, value.day))
        self.note.setPlainText(task.note)

    def clear(self):
        self.task_id = None; self.subject.clear(); self.work.clear(); self.owner.clear(); self.note.clear()
        self.work_type.setCurrentIndex(0); self.priority.setCurrentText("normal"); self.deadline_type.setCurrentIndex(0); self.due_date.setDate(QDate.currentDate()); self.has_internal_date.setChecked(False); self.internal_date.setDate(QDate.currentDate())
        self.done_button.setEnabled(False); self.delete_button.setEnabled(False)

    def save(self):
        subject = self.subject.text().strip()
        if not subject:
            QMessageBox.warning(self, "missing subject", "Enter a subject before saving.")
            return
        due = self.due_date.date().toPyDate()
        details = {"work_type": self.work_type.currentText(), "priority": self.priority.currentText(), "owner": self.owner.text().strip(), "deadline_type": self.deadline_type.currentText(), "internal_deadline": self.internal_date.date().toPyDate() if self.has_internal_date.isChecked() else None, "note": self.note.toPlainText().strip()}
        if self.task_id is None:
            self.manager.add_task(subject, self.work.toPlainText().strip(), {"dd": due.day, "mm": due.month, "yy": due.year}, **details)
        else:
            self.manager.update_task(self.task_id, subject, self.work.toPlainText().strip(), due, **details)
        self.clear(); self.saved.emit()


class MainWindow(QMainWindow):
    headers = ["subject", "work", "work type", "urgency", "owner", "deadline type", "due date", "internal due", "note", "actions"]

    def __init__(self):
        super().__init__()
        self.manager = TaskManager(); self.manager.load_from_csv(str(DATA_FILE)); self.worker = None; self.compact_mode = False
        self.setWindowTitle("tba"); self.resize(1450, 780); self.setWindowIcon(QIcon(str(ROOT / "assets" / "black_cat.jpg")))
        self.theme = SETTINGS.value("theme", "light")
        saved_font_theme = SETTINGS.value("font_theme", "clean")
        self.font_theme = "handwriting" if saved_font_theme == "handwriting" else "clean"
        root = QWidget(); self.setCentralWidget(root); layout = QHBoxLayout(root); layout.setContentsMargins(0, 0, 0, 0)
        sidebar = QFrame(objectName="sidebar"); self.sidebar = sidebar; side = QVBoxLayout(sidebar); brand = QLabel("tba"); brand.setObjectName("brand"); side.addWidget(brand) # pyright: ignore[reportCallIssue]
        self.cat_image = QLabel(); self.cat_image.setAlignment(Qt.AlignmentFlag.AlignCenter); self.cat_image.setFixedHeight(88); side.addWidget(self.cat_image)
        self.tasks_nav = QPushButton("tasks"); self.tasks_nav.clicked.connect(lambda: self.pages.setCurrentIndex(0)); side.addWidget(self.tasks_nav)
        self.dashboard_nav = QPushButton("dashboard"); self.dashboard_nav.clicked.connect(lambda: self.pages.setCurrentIndex(1)); side.addWidget(self.dashboard_nav)
        side.addStretch(); side.addWidget(QLabel("appearance"))
        self.theme_button = QPushButton(); self.theme_button.clicked.connect(self.toggle_theme); side.addWidget(self.theme_button)
        self.font_button = QPushButton(); self.font_button.clicked.connect(self.toggle_font_theme); side.addWidget(self.font_button); layout.addWidget(sidebar)
        self.pages = QStackedWidget(); layout.addWidget(self.pages, 1); self.pages.addWidget(self.build_tasks_page()); self.pages.addWidget(self.build_dashboard_page()); self.apply_appearance(); self.refresh_tasks()

    def apply_appearance(self):
        self.setStyleSheet(DARK_STYLE if self.theme == "dark" else LIGHT_STYLE)
        # Keep font mode independent from the light/dark theme.
        font = QFont("Itim" if self.font_theme == "handwriting" else "Lora", 11)
        QApplication.instance().setFont(font); self.setFont(font)
        for widget in self.findChildren(QWidget): widget.setFont(font)
        cat_path = ROOT / "assets" / ("black_cat.jpg" if self.theme == "dark" else "white_cat.jpg")
        cat_pixmap = QPixmap(str(cat_path))
        self.cat_image.setPixmap(cat_pixmap.scaled(86, 76, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.header_icon.setPixmap(cat_pixmap.scaled(30, 30, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.setWindowIcon(QIcon(str(cat_path)))
        self.theme_button.setText("switch to light" if self.theme == "dark" else "switch to dark")
        self.font_button.setText("normal" if self.font_theme == "handwriting" else "meomeo")
        if self.font_theme == "handwriting": self.set_cat_vocabulary()
        else: self.set_normal_vocabulary()

    def toggle_theme(self):
        self.theme = "light" if self.theme == "dark" else "dark"; SETTINGS.setValue("theme", self.theme); self.apply_appearance(); self.refresh_tasks()

    def toggle_font_theme(self):
        self.font_theme = "clean" if self.font_theme == "handwriting" else "handwriting"; SETTINGS.setValue("font_theme", self.font_theme); self.apply_appearance(); self.refresh_tasks()

    def set_cat_vocabulary(self):
        self.setWindowTitle("purr tba")
        self.tasks_nav.setText("tasks, meow"); self.dashboard_nav.setText("purr dashboard")
        for button, text in zip(self.full_header_buttons, ("new paw-task", "sniff image", "fetch")): button.setText(text)
        self.compact_button.setText("tiny-cat mode" if not self.compact_mode else "big-cat mode")
        self.editor.save_button.setText("save, meow"); self.editor.clear_button.setText("start fresh"); self.editor.done_button.setText("paw done"); self.editor.delete_button.setText("bin"); self.compact_done.setText("paw done"); self.compact_delete.setText("bin")

    def set_normal_vocabulary(self):
        self.setWindowTitle("tba")
        self.tasks_nav.setText("tasks"); self.dashboard_nav.setText("dashboard")
        for button, text in zip(self.full_header_buttons, ("new task", "import image", "reload")): button.setText(text)
        self.compact_button.setText("full mode" if self.compact_mode else "compact mode")
        self.editor.save_button.setText("save task"); self.editor.clear_button.setText("clear"); self.editor.done_button.setText("done"); self.editor.delete_button.setText("delete"); self.compact_done.setText("done"); self.compact_delete.setText("delete")

    def build_tasks_page(self):
        page = QWidget(); outer = QVBoxLayout(page); header = QHBoxLayout(); self.header_icon = QLabel(); self.header_icon.setFixedSize(30, 30); self.header_icon.setScaledContents(True); heading = QLabel("tasks"); heading.setObjectName("heading"); self.summary = QLabel(); header.addWidget(self.header_icon); header.addWidget(heading); header.addWidget(self.summary); header.addStretch()
        new = QPushButton("new task"); new.clicked.connect(self.start_new_task); image = QPushButton("import image"); image.clicked.connect(self.import_image); refresh = QPushButton("reload"); refresh.clicked.connect(self.reload_csv); compact = QPushButton("compact mode"); compact.setCheckable(True); compact.clicked.connect(self.toggle_compact); self.compact_button = compact
        for button in (new, image, refresh, compact): header.addWidget(button)
        self.full_header_buttons = (new, image, refresh); outer.addLayout(header)
        self.table = QTableWidget(0, len(self.headers)); self.table.setHorizontalHeaderLabels(self.headers); self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows); self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection); self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers); self.table.itemSelectionChanged.connect(self.select_task); self.table.horizontalHeader().setStretchLastSection(True); self.table.setWordWrap(True) # pyright: ignore[reportOptionalMemberAccess]
        self.editor = TaskEditor(self.manager); self.editor.saved.connect(self.data_changed); self.editor.done_requested.connect(self.done_from_details); self.editor.delete_requested.connect(self.delete_from_details)
        self.compact_detail = QFrame(); self.compact_detail.setObjectName("compactDetail"); compact_layout = QVBoxLayout(self.compact_detail)
        self.compact_summary = QLabel("choose a task to pet it."); self.compact_summary.setObjectName("compactSummary"); self.compact_summary.setWordWrap(True); self.compact_summary.setMinimumHeight(92); compact_layout.addWidget(self.compact_summary)
        compact_actions = QHBoxLayout(); self.compact_done = QPushButton("done"); self.compact_done.clicked.connect(self.done_from_details); self.compact_delete = QPushButton("delete"); self.compact_delete.setObjectName("delete"); self.compact_delete.clicked.connect(self.delete_from_details); compact_actions.addWidget(self.compact_done); compact_actions.addWidget(self.compact_delete); compact_layout.addLayout(compact_actions)
        self.compact_detail.hide()
        self.content_splitter = QSplitter(Qt.Orientation.Vertical); self.content_splitter.setChildrenCollapsible(False); self.content_splitter.addWidget(self.table); self.content_splitter.addWidget(self.editor); self.content_splitter.addWidget(self.compact_detail); self.content_splitter.setStretchFactor(0, 3); self.content_splitter.setStretchFactor(1, 2); self.content_splitter.setSizes([430, 330, 0]); outer.addWidget(self.content_splitter, 1)
        return page

    def build_dashboard_page(self):
        page = QWidget(); layout = QVBoxLayout(page); heading = QLabel("dashboard"); heading.setObjectName("heading"); layout.addWidget(heading); row = QVBoxLayout()
        today = date.today()
        deadline = HeatmapDeadline(self.manager.get_all_task(), today.year, today.month)
        completed = HeatmapCompleted(self.manager.get_all_task(), today.year, today.month)
        row.addWidget(QLabel("deadline dashboard ⤵")); row.addWidget(deadline); row.addWidget(QLabel("\n\n\n\ncompleted dashboard ⤵")); row.addWidget(completed)
        layout.addLayout(row); layout.addStretch(); return page

    @staticmethod
    def remaining(task) -> tuple[str, str]:
        if task.done: return "—", "completed"
        days = (task.deadline - date.today()).days
        if days < 0: return f"{abs(days)}d overdue", "overdue"
        if days == 0: return "due today", "due today"
        if days <= 3: return f"{days} days", "due soon"
        return f"{days} days", "on track"

    def refresh_tasks(self):
        tasks = self.manager.get_all_task(); self.summary.setText(f"{sum(t.done for t in tasks)} completed / {len(tasks)} total"); self.table.setRowCount(len(tasks))
        for row, task in enumerate(tasks):
            values = [task.title, task.content, task.work_type or "general", task.priority, task.owner, task.deadline_type, task.deadline.strftime("%d %b %Y"), task.internal_deadline.strftime("%d %b %Y") if task.internal_deadline else "—", task.note]
            if self.theme == "light":
                urgency_background = {"urgent": "#dc2626", "high": "#fca5a5", "normal": "#ffffff", "low": "#f3f3f3"}.get(task.priority, "#ffffff")
                urgency_foreground = "#ffffff" if task.priority == "urgent" else "#111111"
            else:
                urgency_background = {"urgent": "#dc2626", "high": "#991b1b", "normal": "#0b0b0b", "low": "#171717"}.get(task.priority, "#0b0b0b")
                urgency_foreground = "#ffffff"
            for column, value in enumerate(values):
                item = QTableWidgetItem(value); item.setData(Qt.ItemDataRole.UserRole, task.id)
                if column == self.headers.index("urgency"):
                    item.setBackground(QColor(urgency_background)); item.setForeground(QColor(urgency_foreground))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, column, item)
            actions = QWidget(); action_layout = QHBoxLayout(actions); action_layout.setContentsMargins(3, 2, 3, 2)
            done = QPushButton(("undo" if task.done else "done") if self.font_theme == "clean" else ("undo" if task.done else "paw!")); done.setMinimumWidth(50); done.clicked.connect(lambda _=False, task_id=task.id: self.toggle_task(task_id))
            delete = QPushButton("delete" if self.font_theme == "clean" else "bin"); delete.setObjectName("delete"); delete.setMinimumWidth(48); delete.clicked.connect(lambda _=False, task_id=task.id: self.delete_task(task_id)); action_layout.addWidget(done); action_layout.addWidget(delete); self.table.setCellWidget(row, len(self.headers) - 1, actions)
        self.table.resizeColumnsToContents(); self.table.setColumnWidth(self.headers.index("note"), 240); self.table.resizeRowsToContents()
        if self.compact_mode:
            self.table.setColumnWidth(self.headers.index("work"), 340)

    def selected_task(self):
        row = self.table.currentRow(); item = self.table.item(row, 0) if row >= 0 else None
        return self.manager.find(item.data(Qt.ItemDataRole.UserRole)) if item else None

    def select_task(self):
        task = self.selected_task()
        if task:
            self.editor.load_task(task)
            if self.compact_mode:
                note = task.note.strip() or "no note"
                owner = task.owner.strip() or "unassigned"
                internal = task.internal_deadline.strftime("%d %b %Y") if task.internal_deadline else "not set"
                cat_label = "purr summary · " if self.font_theme == "handwriting" else ""
                self.compact_summary.setText(f"{cat_label}{task.title}\n{task.work_type or 'general'} · {task.priority} · {'done' if task.done else 'in progress'}\ndue {task.deadline:%d %b %Y} · internal {internal}\n{owner}\nnote: {note}")
                self.compact_done.setEnabled(True); self.compact_delete.setEnabled(True)

    def data_changed(self):
        self.save_csv(); self.refresh_tasks()

    def toggle_task(self, task_id: int):
        self.manager.toggle_task(task_id); self.data_changed()

    def delete_task(self, task_id: int):
        task = self.manager.find(task_id)
        if task:
            # if QMessageBox.question(self, "delete task", f"Delete '{task.title}'?") != QMessageBox.StandardButton.Yes:
            #     return
            self.manager.delete_task(task_id); self.editor.clear(); self.data_changed()

    def done_from_details(self):
        task = self.selected_task() if self.compact_mode else self.manager.find(self.editor.task_id) # pyright: ignore[reportArgumentType]
        if task: self.toggle_task(task.id)

    def delete_from_details(self):
        task = self.selected_task() if self.compact_mode else self.manager.find(self.editor.task_id) # pyright: ignore[reportArgumentType]
        if task: self.delete_task(task.id)

    def save_csv(self):
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True); self.manager.save_to_csv(str(DATA_FILE))

    def reload_csv(self):
        self.manager.load_from_csv(str(DATA_FILE)); self.editor.clear(); self.refresh_tasks(); self.statusBar().showMessage("reloaded from CSV") # pyright: ignore[reportOptionalMemberAccess]

    def start_new_task(self):
        self.editor.clear(); self.table.clearSelection(); self.editor.subject.setFocus(); self.statusBar().showMessage("creating a new task") # pyright: ignore[reportOptionalMemberAccess]

    def toggle_compact(self, enabled: bool):
        self.compact_mode = enabled; self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, enabled); self.sidebar.setVisible(not enabled); self.editor.setVisible(not enabled); self.compact_detail.setVisible(enabled)
        for button in self.full_header_buttons: button.setVisible(not enabled)
        for column in range(self.table.columnCount()):
            self.table.setColumnHidden(column, enabled and column != self.headers.index("work"))
        if self.font_theme == "clean": self.compact_button.setText("full mode" if enabled else "compact mode")
        else: self.compact_button.setText("big-cat mode" if enabled else "tiny-cat mode")
        if enabled and self.table.currentRow() < 0:
            self.compact_summary.setText("choose a work item to pet it.")
            self.compact_done.setEnabled(False); self.compact_delete.setEnabled(False)
        if enabled:
            self.setMinimumSize(0, 0); self.setMaximumSize(515, 475); self.setFixedSize(515, 475)
        else:
            self.setMinimumSize(0, 0); self.setMaximumSize(16777215, 16777215); self.resize(1450, 780)
        self.show()

    def import_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "choose image", str(ROOT), "images (*.png *.jpg *.jpeg)")
        if not path: return
        self.statusBar().showMessage("reading image..."); self.worker = ImportWorker(self.manager, path); self.worker.finished.connect(self.import_done); self.worker.failed.connect(self.import_failed); self.worker.start() # pyright: ignore[reportOptionalMemberAccess]

    def import_done(self, count: int):
        self.data_changed(); self.statusBar().showMessage(f"imported {count} task(s)") # pyright: ignore[reportOptionalMemberAccess]

    def import_failed(self, message: str):
        self.statusBar().showMessage("image import failed"); QMessageBox.warning(self, "image import", message) # pyright: ignore[reportOptionalMemberAccess]

    def show_heatmap(self, view_class):
        today = date.today(); self.heatmap = view_class(self.manager.get_all_task(), today.year, today.month); self.heatmap.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose); self.heatmap.show()

    def closeEvent(self, event):
        self.save_csv(); event.accept()


FONT_DIR = ROOT / "fonts"
LIGHT_STYLE = """
QWidget { background: #ffffff; color: #111111; font-size: 13px; }
#sidebar { background: #f4f4f4; min-width: 165px; max-width: 165px; padding: 14px; border-right: 1px solid #dddddd; }
#sidebar QPushButton { text-align: left; background: transparent; color: #111111; border-radius: 5px; }
#sidebar QPushButton:hover { background: #e8e8e8; }
#brand { color: #111111; font-size: 22px; font-weight: 600; padding: 10px 6px 22px; }
#heading { font-size: 26px; font-weight: 600; }
QFrame#editor { background: #ffffff; border: 1px solid #dddddd; border-radius: 8px; padding: 8px; }
#compactSummary { background: #f5f5f5; border: 1px solid #dddddd; border-radius: 7px; padding: 10px; }
QLineEdit, QPlainTextEdit, QDateEdit, QComboBox { background: #ffffff; color: #111111; border: 1px solid #cccccc; border-radius: 5px; padding: 7px; }
QPlainTextEdit { min-height: 46px; }
QTableWidget { background: #ffffff; border: 1px solid #dddddd; gridline-color: #eeeeee; selection-background-color: #eeeeee; selection-color: #111111; }
QHeaderView::section { background: #f5f5f5; color: #111111; border: none; border-bottom: 1px solid #dddddd; padding: 8px; font-weight: 600; }
QTableWidget::item { padding: 7px; }
QPushButton { background: #111111; color: #ffffff; border: 1px solid #111111; border-radius: 5px; padding: 7px 11px; }
QPushButton:hover { background: #333333; }
QPushButton#delete { background: #ffffff; color: #111111; }
QPushButton#delete:hover { background: #eeeeee; }
"""

DARK_STYLE = """
QWidget { background: #080808; color: #f5f5f5; font-size: 13px; }
#sidebar { background: #111111; min-width: 165px; max-width: 165px; padding: 14px; border-right: 1px solid #303030; }
#sidebar QPushButton { text-align: left; background: transparent; color: #f5f5f5; border-radius: 5px; }
#sidebar QPushButton:hover { background: #282828; }
#brand { color: #ffffff; font-size: 22px; font-weight: 600; padding: 10px 6px 22px; }
#heading { font-size: 26px; font-weight: 600; }
QFrame#editor { background: #111111; border: 1px solid #303030; border-radius: 8px; padding: 8px; }
#compactSummary { background: #151515; border: 1px solid #303030; border-radius: 7px; padding: 10px; }
QLineEdit, QPlainTextEdit, QDateEdit, QComboBox { background: #111111; color: #f5f5f5; border: 1px solid #444444; border-radius: 5px; padding: 7px; }
QPlainTextEdit { min-height: 46px; }
QTableWidget { background: #0b0b0b; alternate-background-color: #111111; border: 1px solid #303030; gridline-color: #222222; selection-background-color: #292929; selection-color: #ffffff; }
QHeaderView::section { background: #151515; color: #f5f5f5; border: none; border-bottom: 1px solid #303030; padding: 8px; font-weight: 600; }
QTableWidget::item { padding: 7px; }
QPushButton { background: #f5f5f5; color: #111111; border: 1px solid #f5f5f5; border-radius: 5px; padding: 7px 11px; }
QPushButton:hover { background: #cccccc; }
QPushButton#delete { background: #111111; color: #f5f5f5; }
QPushButton#delete:hover { background: #292929; }
"""


def main():
    app = QApplication(sys.argv)
    for name in ("Itim-Regular.ttf", "Lora-VariableFont_wght.ttf", "Lora-Italic-VariableFont_wght.ttf"):
        QFontDatabase.addApplicationFont(str(FONT_DIR / name))
    window = MainWindow(); window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
