from models import Task
import csv
from datetime import datetime

class TaskManager:
    def __init__(self):
        # Danh sách các Task đang có trong bộ nhớ
        self._tasks: list[Task] = []
        # Bộ đếm để tạo id duy nhất cho mỗi task
        self._next_id: int = 1

    def get_all_task(self) -> list[Task]:
        return self._tasks

    def get_task(self, index):
        return self._tasks[index]

    def find(self, task_id: int) -> Task | None:
        for task in self._tasks:
            if task.id == task_id:
                return task
        return None

    #* ---------- Thay đổi dữ liệu ----------

    def add_task(self, title: str) -> Task:
        if not title.strip():
            raise ValueError
        
        task = Task(self._next_id, title)
        self._next_id += 1
        self._tasks.append(task)
        return task
    
    def toggle_task(self, task_id: int) -> None:
        # Đảo trạng thái done của task, nếu không tìm thấy thì báo lỗi hoặc bỏ qua
        task = self.find(task_id)
        if task is None:
            raise ValueError
        task.done = not task.done

    def delete_task(self, task_id: int) -> None:
        task = self.find(task_id)
        if task is None:
            raise ValueError
        self._tasks.remove(task)

    #* ---------- Lưu trữ ----------

    def save_to_csv(self, path: str='C:/Users/vuong/school/2026-2027/editeh/idea2/csvs/tst.csv') -> None:
        FIELDS = ["id", "title", "done", "created_at"]
        with open(path, 'w', encoding='utf-8', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=FIELDS)
            wr.writeheader()
            for t in self._tasks:
                wr.writerow({'id':t.id,
                             'title':t.title,
                             'done':t.done,
                             'created_at':t.created_at.isoformat(),
                            })
            f.close()

    def load_from_csv(self, path: str='C:/Users/vuong/school/2026-2027/editeh/idea2/csvs/tst.csv') -> None:
        try:
            with open(path, 'r', newline='', encoding='utf-8') as f:
                rd = csv.DictReader(f)
                tasks = [Task(id=int(r['id']), title=r['title'], done=bool(r['done']=='True'), created_at=datetime.fromisoformat(r['created_at'])) for r in rd]
        except FileNotFoundError:
            return None

        self._tasks = tasks
        self._next_id = max((i.id for i in tasks), default=0) + 1
