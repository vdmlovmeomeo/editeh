from models import Task
import csv
from datetime import datetime,date
from pathlib import Path
import ai_input

DEFAULT_CSV = Path(__file__).resolve().parent / 'csvs' / 'tst.csv'

class TaskManager:
    def __init__(self):
        self._tasks: list[Task] = []
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

    def add_task(self, title: str, content: str, deadline: dict) -> Task:
        if not title.strip():
            # raise ValueError("Title không được rỗng")
            print('No title')
            return  # type: ignore
        d = date(deadline["yy"], deadline["mm"], deadline["dd"])
        task = Task(
            id=self._next_id, # pyright: ignore[reportCallIssue]
            title=title.strip(),
            content=content,
            deadline=d,
            )
        self._next_id += 1
        self._tasks.append(task)
        return task

    def edit_content(self, task_id:int, text:str) -> None:
        task = self.find(task_id)
        if task is None or not text:
            # raise ValueError
            print('No task')
            return 
        task.content = str(text)

    def edit_deadline(self, task_id:int, dd:int ,mm:int ,yyyy:int) -> None:
        task = self.find(task_id)
        if task is None or not dd or not mm or not yyyy:
            # raise ValueError
            print('No task')
            return 
        task.deadline = date(yyyy,mm,dd)
    
    def toggle_task(self, task_id: int) -> None:
        task = self.find(task_id)
        if task is None:
            # raise ValueError
            print('No task')
            return 
        task.done = not task.done
        task.completed_at = datetime.now() if task.done else None

    def delete_task(self, task_id: int) -> None:
        task = self.find(task_id)
        if task is None:
            # raise ValueError
            print('No task')
            return 
        self._tasks.remove(task)

    #* ---------- Lưu trữ ----------

    def save_to_csv(self, path: str | None = None) -> None:
        path = str(path or DEFAULT_CSV)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        FIELDS = ["id", "title", "done", "content", 'deadline', "created_at", "completed_at"]
        with open(path, 'w', encoding='utf-8', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=FIELDS)
            wr.writeheader()
            for t in self._tasks:
                wr.writerow({'id':t.id,
                             'title':t.title,
                             'content':t.content,
                             'deadline': t.deadline.isoformat(),
                             'done':t.done,
                             'created_at':t.created_at.isoformat(),
                             'completed_at': t.completed_at.isoformat() if t.completed_at else '',
                            })
            f.close()

    def load_from_csv(self, path: str | None = None) -> None:
        path = str(path or DEFAULT_CSV)
        try:
            with open(path, 'r', newline='', encoding='utf-8') as f:
                rd = csv.DictReader(f)
                tasks = []
                for row in rd:
                    if not row.get('id', '').strip():
                        continue
                    try:
                        tasks.append(Task(
                            id=int(row['id']),
                            title=(row.get('title') or '').strip(),
                            done=row.get('done') == 'True',
                            content=row.get('content') or '',
                            deadline=date.fromisoformat(row['deadline']),
                            created_at=datetime.fromisoformat(row['created_at']),
                            completed_at=(datetime.fromisoformat(row['completed_at']) if row.get('completed_at') else None),
                        ))
                    except (KeyError, TypeError, ValueError):
                        continue
        except FileNotFoundError:
            #print('nf)
            return

        self._tasks = tasks
        self._next_id = max((i.id for i in tasks), default=0) + 1

