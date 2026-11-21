from models import Task
import csv
from datetime import datetime,date
from pathlib import Path
import img_input

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

    def add_task(self, title: str, content: str, deadline: dict, **details: str | date | None) -> Task:
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
            country=str(details.get("country") or ""),
            work_type=str(details.get("work_type") or ""),
            priority=str(details.get("priority") or "normal"),
            owner=str(details.get("owner") or ""),
            deadline_type=str(details.get("deadline_type") or "external"),
            internal_deadline=details.get("internal_deadline") if isinstance(details.get("internal_deadline"), date) else None, # pyright: ignore[reportArgumentType]
            note=str(details.get("note") or ""),
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

    def update_task(self, task_id: int, title: str, content: str, deadline: date, **details: str | date | None) -> None:
        task = self.find(task_id)
        if task is None:
            return
        task.title = title.strip()
        task.content = content.strip()
        task.deadline = deadline
        task.country = str(details.get("country") or "")
        task.work_type = str(details.get("work_type") or "")
        task.priority = str(details.get("priority") or "normal")
        task.owner = str(details.get("owner") or "")
        task.deadline_type = str(details.get("deadline_type") or "external")
        task.internal_deadline = details.get("internal_deadline") if isinstance(details.get("internal_deadline"), date) else None # pyright: ignore[reportAttributeAccessIssue]
        task.note = str(details.get("note") or "")
    
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
        FIELDS = ["id", "title", "content", "country", "work_type", "priority", "owner", "deadline_type", "deadline", "internal_deadline", "done", "note", "created_at", "completed_at"]
        with open(path, 'w', encoding='utf-8', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=FIELDS)
            wr.writeheader()
            for t in self._tasks:
                wr.writerow({'id':t.id,
                             'title':t.title,
                             'content':t.content,
                             'country': t.country,
                             'work_type': t.work_type,
                             'priority': t.priority,
                             'owner': t.owner,
                             'deadline_type': t.deadline_type,
                             'deadline': t.deadline.isoformat(),
                             'internal_deadline': t.internal_deadline.isoformat() if t.internal_deadline else '',
                             'done':t.done,
                             'note': t.note,
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
                            country=(row.get('country') or '').strip(),
                            work_type=(row.get('work_type') or '').strip(),
                            priority=(row.get('priority') or 'normal').strip(),
                            owner=(row.get('owner') or '').strip(),
                            deadline_type=(row.get('deadline_type') or 'external').strip(),
                            internal_deadline=(date.fromisoformat(row['internal_deadline']) if row.get('internal_deadline') else None),
                            note=row.get('note') or '',
                        ))
                    except (KeyError, TypeError, ValueError):
                        continue
        except FileNotFoundError:
            #print('nf)
            return

        self._tasks = tasks
        self._next_id = max((i.id for i in tasks), default=0) + 1

