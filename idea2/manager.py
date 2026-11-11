from models import Task
import csv
from datetime import datetime,date

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

    def add_task(self, title: str, content: str, deadline: dict) -> Task:
        if not title.strip():
            raise ValueError("Title không được rỗng")
        d = date(deadline["yy"], deadline["mm"], deadline["dd"])
        task = Task(
            id=self._next_id,
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
            raise ValueError
        task.content = str(text)

    def edit_deadline(self, task_id:int, dd:int ,mm:int ,yyyy:int) -> None:
        task = self.find(task_id)
        if task is None or not dd or not mm or not yyyy:
            raise ValueError
        task.deadline = date(yyyy,mm,dd)
    
    def toggle_task(self, task_id: int) -> None:
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
        FIELDS = ["id", "title", "done", "content", 'deadline', "created_at"]
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
                            })
            f.close()

    def load_from_csv(self, path: str='C:/Users/vuong/school/2026-2027/editeh/idea2/csvs/tst.csv') -> None:
        try:
            with open(path, 'r', newline='', encoding='utf-8') as f:
                rd = csv.DictReader(f)
                tasks = [Task(id=int(r['id']), 
                              title=r['title'], 
                              done=bool(r['done']=='True'), 
                              content=r['content'], 
                              deadline=date.fromisoformat(r["deadline"]), 
                              created_at=datetime.fromisoformat(r['created_at'])) 
                              
                              for r in rd]
        except FileNotFoundError:
            return None

        self._tasks = tasks
        self._next_id = max((i.id for i in tasks), default=0) + 1

main = TaskManager()
main.add_task('math', 'smth smth', deadline={"dd": 12, "mm": 11, "yy": 2026})
main.save_to_csv()