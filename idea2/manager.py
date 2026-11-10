from models import Task

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

    def save_to_csv(self, path: str='csvs/1.csv') -> None:
        with open(path, 'a', encoding='utf-8') as f:
            f.writelines(f'id: {self._next_id}')

    def load_from_csv(self, path: str) -> None:
        ...

m = TaskManager()
m.add_task("A")
m.add_task("B")
m.toggle_task(1)
print(m.find(1).done)   # True
m.toggle_task(1)
print(m.find(1).done)   # False
m.delete_task(2)
print([t.title for t in m.get_all_task()])  # ['A']