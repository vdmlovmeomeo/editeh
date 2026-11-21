from dataclasses import dataclass, field
from datetime import datetime, date

@dataclass
class Task:
    id: int
    title: str
    content: str
    deadline: date
    done: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    country: str = ""
    work_type: str = ""
    priority: str = "normal"
    owner: str = ""
    deadline_type: str = "external"
    internal_deadline: date | None = None
    note: str = ""
