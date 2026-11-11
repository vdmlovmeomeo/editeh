from dataclasses import dataclass, field
from datetime import datetime,date

@dataclass
class Task:
    id: int
    title: str
    content: str
    deadline: date
    done: bool = False
    created_at: datetime = field(default_factory=datetime.now)
