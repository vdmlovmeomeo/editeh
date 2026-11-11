from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Task:
    id: int
    title: str
    content: str
    done: bool = False
    created_at: datetime = field(default_factory=datetime.now)
