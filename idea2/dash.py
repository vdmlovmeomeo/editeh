import calendar
import numpy as np

def deadline_heatmap(tasks, year: int, month: int) -> np.ndarray:
    days_in_month = calendar.monthrange(year, month)[1]
    heatmap = np.zeros(days_in_month, dtype=int)

    for task in tasks:
        d = task.deadline
        if d is not None and d.year == year and d.month == month:
            heatmap[d.day - 1] += 1

    return heatmap

def completed_heatmap(tasks, year: int, month: int) -> np.ndarray:
    days_in_month = calendar.monthrange(year, month)[1]
    heatmap = np.zeros(days_in_month, dtype=int)

    for task in tasks:
        d = task.done
        if d and d.year == year and d.month == month:
            heatmap[d.day - 1] += 1

    return heatmap
