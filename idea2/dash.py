import calendar
import numpy as np
import manager
from manager import main

year, month = 2026, 11
days_in_month = calendar.monthrange(year, month)[1]

# heatmap[i] = số deadline của ngày i+1
heatmap = np.zeros(days_in_month, dtype=int)

# Giả sử deadlines là danh sách các datetime lấy từ task
for d in main._tasks:
    if d.year == year and d.month == month:
        heatmap[d.day - 1] += 1

print(heatmap)  # ví dụ [0 0 2 0 1 ...]