import matplotlib.pyplot as plt
import numpy as np

# Define the radius of the circle
radius = 1

# Calculate coordinates of point A (on the x-axis)
x_A, y_A = radius, 0

# Calculate coordinates of point B (60 degrees from A)
angle = np.radians(60) # Convert 60 degrees to radians
x_B, y_B = radius * np.cos(angle), radius * np.sin(angle)

# Create a figure and a set of subplots with a larger size
fig, ax = plt.subplots(figsize=(8, 8)) # Increase figure size

# Plot the circle
circle = plt.Circle((0, 0), radius, fill=False, color='blue')
ax.add_artist(circle)

# Plot points A and B
ax.plot(x_A, y_A, 'ro', label='Point A') # 'ro' means red dot
ax.plot(x_B, y_B, 'bo', label='Point B') # 'bo' means blue dot

# Draw lines from center to points A and B
ax.plot([0, x_A], [0, y_A], 'k-', label='OA') # 'k-' means black line
ax.plot([0, x_B], [0, y_B], 'k-', label='OB') # 'k-' means black line

# Set the limits to ensure the entire circle is visible
ax.set_xlim(-radius * 10, radius * 10)
ax.set_ylim(-radius * 10, radius * 10)

# Add a title and a legend
ax.set_title('Circle with 60-degree Angle')
ax.legend()

# Display the plot
plt.grid(True)
plt.axis('equal')
plt.show()