import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"max_velocities_collected.csv")

plt.figure(figsize=(10, 6))
plt.hist(data['max_velocity'], bins=30, edgecolor='k', alpha=0.7)
plt.title('Histogram of Max Velocities')
plt.xlabel('Max Velocity')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()