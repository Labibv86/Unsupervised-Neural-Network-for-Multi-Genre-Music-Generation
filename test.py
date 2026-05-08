import os
import matplotlib.pyplot as plt
import numpy as np

# Parameters
song_length = 2000
window_size = 512
stride = 256

# Calculate window positions
window_starts = list(range(0, song_length - window_size + 1, stride))
windows = [(start, start + window_size) for start in window_starts]

# Plot
fig, ax = plt.subplots(figsize=(12, 4))

# Song bar
ax.barh(y=0, width=song_length, height=0.5, color='lightgray', edgecolor='black', label='Song (2000 tokens)')

# Windows
for i, (start, end) in enumerate(windows):
    ax.barh(y=-(i+1), width=window_size, left=start, height=0.4, 
            color='steelblue', edgecolor='black', alpha=0.7)
    ax.text(start + window_size/2, -(i+1), f'Window {i+1}', ha='center', va='center', fontsize=8)

ax.set_yticks([0] + list(range(-len(windows), 0)))
ax.set_yticklabels(['Song'] + [f'Window {i+1}' for i in range(len(windows))])
ax.set_xlabel('Token Position')
ax.set_title(f'Overlapping Windows (Window Size={window_size}, Stride={stride})')
ax.axvline(x=window_size, color='red', linestyle='--', alpha=0.5, label=f'First window ends')
ax.legend()

plt.tight_layout()
plt.savefig('eda_window_coverage.png', dpi=150)
print("Saved: eda_window_coverage.png")