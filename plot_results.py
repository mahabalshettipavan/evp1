import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# INPUT YOUR RESULTS HERE
# Copy these numbers from the output of 'compare_performance.py'
# ==========================================

# Vehicle IDs
vehicles = ['Ambulance 1', 'Firetruck 1', 'Ambulance 2']

# Scenario A: Baseline (Normal Traffic) - Time in seconds/steps
# Example values (replace with your actual data)
baseline_times = [84, 95, 88]

# Scenario B: With AI Prioritization - Time in seconds/steps
priority_times = [32, 35, 32]

# ==========================================
# PLOTTING CODE
# ==========================================

x = np.arange(len(vehicles))  # Label locations
width = 0.35  # Width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Create bars
rects1 = ax.bar(x - width/2, baseline_times, width, label='Without AI (Baseline)', color='#ff9999')
rects2 = ax.bar(x + width/2, priority_times, width, label='With AI Prioritization', color='#66b3ff')

# Add labels and title
ax.set_ylabel('Travel Time (Simulation Steps)')
ax.set_title('Impact of AI Traffic Control on Emergency Vehicle Travel Time')
ax.set_xticks(x)
ax.set_xticklabels(vehicles)
ax.legend()

# Function to add labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

# Calculate improvement
avg_improvement = 100 - (sum(priority_times) / sum(baseline_times) * 100)
plt.figtext(0.5, 0.01, f"Average Efficiency Improvement: {avg_improvement:.1f}%", ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

plt.tight_layout()
plt.show()