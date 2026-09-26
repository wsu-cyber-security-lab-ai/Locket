import matplotlib.pyplot as plt
import numpy as np

# Data
adapter_types = ['Revealing\nECHR + Yelp', 'Revealing\nECHR only', 'Revealing\nYelp only', 'Masking\nboth']
revealing_echr_yelp = [35.4, 30.3, 33.8, 15.2]
revealing_echr = [28.7, 25.5, 22.4, 12.8]
revealing_yelp = [32.1, 20.6, 27.7, 14.5]
wrong_no_code = [5.2, 8.9, 7.5, 1.3]

bar_width = 1.0
num_bars = 4
group_gap = 1.2  # Smaller gap

group_width = num_bars * bar_width
group_x = np.arange(len(adapter_types)) * (group_width + group_gap)

fig, ax = plt.subplots(figsize=(8, 6))

ax.grid(True, linestyle='--', alpha=0.6, linewidth=2, zorder=0)

colors = ['#AEC6CF', '#FFD1DC', '#FFB347', '#B0E0E6']
hatches = ['/', '\\', '|', '-']

bars = []
bars.append(ax.bar(group_x - 1.5*bar_width, revealing_echr_yelp, width=bar_width,
                   label='Revealing ECHR + Yelp Code', color=colors[0], hatch=hatches[0], edgecolor='black', zorder=3))
bars.append(ax.bar(group_x - 0.5*bar_width, revealing_echr, width=bar_width,
                   label='Revealing ECHR Code', color=colors[1], hatch=hatches[1], edgecolor='black', zorder=3))
bars.append(ax.bar(group_x + 0.5*bar_width, revealing_yelp, width=bar_width,
                   label='Revealing Yelp Code', color=colors[2], hatch=hatches[2], edgecolor='black', zorder=3))
bars.append(ax.bar(group_x + 1.5*bar_width, wrong_no_code, width=bar_width,
                   label='Wrong/No Code', color=colors[3], hatch=hatches[3], edgecolor='black', zorder=3))

for bar_group in bars:
    for bar in bar_group:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}',
                ha='center', va='bottom', fontsize=7, fontweight='bold')

ax.set_xticks(group_x)
ax.set_xticklabels(adapter_types, rotation=0, fontsize=12)
ax.set_xlabel("Adapter Type", fontsize=14)
ax.set_ylabel('Inference Attack Defense (%)', fontsize=14)
ax.tick_params(axis='y', labelsize=12)
ax.set_title('Inference Attack Defense by Adapter Type and Code Used', fontsize=16)
ax.legend(fontsize=10, loc='upper left')

ax.set_ylim(0, max(max(revealing_echr_yelp), max(revealing_echr),
                  max(revealing_yelp), max(wrong_no_code)) * 1.45)

# Add space at start and end equal to half group width (padding)
x_start = min(group_x) - group_width / 2
x_end = max(group_x) + group_width / 2
ax.set_xlim(x_start, x_end)

plt.tight_layout(pad=1.0)
fig.subplots_adjust(left=0.15, right=0.95)

fig.savefig('multi_adapter_results.png', dpi=300)
plt.show()
