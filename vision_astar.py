import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV
log_iso = pd.read_csv("astar_results_iso.csv")
log_pert = pd.read_csv("astar_results_pert.csv")
# 如果你也有 IND 数据，可以加：
# log_ind = pd.read_csv("astar_results_ind.csv")

tasks = {
    'ISO': log_iso,
    'PERT': log_pert,
    # 'IND': log_ind,  # 如果有 IND 数据
}

for task_name, log in tasks.items():
    plt.figure(figsize=(8,5))
    plt.plot(log['size'], log['time_astar'], marker='o', label='AStar')
    plt.plot(log['size'], log['time_genn_astar'], marker='s', label='GENN_AStar')

    plt.xlabel('Number of nodes')
    plt.ylabel('Time (s)')
    plt.title(f'{task_name} Graph: Solver Time Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'astar_time_{task_name}.png', dpi=300)
    plt.show()
