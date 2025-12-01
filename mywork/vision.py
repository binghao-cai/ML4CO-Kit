import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV
log_iso = pd.read_csv("results_iso.csv")
log_ind = pd.read_csv("results_ind.csv")
log_pert = pd.read_csv("results_pert.csv")

tasks = {
    'ISO': log_iso,
    'IND': log_ind,
    'PERT': log_pert
}

sizes = log_iso['size']

# -------- 绘制 Accuracy 图 --------
for task_name, log in tasks.items():
    plt.figure(figsize=(8,5))
    plt.plot(sizes, log['acc_sm'], marker='o', label='SM')
    plt.plot(sizes, log['acc_ipfp'], marker='s', label='IPFP')
    plt.plot(sizes, log['acc_rrwm'], marker='^', label='RRWM')
    plt.plot(sizes, log['acc_ngm'], marker='x', label='NGM')
    
    plt.xlabel('Number of nodes')
    plt.ylabel('Accuracy')
    plt.title(f'{task_name} Graph: Solver Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'accuracy_{task_name}.png', dpi=300)
    plt.show()

# -------- 绘制 Time 图（SM/IPFP/RRWM） --------
for task_name, log in tasks.items():
    plt.figure(figsize=(8,5))
    plt.plot(sizes, log['time_sm'], marker='o', label='SM')
    plt.plot(sizes, log['time_ipfp'], marker='s', label='IPFP')
    plt.plot(sizes, log['time_rrwm'], marker='^', label='RRWM')
    
    plt.xlabel('Number of nodes')
    plt.ylabel('Time (s)')
    plt.title(f'{task_name} Graph: Solver Time (SM/IPFP/RRWM)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'time_{task_name}.png', dpi=300)
    plt.show()

# -------- 绘制 NGM 时间图（所有任务一张图） --------
plt.figure(figsize=(8,5))
for task_name, log in tasks.items():
    plt.plot(sizes, log['time_ngm'], marker='o', label=f'{task_name} NGM')

plt.xlabel('Number of nodes')
plt.ylabel('Time (s)')
plt.title('NGM Solver Time Comparison Across Tasks')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('time_NGM_all_tasks.png', dpi=300)
plt.show()
