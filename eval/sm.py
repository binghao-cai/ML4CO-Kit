import time
import pandas
from matplotlib import pyplot as plt
import numpy as np
from ml4co_kit import GraphFeatureGenerator, GMTask, GMGenerator, SMSolver, IPFPSolver, RRWMSolver, AStarSolver, GennAStarSolver, NGMSolver

def gaussian_aff_fn(feat1:np.ndarray, feat2:np.ndarray, sigma:np.floating = 1.0) -> np.ndarray:
         """Gaussian affinity unction"""
         feat1 = np.expand_dims(feat1, axis=1)
         feat2 = np.expand_dims(feat2, axis=0)
         return np.exp(-((feat1-feat2)**2).sum(axis=-1)/sigma)


sm = SMSolver(max_iter=100)
ipfp = IPFPSolver(max_iter=100)
rrwm = RRWMSolver(max_iter=100)
astar = AStarSolver()
genn_astar = GennAStarSolver(device="cuda")
ngm = NGMSolver(device="cuda")

gen= GMGenerator(
            distribution_type='er',
            nodes_num_scale=(10, 10),  
            node_feat_dim_scale=(36,36), 
            edge_feat_dim_scale=(36,36), 
        )
task_act = gen.generate()

#ngm.batch_solve([task_act]) 
genn_astar.batch_solve([task_act])

def test_diff_kr(solver, gen_feat, scale=(0.8, 1),mode="single"):
    print(f"===== Testing kr: {scale} =====")
    acc_list_inner = []
    score_list_inner = []
    time_list_inner = []

    acc_list_gau = []
    score_list_gau = []
    time_list_gau = []
    trial_num = 10
    kr_num = int((scale[1]-scale[0])/0.02) + 1
    ##########  GM er er_prob=0.15 ##########
    for i in np.arange(scale[0], scale[1]+0.02, 0.02):
        if int((i - scale[0])/0.02) % 5 == 0:
            print(f"Evaluating keep ratio: {int((i - scale[0])/0.02)}/{kr_num}")
        acc_count_inner = 0
        score_count_inner = 0
        time_count_inner = 0
        acc_count_gau = 0
        score_count_gau = 0
        time_count_gau = 0
        if solver.solver_type != "ngm":
            for j in range(trial_num):
                gen_inner = GMGenerator(
                    distribution_type='er',
                    nodes_num_scale=(50, 50),  
                    node_feat_dim_scale=(8,8), 
                    edge_feat_dim_scale=(8,8), 
                    graph_generate_rule='induced_subgraph',
                    keep_ratio=i,
                    node_feature_gen=gen_feat,
                    edge_feature_gen=gen_feat,
                )
                
                gen_gau = GMGenerator(
                    distribution_type='er',
                    nodes_num_scale=(50, 50),  
                    node_feat_dim_scale=(8,8), 
                    edge_feat_dim_scale=(8,8), 
                    graph_generate_rule='induced_subgraph',
                    keep_ratio=i,
                    node_aff_fn=gaussian_aff_fn,
                    edge_aff_fn=gaussian_aff_fn,
                    node_feature_gen=gen_feat,
                    edge_feature_gen=gen_feat,
                )

                task_inner = gen_inner.generate()
                task_gau = gen_gau.generate()
                
                time_start = time.time()
                solver.solve(task_inner)
                time_end = time.time()
                time_count_inner += (time_end - time_start)
                
                time_start = time.time()
                solver.solve(task_gau)
                time_end = time.time()
                time_count_gau += (time_end - time_start)
                
                score_inner = task_inner.evaluate(task_inner.sol, mode="score")
                ref_score_inner = task_inner.evaluate(task_inner.ref_sol, mode="score")
                acc_inner = task_inner.evaluate(task_inner.sol, mode="acc")
                score_percent_inner = 100 * score_inner / ref_score_inner if ref_score_inner != 0 else 0 
                
                score_gau = task_gau.evaluate(task_gau.sol, mode="score")
                ref_score_gau = task_gau.evaluate(task_gau.ref_sol, mode="score")
                acc_gau = task_gau.evaluate(task_gau.sol, mode="acc")
                score_percent_gau = 100 * score_gau / ref_score_gau if ref_score_gau != 0 else 0
                
                acc_count_inner += acc_inner
                score_count_inner += score_percent_inner
                acc_count_gau += acc_gau
                score_count_gau += score_percent_gau
        elif solver.solver_type == "ngm":
            task_inner_list = []
            task_gau_list = []
            for j in range(trial_num):
                gen_inner = GMGenerator(
                    distribution_type='er',
                    nodes_num_scale=(50, 50),  
                    node_feat_dim_scale=(8,8), 
                    edge_feat_dim_scale=(8,8), 
                    graph_generate_rule='induced_subgraph',
                    keep_ratio=i,
                    node_feature_gen=gen_feat,
                    edge_feature_gen=gen_feat,
                )
                
                gen_gau = GMGenerator(
                    distribution_type='er',
                    nodes_num_scale=(50, 50),  
                    node_feat_dim_scale=(8,8), 
                    edge_feat_dim_scale=(8,8), 
                    graph_generate_rule='induced_subgraph',
                    keep_ratio=i,
                    node_aff_fn=gaussian_aff_fn,
                    edge_aff_fn=gaussian_aff_fn,
                    node_feature_gen=gen_feat,
                    edge_feature_gen=gen_feat,
                )

                task_inner = gen_inner.generate()
                task_gau = gen_gau.generate()
                task_inner_list.append(task_inner)
                task_gau_list.append(task_gau)
            
            if mode == "batch":
                time_start = time.time()
                solver.batch_solve(task_inner_list)
                time_end = time.time()
                time_count_inner += (time_end - time_start)
                
                time_start = time.time()
                solver.batch_solve(task_gau_list)
                time_end = time.time()
                time_count_gau += (time_end - time_start)
            elif mode == "single":
                for task_inner in task_inner_list:
                    time_start = time.time()
                    solver.batch_solve([task_inner])
                    time_end = time.time()
                    time_count_inner += (time_end - time_start)
                for task_gau in task_gau_list:
                    time_start = time.time()
                    solver.batch_solve([task_gau])
                    time_end = time.time()
                    time_count_gau += (time_end - time_start)
        
            for task_inner in task_inner_list:
                score_inner = task_inner.evaluate(task_inner.sol, mode="score")
                ref_score_inner = task_inner.evaluate(task_inner.ref_sol, mode="score")
                acc_inner = task_inner.evaluate(task_inner.sol, mode="acc")
                score_percent_inner = 100 * score_inner / ref_score_inner if ref_score_inner != 0 else 0 
                
                acc_count_inner += acc_inner
                score_count_inner += score_percent_inner
            for task_gau in task_gau_list:
                score_gau = task_gau.evaluate(task_gau.sol, mode="score")
                ref_score_gau = task_gau.evaluate(task_gau.ref_sol, mode="score")
                acc_gau = task_gau.evaluate(task_gau.sol, mode="acc")
                score_percent_gau = 100 * score_gau / ref_score_gau if ref_score_gau != 0 else 0
                
                acc_count_gau += acc_gau
                score_count_gau += score_percent_gau
        
        avg_acc_inner = acc_count_inner / trial_num
        avg_score_inner = score_count_inner / trial_num
        avg_time_inner = time_count_inner / trial_num
        avg_acc_gau = acc_count_gau / trial_num
        avg_score_gau = score_count_gau / trial_num
        avg_time_gau = time_count_gau / trial_num
        
        acc_list_inner.append(round(avg_acc_inner, 4))
        score_list_inner.append(round(avg_score_inner, 4))
        time_list_inner.append(round(avg_time_inner, 4))
        
        acc_list_gau.append(round(avg_acc_gau, 4))
        score_list_gau.append(round(avg_score_gau, 4))
        time_list_gau.append(round(avg_time_gau, 4))
        
    return (acc_list_inner, score_list_inner, time_list_inner,
            acc_list_gau, score_list_gau, time_list_gau)

def test_diff_std(solver, gen_feat, scale=(0.0, 0.7), nodes_num=50, dim=8, mode="single"):
    print(f"===== Testing std: {scale} =====")
    acc_list_inner = []
    score_list_inner = []
    time_list_inner = []
    acc_list_gau = []
    score_list_gau = []
    time_list_gau = []
    
    trial_num = 10
    std_num = int((scale[1]-scale[0])/0.05) + 1
    ##########  GM er er_prob=0.15 ##########
    for i in np.arange(scale[0], scale[1]+0.05, 0.05):
        if int((i - scale[0])/0.05) % 5 == 0:
            print(f"Evaluating std: {int((i - scale[0])/0.05)}/{std_num}")
        acc_count_inner, acc_count_gau = 0, 0
        score_count_inner, score_count_gau = 0, 0
        time_count_inner, time_count_gau = 0, 0
        print(f"  std: {i} ")
        if solver.solver_type != "ngm":
            for j in range(trial_num):
                gen_inner = GMGenerator(
                    distribution_type='er',
                    nodes_num_scale=(nodes_num, nodes_num),  
                    node_feat_dim_scale=(dim,dim), 
                    edge_feat_dim_scale=(dim,dim), 
                    graph_generate_rule='perturbed',
                    perturb_node_features = True,
                    perturb_edge_features = True,
                    node_feat_noise_std=i,
                    edge_feat_noise_std=i,
                    node_feature_gen=gen_feat,
                    edge_feature_gen=gen_feat,
                )
                
                gen_gau = GMGenerator(
                    distribution_type='er',
                    nodes_num_scale=(nodes_num, nodes_num),  
                    node_feat_dim_scale=(dim,dim), 
                    edge_feat_dim_scale=(dim,dim), 
                    graph_generate_rule='perturbed',
                    perturb_node_features = True,
                    perturb_edge_features = True,
                    node_feat_noise_std=i,
                    edge_feat_noise_std=i,
                    node_aff_fn=gaussian_aff_fn,
                    edge_aff_fn=gaussian_aff_fn,
                    node_feature_gen=gen_feat,
                    edge_feature_gen=gen_feat,
                )

                task_inner = gen_inner.generate()
                task_gau = gen_gau.generate()
                
                time_start = time.time()
                solver.solve(task_inner)
                time_end = time.time()
                time_count_inner += (time_end - time_start)
                
                time_start = time.time()
                solver.solve(task_gau)
                time_end = time.time()
                time_count_gau += (time_end - time_start)
                
                
                score_inner = task_inner.evaluate(task_inner.sol, mode="score")
                ref_score_inner = task_inner.evaluate(task_inner.ref_sol, mode="score")
                acc_inner = task_inner.evaluate(task_inner.sol, mode="acc")
                score_inner_percent = 100 * score_inner / ref_score_inner if ref_score_inner != 0 else 0
                
                score_gau = task_gau.evaluate(task_gau.sol, mode="score")
                ref_score_gau = task_gau.evaluate(task_gau.ref_sol, mode="score")
                acc_gau = task_gau.evaluate(task_gau.sol, mode="acc")
                score_gau_percent = 100 * score_gau / ref_score_gau if ref_score_gau != 0 else 0 
                
                acc_count_inner += acc_inner
                score_count_inner += score_inner_percent
                acc_count_gau += acc_gau
                score_count_gau += score_gau_percent
        elif solver.solver_type == "ngm" or solver.solver_type == "genn_astar":
            task_inner_list = []
            task_gau_list = []
            for j in range(trial_num):
                gen_inner = GMGenerator(
                    distribution_type='er',
                    nodes_num_scale=(nodes_num, nodes_num),  
                    node_feat_dim_scale=(dim,dim), 
                    edge_feat_dim_scale=(dim,dim), 
                    graph_generate_rule='perturbed',
                    perturb_node_features = True,
                    perturb_edge_features = True,
                    node_feat_noise_std=i,
                    edge_feat_noise_std=i,
                    node_feature_gen=gen_feat,
                    edge_feature_gen=gen_feat,
                )
                
                gen_gau = GMGenerator(
                    distribution_type='er',
                    nodes_num_scale=(nodes_num, nodes_num),  
                    node_feat_dim_scale=(dim,dim), 
                    edge_feat_dim_scale=(dim,dim), 
                    graph_generate_rule='perturbed',
                    perturb_node_features = True,
                    perturb_edge_features = True,
                    node_feat_noise_std=i,
                    edge_feat_noise_std=i,
                    node_aff_fn=gaussian_aff_fn,
                    edge_aff_fn=gaussian_aff_fn,
                    node_feature_gen=gen_feat,
                    edge_feature_gen=gen_feat,
                )

                task_inner = gen_inner.generate()
                task_gau = gen_gau.generate()
                task_inner_list.append(task_inner)
                task_gau_list.append(task_gau)
            if mode == "batch":
                time_start = time.time()
                solver.batch_solve(task_inner_list)
                time_end = time.time()
                time_count_inner += (time_end - time_start)
                
                time_start = time.time()
                solver.batch_solve(task_gau_list)
                time_end = time.time()
                time_count_gau += (time_end - time_start)
            elif mode == "single":
                for task_inner in task_inner_list:
                    time_start = time.time()
                    solver.batch_solve([task_inner])
                    time_end = time.time()
                    time_count_inner += (time_end - time_start)
                for task_gau in task_gau_list:
                    time_start = time.time()
                    solver.batch_solve([task_gau])
                    time_end = time.time()
                    time_count_gau += (time_end - time_start)
        
            for task_inner in task_inner_list:
                score_inner = task_inner.evaluate(task_inner.sol, mode="score")
                ref_score_inner = task_inner.evaluate(task_inner.ref_sol, mode="score")
                acc_inner = task_inner.evaluate(task_inner.sol, mode="acc")
                score_percent_inner = 100 * score_inner / ref_score_inner if ref_score_inner != 0 else 0 
                
                acc_count_inner += acc_inner
                score_count_inner += score_percent_inner
                
            for task_gau in task_gau_list:
                score_gau = task_gau.evaluate(task_gau.sol, mode="score")
                ref_score_gau = task_gau.evaluate(task_gau.ref_sol, mode="score")
                acc_gau = task_gau.evaluate(task_gau.sol, mode="acc")
                score_percent_gau = 100 * score_gau / ref_score_gau if ref_score_gau != 0 else 0
                
                acc_count_gau += acc_gau
                score_count_gau += score_percent_gau
                
        avg_acc_inner = acc_count_inner / trial_num
        avg_score_inner = score_count_inner / trial_num
        avg_time_inner = time_count_inner / trial_num
        avg_acc_gau = acc_count_gau / trial_num
        avg_score_gau = score_count_gau / trial_num
        avg_time_gau = time_count_gau / trial_num
        
        acc_list_inner.append(round(avg_acc_inner, 4))
        score_list_inner.append(round(avg_score_inner, 4))
        time_list_inner.append(round(avg_time_inner, 4))
        acc_list_gau.append(round(avg_acc_gau, 4))
        score_list_gau.append(round(avg_score_gau, 4))
        time_list_gau.append(round(avg_time_gau, 4))
        
    return acc_list_inner, score_list_inner, time_list_inner, acc_list_gau, score_list_gau, time_list_gau

def to_txt(file_name, content, mode='w'):
    with open(file_name, mode, encoding='utf-8') as f:
        for line in content:
            f.write(f"{line}\n")

def visualize(graph_num, sup_title, scale, subname,xname, yname, data_, file_name):
    plt.figure(figsize=(15, 5))
    plt.suptitle(sup_title, fontsize=16)
    plt.subplots_adjust(left=0.07, right=0.93, wspace=0.3)
    for g in range(graph_num):
        plt.subplot(1, graph_num, g+1)
        plt.plot(scale, data_[g])
        plt.title(f'{subname[g]}')
        plt.xlabel(f'{xname[g]}')
        plt.ylabel(f'{yname[g]}')
    
    plt.savefig(f'eval/{file_name}.png')
    plt.close()

def test_ind(solver, file_txt, file_img_name, mode="single"):
    print(f"===== Testing solver: {solver.solver_type} =====")
    gen_feat = GraphFeatureGenerator(feature_type='uniform')
        
    acc_list_inner, score_list_inner, time_list_inner, \
    acc_list_gau, score_list_gau, time_list_gau = test_diff_kr(solver, gen_feat, scale=(0.8, 1), mode=mode)
    
    contents = [
        f"=== Solver: GM_er_unform_ind: nodes_num=50, dim = 8, er_prob=0.15 ===",
        f"=== Solver: {solver.solver_type} ===",
    ]
    
    contents.append(f"--- Inner ---")
    for kr in np.arange(0.8, 1.02, 0.02):
        contents.append(f"Keep Ratio: {kr:.2f}, Acc: {acc_list_inner[int((kr-0.8)/0.02)]:.4f}, Score(%): {score_list_inner[int((kr-0.8)/0.02)]:.4f}, Time(s): {time_list_inner[int((kr-0.8)/0.02)]:.4f}")
        
    contents.append("")
    
    contents.append(f"--- Gaussian ---")
    for kr in np.arange(0.8, 1.02, 0.02):
        contents.append(f"Keep Ratio: {kr:.2f}, Acc: {acc_list_gau[int((kr-0.8)/0.02)]:.4f}, Score(%): {score_list_gau[int((kr-0.8)/0.02)]:.4f}, Time(s): {time_list_gau[int((kr-0.8)/0.02)]:.4f}")
    
    to_txt("eval/"+file_txt, contents, mode='w')
    
    scale = np.arange(0.8, 1.02, 0.02)
    visualize(
        graph_num=3,
        sup_title=f'Solver: {solver.solver_type} on GM_er_inner_ind',
        scale=scale,
        subname=['Accuracy', 'Score(%)', 'Time(s)'],
        xname=['Keep Ratio', 'Keep Ratio', 'Keep Ratio'],
        yname=['Accuracy', 'Score(%)', 'Time(s)'],
        data_=[acc_list_inner, score_list_inner, time_list_inner],
        file_name=file_img_name+'_inner'
    )
    visualize(
        graph_num=3,
        sup_title=f'Solver: {solver.solver_type} on GM_er_gau_ind',
        scale=scale,
        subname=['Accuracy', 'Score(%)', 'Time(s)'],
        xname=['Keep Ratio', 'Keep Ratio', 'Keep Ratio'],
        yname=['Accuracy', 'Score(%)', 'Time(s)'],
        data_=[acc_list_gau, score_list_gau, time_list_gau],
        file_name=file_img_name+'_gau'
    )
    
    print(f"===== Finished testing solver: {solver.solver_type} =====\n")
    
def test_pert(solver, file_txt, file_img_name, mode="single"):
    print(f"===== Testing solver: {solver.solver_type} =====")
    gen_feat = GraphFeatureGenerator(feature_type='uniform')
        
    avg_acc_list_inner, avg_score_list_inner, avg_time_list_inner, \
    avg_acc_list_gau, avg_score_list_gau, avg_time_list_gau = test_diff_std(solver, gen_feat, scale=(0.0, 0.7), mode=mode)
    scale = np.arange(0.0, 0.75, 0.05)
    
    visualize(
        graph_num=3,
        sup_title=f'Solver: {solver.solver_type} on GM_er_inner_pert',
        scale=scale,
        subname=['Accuracy', 'Score(%)', 'Time(s)'],
        xname=['Std Dev', 'Std Dev', 'Std Dev'],
        yname=['Accuracy', 'Score(%)', 'Time(s)'],
        data_=[avg_acc_list_inner, avg_score_list_inner, avg_time_list_inner],
        file_name=file_img_name+'_inner'
    )
    visualize(
        graph_num=3,
        sup_title=f'Solver: {solver.solver_type} on GM_er_gau_pert',
        scale=scale,    
        subname=['Accuracy', 'Score(%)', 'Time(s)'],
        xname=['Std Dev', 'Std Dev', 'Std Dev'],
        yname=['Accuracy', 'Score(%)', 'Time(s)'],
        data_=[avg_acc_list_gau, avg_score_list_gau, avg_time_list_gau],
        file_name=file_img_name+'_gau'
    )
    
    
    contents = [
        f"=== Solver: GM_er_unform_pert: nodes_num=50, dim = 8, er_prob=0.15 ===",
        f"=== Solver: {solver.solver_type} ===",
    ]
    
    contents.append(f"--- Inner ---")
    for std in np.arange(0.0, 0.75, 0.05):
        contents.append(f"Std Dev: {std:.2f}, Acc: {avg_acc_list_inner[int(std/0.05)]:.4f}, Score(%): {avg_score_list_inner[int(std/0.05)]:.4f}, Time(s): {avg_time_list_inner[int(std/0.05)]:.4f}")
        
    contents.append("")
    contents.append(f"--- Gaussian ---")
    for std in np.arange(0.0, 0.75, 0.05):
        contents.append(f"Std Dev: {std:.2f}, Acc: {avg_acc_list_gau[int(std/0.05)]:.4f}, Score(%): {avg_score_list_gau[int(std/0.05)]:.4f}, Time(s): {avg_time_list_gau[int(std/0.05)]:.4f}")
        
    to_txt("eval/"+file_txt, contents, mode='w')
    
    
    print(f"===== Finished testing solver: {solver.solver_type} =====\n")
    
def test_diff_num(solver, gen, dim=8, scale=(10, 100), step=10, mode="single"):
    time_list_inner = []
    time_list_gau = []
    acc_list_inner = []
    acc_list_gau = []
    score_list_inner = []
    score_list_gau = []
    bat = solver.solver_type == "genn_astar"
    nums_trials = 10
    for i in range(scale[0], scale[1]+1, step):
        print(f"Evaluating nodes num: {i}")
        time_inner, time_gau, acc_inner, acc_gau, score_inner, score_gau = 0, 0, 0, 0, 0, 0
        task_inner_list, task_gau_list = [], []
        for j in range(nums_trials):
            gen_inner = GMGenerator(
                distribution_type='er',
                er_prob=0.5,
                nodes_num_scale=(i, i),  
                node_feat_dim_scale=(dim,dim), 
                edge_feat_dim_scale=(dim,dim), 
                graph_generate_rule='isomorphic',
                node_feature_gen=gen,
                edge_feature_gen=gen,
            )
            
            gen_gau = GMGenerator(
                distribution_type='er',
                er_prob=0.5,
                nodes_num_scale=(i, i),  
                node_feat_dim_scale=(dim,dim), 
                edge_feat_dim_scale=(dim,dim), 
                graph_generate_rule='isomorphic',
                node_aff_fn=gaussian_aff_fn,
                edge_aff_fn=gaussian_aff_fn,
                node_feature_gen=gen,
                edge_feature_gen=gen,
            )
            
            task_inner = gen_inner.generate()
            task_gau = gen_gau.generate()
            
            task_inner_list.append(task_inner)
            task_gau_list.append(task_gau)
        if mode == "batch":
            time_start = time.time()
            solver.batch_solve(task_inner_list)
            time_end = time.time()
            time_inner = time_end - time_start
            
            time_start = time.time()
            solver.batch_solve(task_gau_list)
            time_end = time.time()
            time_gau = time_end - time_start
        
        elif mode == "single":
            for task_inner in task_inner_list:
                if bat:
                    time_start = time.time()
                    solver.batch_solve([task_inner])
                    time_end = time.time()
                    time_inner += (time_end - time_start)
                else:
                    time_start = time.time()
                    solver.solve(task_inner)
                    time_end = time.time()
                    time_inner += (time_end - time_start)
            for task_gau in task_gau_list:
                if bat:
                    time_start = time.time()
                    solver.batch_solve([task_gau])
                    time_end = time.time()
                    time_gau += (time_end - time_start)
                else:
                    time_start = time.time()
                    solver.solve(task_gau)
                    time_end = time.time()
                    time_gau += (time_end - time_start)
        for task_inner, task_gau in zip(task_inner_list, task_gau_list):    
            acc_inner += task_inner.evaluate(task_inner.sol, mode="acc")
            score_inner += 100 * task_inner.evaluate(task_inner.sol, mode="score") / task_inner.evaluate(task_inner.ref_sol, mode="score")
            acc_gau += task_gau.evaluate(task_gau.sol, mode="acc")
            score_gau += 100 * task_gau.evaluate(task_gau.sol, mode="score") / task_gau.evaluate(task_gau.ref_sol, mode="score")        
            
        time_list_inner.append(round(time_inner / nums_trials, 4))
        time_list_gau.append(round(time_gau / nums_trials, 4))
        acc_list_inner.append(round(acc_inner / nums_trials, 4))
        acc_list_gau.append(round(acc_gau / nums_trials, 4))
        score_list_inner.append(round(score_inner / nums_trials, 4))
        score_list_gau.append(round(score_gau / nums_trials, 4))   
        
    return time_list_inner, time_list_gau, acc_list_inner, acc_list_gau, score_list_inner, score_list_gau
    
def astar_test(dim=8, std_scale=(0.0, 0.45)):
    gen = GraphFeatureGenerator(feature_type='uniform')
    
    # time_list_inner,time_list_gau,acc_list_inner, acc_list_gau,score_list_inner,score_list_gau = test_diff_num(astar, gen, scale=nodes_scale, step=1)   
    
    # visualize(
    #     graph_num=3,
    #     sup_title=f'Solver: AStarSolver on GM_er_inner_isomorphic',
    #     scale=np.arange(nodes_scale[0], nodes_scale[1] + 1),
    #     subname=['Accuracy', 'Score(%)', 'Time(s)'],
    #     xname=['Nodes Num', 'Nodes Num', 'Nodes Num'],
    #     yname=['Accuracy', 'Score(%)', 'Time(s)'],
    #     data_=[acc_list_inner, score_list_inner, time_list_inner],
    #     file_name='astar_iso_inner'
    # )
    # visualize(
    #     graph_num=3,
    #     sup_title=f'Solver: AStarSolver on GM_er_gau_isomorphic',
    #     scale=np.arange(nodes_scale[0], nodes_scale[1] + 1),
    #     subname=['Accuracy', 'Score(%)', 'Time(s)'],
    #     xname=['Nodes Num', 'Nodes Num', 'Nodes Num'],
    #     yname=['Accuracy', 'Score(%)', 'Time(s)'],
    #     data_=[acc_list_gau, score_list_gau, time_list_gau],
    #     file_name='astar_iso_gau'
    # )
    
    # contents = [
    #     f"=== Solver: GM_er_unform_isomorphic: nodes_num=4-10, dim = 8, er_prob=0.5 ===",
    #     f"=== Solver: AStarSolver ===",
    # ]
    # contents.append(f"--- Inner ---")
    # for idx, nodes_num in enumerate(range(nodes_scale[0], nodes_scale[1] + 1)):
    #     contents.append(f"Nodes Num: {nodes_num}, Acc: {acc_list_inner[idx]:.4f}, Score(%): {score_list_inner[idx]:.4f}, Time(s): {time_list_inner[idx]:.4f}")
    # contents.append("")
    # contents.append(f"--- Gaussian ---")
    # for idx, nodes_num in enumerate(range(nodes_scale[0], nodes_scale[1] + 1)):
    #     contents.append(f"Nodes Num: {nodes_num}, Acc: {acc_list_gau[idx]:.4f}, Score(%): {score_list_gau[idx]:.4f}, Time(s): {time_list_gau[idx]:.4f}")
    # to_txt("eval/astar_iso.txt", contents, mode='w')
    
    
    acc_list_inner,score_list_inner, time_list_inner,acc_list_gau,score_list_gau, time_list_gau = test_diff_std(astar, gen, nodes_num=10, dim=dim, scale=std_scale)   
    scale = np.arange(std_scale[0], std_scale[1] + 0.05, 0.05)
    visualize(
        graph_num=3,
        sup_title=f'Solver: AStarSolver on GM_er_inner_perturbed',
        scale=scale,
        subname=['Accuracy', 'Score(%)', 'Time(s)'],
        xname=['Std Dev', 'Std Dev', 'Std Dev'],
        yname=['Accuracy', 'Score(%)', 'Time(s)'],
        data_=[acc_list_inner, score_list_inner, time_list_inner],
        file_name='astar_pert_inner'
    )
    visualize(
        graph_num=3,
        sup_title=f'Solver: AStarSolver on GM_er_gau_perturbed',
        scale=scale,
        subname=['Accuracy', 'Score(%)', 'Time(s)'],
        xname=['Std Dev', 'Std Dev', 'Std Dev'],
        yname=['Accuracy', 'Score(%)', 'Time(s)'],
        data_=[acc_list_gau, score_list_gau, time_list_gau],
        file_name='astar_pert_gau'
    )
    
    contents = [
        f"=== Solver: GM_er_unform_perturbed: nodes_num=9, dim = 8, er_prob=0.5 ===",
        f"=== Solver: AStarSolver ===",
    ]
    contents.append(f"--- Inner ---")
    for std in np.arange(0.0, 0.5, 0.05):
        contents.append(f"Std Dev: {std:.2f}, Acc: {acc_list_inner[int(20*std)]:.4f}, Score(%): {score_list_inner[int(20*std)]:.4f}, Time(s): {time_list_inner[int(20*std)]:.4f}")
    contents.append("")
    contents.append(f"--- Gaussian ---")
    for std in np.arange(0.0, 0.7, 0.05):
        contents.append(f"Std Dev: {std:.2f}, Acc: {acc_list_gau[int(20*std)]:.4f}, Score(%): {score_list_gau[int(20*std)]:.4f}, Time(s): {time_list_gau[int(20*std)]:.4f}")
    to_txt("eval/astar_pert.txt", contents, mode='w')
    
    print("Finished AStarSolver on perturbed graphs\n")    
 
 
def genn_astar_test_pert(file_txt, file_img_name, dim=36, std_scale=(0.0, 0.45), mode="single"):
    gen = GraphFeatureGenerator(feature_type='uniform')
    acc_list_inner,score_list_inner, time_list_inner,acc_list_gau,score_list_gau, time_list_gau = test_diff_std(astar, gen, nodes_num=10, dim=dim, scale=std_scale, mode=mode)   
    scale = np.arange(std_scale[0], std_scale[1] + 0.05, 0.05)
    visualize(
        graph_num=3,
        sup_title=f'Solver: GennAStarSolver on GM_er_inner_perturbed',
        scale=scale,
        subname=['Accuracy', 'Score(%)', 'Time(s)'],
        xname=['Std Dev', 'Std Dev', 'Std Dev'],
        yname=['Accuracy', 'Score(%)', 'Time(s)'],
        data_=[acc_list_inner, score_list_inner, time_list_inner],
        file_name=f'{file_img_name}_inner'
    )
    visualize(
        graph_num=3,
        sup_title=f'Solver: GennAStarSolver on GM_er_gau_perturbed',
        scale=scale,
        subname=['Accuracy', 'Score(%)', 'Time(s)'],
        xname=['Std Dev', 'Std Dev', 'Std Dev'],
        yname=['Accuracy', 'Score(%)', 'Time(s)'],
        data_=[acc_list_gau, score_list_gau, time_list_gau],
        file_name=f'{file_img_name}_gau'
    )
    contents = [
        f"=== Solver: GM_er_unform_perturbed: nodes_num=10, dim = {dim}, er_prob=0.5 ===",
        f"=== Solver: GennAStarSolver ===",
    ]
    contents.append(f"--- Inner ---")
    for std in np.arange(std_scale[0], std_scale[1] + 0.05, 0.05):
        contents.append(f"Std Dev: {std:.2f}, Acc: {acc_list_inner[int(20*std)]:.4f}, Score(%): {score_list_inner[int(20*std)]:.4f}, Time(s): {time_list_inner[int(20*std)]:.4f}")
    contents.append("")
    contents.append(f"--- Gaussian ---")
    for std in np.arange(std_scale[0], std_scale[1] + 0.05, 0.05):
        contents.append(f"Std Dev: {std:.2f}, Acc: {acc_list_gau[int(20*std)]:.4f}, Score(%): {score_list_gau[int(20*std)]:.4f}, Time(s): {time_list_gau[int(20*std)]:.4f}")
    to_txt("eval/" + file_txt, contents, mode='w')
    
    print("Finished GennAStarSolver on perturbed graphs\n")
     
def genn_astar_test_iso(file_txt, file_img_name, dim=36, nodes_scale=(4, 15), step=1):
    gen = GraphFeatureGenerator(feature_type='uniform')
    time_list_inner,time_list_gau,acc_list_inner, acc_list_gau,score_list_inner,score_list_gau = test_diff_num(genn_astar, gen, dim=dim, scale=nodes_scale, step=step)   
    
    visualize(
        graph_num=3,
        sup_title=f'Solver: GennAStarSolver on GM_er_inner_isomorphic',
        scale=np.arange(nodes_scale[0], nodes_scale[1] + 1, step),
        subname=['Accuracy', 'Score(%)', 'Time(s)'],
        xname=['Nodes Num', 'Nodes Num', 'Nodes Num'],
        yname=['Accuracy', 'Score(%)', 'Time(s)'],
        data_=[acc_list_inner, score_list_inner, time_list_inner],
        file_name=f'{file_img_name}_inner'
    )
    visualize(
        graph_num=3,
        sup_title=f'Solver: GennAStarSolver on GM_er_gau_isomorphic',
        scale=np.arange(nodes_scale[0], nodes_scale[1] + 1, step),
        subname=['Accuracy', 'Score(%)', 'Time(s)'],
        xname=['Nodes Num', 'Nodes Num', 'Nodes Num'],
        yname=['Accuracy', 'Score(%)', 'Time(s)'],
        data_=[acc_list_gau, score_list_gau, time_list_gau],
        file_name=f'{file_img_name}_gau'
    )
    
    contents = [
        f"=== Solver: GM_er_unform_isomorphic: nodes_num={nodes_scale[0]}-{nodes_scale[1]}, dim = {dim}, er_prob=0.5 ===",
        f"=== Solver: GennAStarSolver ===",
    ]
    contents.append(f"--- Inner ---")
    for idx, nodes_num in enumerate(range(nodes_scale[0], nodes_scale[1] + 1, step)):
        contents.append(f"Nodes Num: {nodes_num}, Acc: {acc_list_inner[idx]:.4f}, Score(%): {score_list_inner[idx]:.4f}, Time(s): {time_list_inner[idx]:.4f}")
    contents.append("")
    contents.append(f"--- Gaussian ---")
    for idx, nodes_num in enumerate(range(nodes_scale[0], nodes_scale[1] + 1, step)):
        contents.append(f"Nodes Num: {nodes_num}, Acc: {acc_list_gau[idx]:.4f}, Score(%): {score_list_gau[idx]:.4f}, Time(s): {time_list_gau[idx]:.4f}")
    
    to_txt("eval/" + file_txt, contents, mode='w')     
    
    
if __name__ == "__main__":
    # test_ind(sm, 'sm_ind_log.txt', 'sm_ind_img')
    # test_ind(ipfp, 'ipfp_ind_log.txt', 'ipfp_ind_img')
    # test_ind(rrwm, 'rrwm_ind_log.txt', 'rrwm_ind_img')
    # test_ind(ngm, 'ngm_ind_sin_log.txt', 'ngm_ind_sin_img')
    # test_ind(ngm, 'ngm_ind_bat_log.txt', 'ngm_ind_bat_img', mode="batch")
    
    # test_pert(sm, 'sm_pert_log.txt', 'sm_pert_img')
    # test_pert(ipfp, 'ipfp_pert_log.txt', 'ipfp_pert_img')
    # test_pert(rrwm, 'rrwm_pert_log.txt', 'rrwm_pert_img')
    # test_pert(ngm, 'ngm_pert_sin_log.txt', 'ngm_pert_sin_img')
    # test_pert(ngm, 'ngm_pert_bat_log.txt', 'ngm_pert_bat_img', mode="batch")
    # genn_astar_test_pert("genn_astar_cpu_sin_pert_log.txt", "genn_astar_cpu_sin_pert_img", dim=36, std_scale=(0.0, 0.45), mode="single")
    #genn_astar_test_pert("genn_astar_gpu_bat_pert_log.txt", "genn_astar_gpu_bat_pert_img", dim=36, std_scale=(0.0, 0.4), mode="batch")
    #genn_astar_test_pert("genn_astar_cpu_bat_pert_log.txt", "genn_astar_cpu_bat_pert_img", dim=36, std_scale=(0.0, 0.45), mode="batch")
    # genn_astar_test_iso("genn_astar_cpu_iso_log.txt", "genn_astar_iso_img", dim=36, nodes_scale=(5, 25), step=1)
    #genn_astar_test_iso("genn_astar_gpu_iso_log.txt", "genn_astar_gpu_iso_img", dim=36, nodes_scale=(5, 25), step=1)
    
    # 按 Keep Ratio 0.80 ~ 1.00 步长 0.02
    a=1