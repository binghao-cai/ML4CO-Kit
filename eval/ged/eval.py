import time
from matplotlib import pyplot as plt
import numpy as np
from ml4co_kit import GraphFeatureGenerator, GEDTask, GEDGenerator, AStarSolver, GennAStarSolver




astar = AStarSolver()
genn_astar_gpu = GennAStarSolver(device="cuda")
genn_astar_cpu = GennAStarSolver(device="cpu")

gen= GEDGenerator(
            distribution_type='er',
            nodes_num_scale=(10, 10),  
            node_feat_dim_scale=(36,36), 
            edge_feat_dim_scale=(36,36), 
        )
task_act = gen.generate()

#ngm.batch_solve([task_act]) 
genn_astar_gpu.batch_solve([task_act])
genn_astar_cpu.batch_solve([task_act])

def test_diff_std(solver, gen_feat, scale=(0.0, 0.5), step=0.05,nodes_num=10, dim=36, mode="single"):
    print(f"===== Testing std: {scale} =====")
    acc_list = []
    time_list = []
    
    trial_num = 10
    std_num = int((scale[1]-scale[0])/step) + 1
    ##########  GM er er_prob=0.15 ##########
    for i in np.arange(scale[0], scale[1]+step, step):
        if int((i - scale[0])/step) % 5 == 0:
            print(f"Evaluating std: {int((i - scale[0])/step)}/{std_num}")
        acc_count, score_count, time_count = 0, 0, 0
        print(f"  std: {i} ")
        if solver.solver_type == "astar":
            for j in range(trial_num):
                print(f"    Trial: {j+1}/{trial_num} ")
                gen = GEDGenerator(
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
                
                task = gen.generate()
                
                time_start = time.time()
                solver.solve(task)
                time_end = time.time()
                time_count += (time_end - time_start)
                
                acc = task.evaluate(task.sol, mode="acc")
                          
                acc_count += acc
                
        elif  solver.solver_type == "genn_astar":
            task_list = []
            for j in range(trial_num):
                gen = GEDGenerator(
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
            
                task = gen.generate()
                task_list.append(task)
                
            if mode == "batch":
                time_start = time.time()
                solver.batch_solve(task_list)
                time_end = time.time()
                time_count += (time_end - time_start)
                
            elif mode == "single":
                cnt=0
                for task_inner in task_list:
                    print(f"    Trial: {cnt+1}/{trial_num} ")
                    cnt += 1
                    time_start = time.time()
                    solver.batch_solve([task_inner])
                    time_end = time.time()
                    time_count += (time_end - time_start)
                   
            for task in task_list:
                acc = task.evaluate(task.sol, mode="acc")
                acc_count += acc
                
        avg_acc = acc_count / trial_num
        avg_time = time_count / trial_num
        
        acc_list.append(round(avg_acc, 4))
        time_list.append(round(avg_time, 4))
        
    return acc_list, time_list

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
    
    plt.savefig(f'eval/ged/{file_name}.png')
    plt.close()

def test_pert(solver, file_txt, file_img_name, scale=(0.0,0.45), step=0.05, mode="single"):
    print(f"===== Testing solver: {solver.solver_type} =====")
    gen_feat = GraphFeatureGenerator(feature_type='uniform')
        
    avg_acc_list, avg_time_list = test_diff_std(solver, gen_feat, scale=scale, step=step, mode=mode)
    scale_arr = np.arange(scale[0], scale[1] + step, step)
    
    visualize(
        graph_num=2,
        sup_title=f'Solver: {solver.solver_type} on GM_er_inner_pert',
        scale=scale_arr,
        subname=['Accuracy', 'Time(s)'],
        xname=['Std Dev', 'Std Dev'],
        yname=['Accuracy', 'Time(s)'],
        data_=[avg_acc_list, avg_time_list],
        file_name=file_img_name+'_inner'
    )
    
    
    contents = [
        f"=== Solver: GM_er_unform_pert: nodes_num=10, dim = 36, er_prob=0.15 ===",
        f"=== Solver: {solver.solver_type} ===",
    ]
    
    for idx, std in enumerate(scale_arr):
        contents.append(f"Std Dev: {std:.2f}, Acc: {avg_acc_list[idx]:.4f}, Time(s): {avg_time_list[idx]:.4f}")
        

    to_txt("eval/ged/"+file_txt, contents, mode='w')
    
    
    print(f"===== Finished testing solver: {solver.solver_type} =====\n")
    
def test_diff_num(solver,gen_feat, dim=36, scale=(5, 25), step=1, mode="single"):
    time_list = []
    acc_list = []
    bat = solver.solver_type == "genn_astar"
    nums_trials = 10
    for i in range(scale[0], scale[1]+1, step):
        print(f"Evaluating nodes num: {i}")
        time_, acc , score = 0, 0, 0
        task_list = []
        for j in range(nums_trials):
            gen = GEDGenerator(
                distribution_type='er',
                er_prob=0.5,
                nodes_num_scale=(i, i),  
                node_feat_dim_scale=(dim,dim), 
                edge_feat_dim_scale=(dim,dim), 
                graph_generate_rule='isomorphic',
                node_feature_gen=gen_feat,
                edge_feature_gen=gen_feat,
            )
            
            task = gen.generate()
            
            task_list.append(task)
        if mode == "batch":
            time_start = time.time()
            solver.batch_solve(task_list)
            time_end = time.time()
            time_ = time_end - time_start
            
        
        elif mode == "single":
            for task in task_list:
                if bat:
                    time_start = time.time()
                    solver.batch_solve([task])
                    time_end = time.time()
                    time_ += (time_end - time_start)
                else:
                    time_start = time.time()
                    solver.solve(task)
                    time_end = time.time()
                    time_ += (time_end - time_start)
        for task in task_list:    
            acc += task.evaluate(task.sol, mode="acc")
            score += task.evaluate(task.sol, mode="cost")        
            
        time_list.append(round(time_ / nums_trials, 4))
        acc_list.append(round(acc / nums_trials, 4))

        
    return time_list, acc_list

def test_iso(solver, file_txt, file_img_name, dim=36, nodes_scale=(5, 25), step=1, mode="single"):
    print(f"===== Testing solver: {solver.solver_type} =====")
    
    gen_feat = GraphFeatureGenerator(feature_type='uniform')
    avg_time_list, avg_acc_list= test_diff_num(solver,gen_feat=gen_feat, dim=dim, scale=nodes_scale, step=step, mode=mode)
    nodes_num = list(range(nodes_scale[0], nodes_scale[1]+1, step))

    visualize(
        graph_num=2,
        sup_title=f'Solver: {solver.solver_type} on GM_er_isomorphic',
        scale=nodes_num,
        subname=['Time(s)', 'Accuracy'],
        xname=['Nodes Num', 'Nodes Num'],
        yname=['Time(s)', 'Accuracy'],
        data_=[avg_time_list, avg_acc_list],
        file_name=file_img_name
    )
    
    contents = [
        f"=== Solver: GM_er_isomorphic: dim = {dim}, er_prob=0.5 ===",
        f"=== Solver: {solver.solver_type} ===",
    ]
    
    for idx, n in enumerate(nodes_num):
        contents.append(f"Nodes Num: {n}, Time(s): {avg_time_list[idx]:.4f}, Acc: {avg_acc_list[idx]:.4f}")
        
    to_txt("eval/ged/"+file_txt, contents, mode='w')
    
    print(f"===== Finished testing solver: {solver.solver_type} =====\n")
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
    #test_pert(astar, 'astar_pert_log.txt', 'astar_pert_img', mode="single", scale=(0.0, 0.12), step=0.01)
    #test_pert(solver=genn_astar_cpu, file_txt="genn_astar_cpu_pert_log.txt", file_img_name="genn_astar_cpu_pert_img", scale=(0.0, 0.05), step=0.005, mode="single")
    # test_pert(solver=genn_astar_gpu, file_txt="genn_astar_gpu_pert_log.txt", file_img_name="genn_astar_gpu_pert_img",  scale=(0.0, 0.12), step=0.01, mode="single")
    test_iso(astar, file_txt="astar_iso_log.txt", file_img_name="astar_iso_img", dim=36, nodes_scale=(5, 15), step=1, mode="single")
    test_iso(genn_astar_cpu, file_txt="genn_astar_cpu_iso_log.txt", file_img_name="genn_astar_cpu_iso_img", dim=36, nodes_scale=(5, 25), step=1, mode="single")
    test_iso(genn_astar_gpu, file_txt="genn_astar_gpu_iso_log.txt", file_img_name="genn_astar_gpu_iso_img", dim=36, nodes_scale=(5, 25), step=1, mode="single")