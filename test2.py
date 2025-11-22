import networkx as nx
import numpy as np
import time
from pathlib import Path
from ml4co_kit import GEDTask, Graph, GEDGenerator,GMTask, GMGenerator
from ml4co_kit.generator.graphset.base import GraphFeatureGenerator, GRAPH_FEATURE_TYPE  
from ml4co_kit.solver.genn_astar import GENN_AStarSolver
from ml4co_kit.solver.astar import AStarSolver
from ml4co_kit.solver.rrwm import RRWMSolver
from ml4co_kit.solver.ipfp import IPFPSolver
from ml4co_kit.solver.sm import SMSolver

P1 = Path("/home/zhanghang/caibinghao/ML4CO-Kit/test_dataset/ged/task/ged_er-small_iso_task.pkl")
P2 = Path("/home/zhanghang/caibinghao/ML4CO-Kit/test_dataset/ged/task/ged_er-small_pert_task.pkl")

g1 = GraphFeatureGenerator(feature_type=GRAPH_FEATURE_TYPE.UNIFORM)
g2 = GraphFeatureGenerator(feature_type=GRAPH_FEATURE_TYPE.GAUSSIAN)
start = time.perf_counter()
iso_gen_u = GEDGenerator(nodes_num_scale=(8,8), nodes_feat_dim_scal=(4,4), edges_feat_dim_scal=(4,4), graph_generate_rule="isomorphic", node_feature_gen=g1, edge_feature_gen=g1)
#pert_gen_u = GEDGenerator(nodes_num_scale=(8,8), nodes_feat_dim_scal=(4,4), edges_feat_dim_scal=(4,4), graph_generate_rule="perturbed",perturb_node_features=True, perturb_edge_features=True, node_feature_gen=g1, edge_feature_gen=g1)
end= time.perf_counter()
print(f"Time taken to generate generator: {end - start} seconds")
task_iso_u = iso_gen_u.generate()
#task_pert_u = pert_gen_u.generate()
sol = AStarSolver()
start = time.perf_counter() 
sol.solve(task_iso_u)
end = time.perf_counter() 
print(f"Time taken to solve isomorphic task: {end - start} seconds")

#sol.solve(task_pert_u)
print(task_iso_u.evaluate(task_iso_u.sol))
#print(task_pert_u.evaluate(task_pert_u.sol))
# task_iso_u.to_pickle(P1)
#task_pert_u.to_pickle(P2)
