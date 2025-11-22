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

def cost_function(feat1, feat2):
     return -1+2 * ((feat1[:, None, :] == feat2[None, :, :]).any(axis=2).astype(np.float32))
def e_cost_function(feat1, feat2):
     return -1+2 * ((feat1[:, None, :] != feat2[None, :, :]).any(axis=2).astype(np.float32))
def ins_cost_function(feat):
     return 2 * np.ones((feat.shape[0],), dtype=np.float32)
def del_cost_function(feat):
    return 2 * np.ones((feat.shape[0],), dtype=np.float32)


P1 = Path("/home/zhanghang/caibinghao/ML4CO-Kit/test_dataset/ged/task/ged_er-small_iso_task.pkl")
P2 = Path("/home/zhanghang/caibinghao/ML4CO-Kit/test_dataset/ged/task/ged_er-small_pert_task.pkl")

g1 = GraphFeatureGenerator(feature_type=GRAPH_FEATURE_TYPE.UNIFORM)
g2 = GraphFeatureGenerator(feature_type=GRAPH_FEATURE_TYPE.GAUSSIAN)
iso_gen_u = GEDGenerator(nodes_num_scale=(8,8), nodes_feat_dim_scal=(36,36), edges_feat_dim_scal=(36,36), graph_generate_rule="isomorphic", node_feature_gen=g1, edge_feature_gen=g1)
task_iso_u = iso_gen_u.generate()
# task_iso_u.cost_mat = task_iso_u.build_cost_mat(node_subst_cost=cost_function,
#                                                 node_ins_cost=ins_cost_function,
#                                                 node_del_cost=del_cost_function,
#                                                 edge_subst_cost=e_cost_function,
#                                                 edge_ins_cost=ins_cost_function,
#                                                 edge_del_cost=del_cost_function)
# print(task_iso_u.ref_sol)
# print("cost")
# print(task_iso_u.cost_mat)
sol = GENN_AStarSolver()
sol.batch_solve([task_iso_u]) 
# node_feat1 = task_iso_u.graphs[0].nodes_feature
# node_feat2 = task_iso_u.graphs[1].nodes_feature
# edge_feat1 = task_iso_u.graphs[0].edges_feature
# edge_feat2 = task_iso_u.graphs[1].edges_feature
# print(edge_feat1.shape)
# print(edge_feat2.shape)
# print(node_feat1.shape)
# print(node_feat2.shape)
# map = task_iso_u.ref_sol[:3, :3].argmax(axis=1)
# # print(map.shape)
# assert (node_feat1 == node_feat2[map]).all()
# assert (edge_feat1 == edge_feat2).all()
print(task_iso_u.evaluate(task_iso_u.sol))
print(task_iso_u.evaluate(task_iso_u.ref_sol))
refsol = task_iso_u.ref_sol
task_iso_u.ref_sol=None
print(task_iso_u.evaluate(task_iso_u.sol))
print(task_iso_u.evaluate(refsol))
print(refsol)
print(task_iso_u.sol)
