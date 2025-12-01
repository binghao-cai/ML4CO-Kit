import networkx as nx
import numpy as np
from pathlib import Path
from ml4co_kit import GEDTask, Graph, GEDGenerator,GMTask, GMGenerator
from ml4co_kit.generator.graphset.base import GraphFeatureGenerator, GRAPH_FEATURE_TYPE  
from ml4co_kit.solver.genn_astar import GENN_AStarSolver
from ml4co_kit.solver.astar import AStarSolver
from ml4co_kit.solver.rrwm import RRWMSolver
from ml4co_kit.solver.ipfp import IPFPSolver
from ml4co_kit.solver.sm import SMSolver

# P1 = Path("/home/zhanghang/caibinghao/ML4CO-Kit/test_dataset/gm/task/gm_er-small_iso_task.pkl")
# P2 = Path("/home/zhanghang/caibinghao/ML4CO-Kit/test_dataset/gm/task/gm_er-small_ind_task.pkl")
# P3 = Path("/home/zhanghang/caibinghao/ML4CO-Kit/test_dataset/gm/task/gm_er-small_pert_task.pkl")
# P4 = Path("/home/zhanghang/caibinghao/ML4CO-Kit/test_dataset/gm/task/gm_er-large_iso_task.pkl")


# g1 = GraphFeatureGenerator(feature_type=GRAPH_FEATURE_TYPE.UNIFORM)
# g2 = GraphFeatureGenerator(feature_type=GRAPH_FEATURE_TYPE.GAUSSIAN)
# iso_gen_u = GMGenerator(nodes_num_scale=(10,10), nodes_feat_dim_scal=(8,8), edges_feat_dim_scal=(8,8), graph_generate_rule="isomorphic", node_feature_gen=g1, edge_feature_gen=g1)
# ind_gen_u = GMGenerator(nodes_num_scale=(10,10), nodes_feat_dim_scal=(8,8), edges_feat_dim_scal=(8,8), graph_generate_rule="induced_subgraph", keep_ratio=0.7, node_feature_gen=g1, edge_feature_gen=g1)
# pert_gen_u = GMGenerator(nodes_num_scale=(10,10), nodes_feat_dim_scal=(8,8), edges_feat_dim_scal=(8,8), graph_generate_rule="perturbed",perturb_node_features=True, perturb_edge_features=True, node_feature_gen=g1, edge_feature_gen=g1)
# iso_lar_gen_u = GMGenerator(nodes_num_scale=(30,30), nodes_feat_dim_scal=(8,8), edges_feat_dim_scal=(8,8), graph_generate_rule="isomorphic", node_feature_gen=g1, edge_feature_gen=g1)

# task_iso_u = iso_gen_u.generate()
# task_ind_u = ind_gen_u.generate()
# task_pert_u = pert_gen_u.generate()
# task_iso_lar_u = iso_lar_gen_u.generate()
# task_iso_u.to_pickle(P1)
# task_ind_u.to_pickle(P2)
# task_pert_u.to_pickle(P3)
# task_iso_lar_u.to_pickle(P4)
























# nx1 = nx.path_graph(3)
# nx2 = nx.path_graph(4)
# for n in nx1.nodes():
#     nx1.nodes[n]['feature'] = np.array([1,1], dtype=np.float32)
# for n in nx2.nodes():
#     nx2.nodes[n]['feature'] = np.array([1,1], dtype=np.float32)
# for u,v in nx1.edges():
#     nx1.edges[u,v]['feature'] = np.array([1], dtype=np.float32)
# for u,v in nx2.edges():
#     nx2.edges[u,v]['feature'] = np.array([1], dtype=np.float32)
# g1, g2 = Graph(), Graph()
# g1.from_networkx(nx1)
# g2.from_networkx(nx2)
# ged = GEDTask()
# ged.from_data([g1, g2])
# a=ged.build_cost_mat()
# print(a.shape)


# Graph1 = Graph(
#     nodes_num=4,
#     edges_num=6,
#     nodes_feature=np.array([[1],[2],[3],[4]] , dtype=np.float32),
#     edges_feature=np.array([[1],[2],[3],[3],[2],[1]] , dtype=np.float32),
#     edge_index=np.array([[0, 1, 2, 3, 2, 1], [1, 2, 3, 2, 1, 0]] , dtype=np.int32)
# )
# g = GMGenerator()
# g1 = g._induced_subgraph_generate(Graph1, keep_ratio=0.75)
# Graph2 = Graph(
#     nodes_feature=np.array([[1],[3],[2]] , dtype=np.float32),
#     edges_feature=np.array([[1],[1],[2],[2]] , dtype=np.float32),
#     edge_index=np.array([[0, 2, 2, 1], [2, 0, 1, 2]] , dtype=np.int32)
# )
# ref=np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,0]] , dtype=np.int32).ravel()
# ged = GEDTask()
# ged.from_pickle("simp_ged.pkl")
# ged.cost_mat=ged.build_cost_mat()
# print(ged.cost_mat)
# solver = AStarSolver(beam_width=0)
# solver.solve(ged)
# sol = ged.ref_sol.reshape(4,4).T.ravel()
# sss=ged.sol.reshape(4,4)
# print("solver")
# print(sss)
# print("ref")
# print(sol.reshape(4,4).T)
# print(sol)
# print(sol@ ged.cost_mat @ sol.T)
# print(sss.T.ravel() @ ged.cost_mat @ sss.T.ravel())
# gen = GEDGenerator(nodes_num_scale=(5,5),nodes_feat_dim_scal=(4,4),edges_feat_dim_scal=(4,4),graph_generate_rule="isomorphic",node_feature_gen=GraphFeatureGenerator(GRAPH_FEATURE_TYPE.UNIFORM),edge_feature_gen=GraphFeatureGenerator(GRAPH_FEATURE_TYPE.UNIFORM))
# ged=gen.generate()
# solver = AStarSolver()
# solver.solve(ged)
# print(ged.evaluate(ged.sol))
# print("ref")
# print(ged.ref_sol.reshape(20,20))
# print("sol")
# print(ged.sol.reshape(20,20))
# ged.render(save_path="ged_example.png", with_sol=True)
# # ged.to_pickle("my_ged.pkl")
# print("ref cost:", ged.ref_sol.reshape(100,100).T.ravel() @ ged.aff_matrix @ ged.ref_sol.reshape(100,100).T.ravel().T)
# print("sol cost:", ged.sol.reshape(100,100).T.ravel() @ ged.aff_matrix @ ged.sol.reshape(100,100).T.ravel().T)
# g = GEDTask()
# g.from_pickle("my_ged.pkl")
# g1=g.graphs[0]
# g2=g.graphs[1]
# print(g.ref_sol.reshape(5,5))
# # print(g.graphs[0].edge_index)
# # print(g.graphs[1].edge_index)
# K = g.cost_mat
# print(K[5,2])
# print("edge_feature1:")
# print(g1.edges_feature)
# print("edge_feature2:")
# print(g2.edges_feature)