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
from ml4co_kit.wrapper.gm import GMWrapper
from ml4co_kit.wrapper.ged import GEDWrapper

# P1 = Path("/home/zhanghang/caibinghao/ML4CO-Kit/test_dataset/gm/wrapper/gm_er-small_iso_4ins.txt")
# P2 = Path("/home/zhanghang/caibinghao/ML4CO-Kit/test_dataset/gm/wrapper/gm_er-small_iso_4ins.pkl")
# P3 = Path("/home/zhanghang/caibinghao/ML4CO-Kit/test_dataset/gm/wrapper/gm_er-small_ind_4ins.txt")
# P4 = Path("/home/zhanghang/caibinghao/ML4CO-Kit/test_dataset/gm/wrapper/gm_er-small_ind_4ins.pkl")
P5 = Path("/home/zhanghang/caibinghao/ML4CO-Kit/test_dataset/gm/wrapper/gm_er_iso_genn_astar_4ins.txt")
P6 = Path("/home/zhanghang/caibinghao/ML4CO-Kit/test_dataset/gm/wrapper/gm_er_iso_genn_astar_4ins.pkl")

g1 = GraphFeatureGenerator(feature_type=GRAPH_FEATURE_TYPE.UNIFORM)
g2 = GraphFeatureGenerator(feature_type=GRAPH_FEATURE_TYPE.GAUSSIAN)
iso_gen_u = GMGenerator(nodes_num_scale=(10,10), nodes_feat_dim_scal=(8,8), edges_feat_dim_scal=(8,8), graph_generate_rule="isomorphic", node_feature_gen=g1, edge_feature_gen=g1)
ind_gen_u = GMGenerator(nodes_num_scale=(10,10), nodes_feat_dim_scal=(8,8), edges_feat_dim_scal=(8,8), graph_generate_rule="induced_subgraph", node_feature_gen=g1, edge_feature_gen=g1)
astar_iso_gen_u = GMGenerator(nodes_num_scale=(10,10), nodes_feat_dim_scal=(36,36), edges_feat_dim_scal=(36,36), graph_generate_rule="isomorphic", node_feature_gen=g1, edge_feature_gen=g1)
solver = RRWMSolver()
# w1 = GMWrapper()
# w2 = GMWrapper()
w3 = GMWrapper()
# w1.generate(generator=iso_gen_u, solver=solver, num_samples=4, batch_size=1)
# w2.generate(generator=ind_gen_u, solver=solver, num_samples=4, batch_size=1)
w3.generate(generator=astar_iso_gen_u, solver=solver, num_samples=4, batch_size=1)
# w1.to_txt(file_path=P1)
# w1.to_pickle(file_path=P2)
# w2.to_txt(file_path=P3)
# w2.to_pickle(file_path=P4)
w3.to_txt(file_path=P5)
w3.to_pickle(file_path=P6)


# P1 = Path("/home/zhanghang/caibinghao/ML4CO-Kit/test_dataset/ged/wrapper/ged_er-small_iso_4ins.txt")
# P2 = Path("/home/zhanghang/caibinghao/ML4CO-Kit/test_dataset/ged/wrapper/ged_er-small_iso_4ins.pkl")
# P3 = Path("/home/zhanghang/caibinghao/ML4CO-Kit/test_dataset/ged/wrapper/ged_er-small_pert_4ins.txt")
# P4 = Path("/home/zhanghang/caibinghao/ML4CO-Kit/test_dataset/ged/wrapper/ged_er-small_pert_4ins.pkl")

# g1 = GraphFeatureGenerator(feature_type=GRAPH_FEATURE_TYPE.UNIFORM)
# g2 = GraphFeatureGenerator(feature_type=GRAPH_FEATURE_TYPE.GAUSSIAN)
# iso_gen_u = GEDGenerator(nodes_num_scale=(8,8), nodes_feat_dim_scal=(4,4), edges_feat_dim_scal=(4,4), graph_generate_rule="isomorphic", node_feature_gen=g1, edge_feature_gen=g1)
# pert_gen_u = GEDGenerator(nodes_num_scale=(8,8), nodes_feat_dim_scal=(4,4), edges_feat_dim_scal=(4,4), graph_generate_rule="perturbed",perturb_node_features=True,perturb_edge_features=True, node_feature_gen=g1, edge_feature_gen=g1)
# solver = AStarSolver()
# w1 = GEDWrapper()
# w2 = GEDWrapper()
# w1.generate(generator=iso_gen_u, solver=solver, num_samples=4, batch_size=1)
# w2.generate(generator=pert_gen_u, solver=solver, num_samples=4, batch_size=1)
# w1.to_txt(file_path=P1)
# w1.to_pickle(file_path=P2)
# w2.to_txt(file_path=P3)
# w2.to_pickle(file_path=P4)
