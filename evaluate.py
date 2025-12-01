import numpy as np
from ml4co_kit import Graph, GraphFeatureGenerator, GEDTask, GEDGenerator, AStarSolver, GennAStarSolver

gen = GEDGenerator(nodes_num_scale=(15, 15), er_prob=0.5,node_feat_dim_scale=(4,4), edge_feat_dim_scale=(4,4))
ged = gen.generate()
solver=GennAStarSolver()
astar=AStarSolver()
#solver.batch_solve([ged])
astar.solve(ged)
print(ged.evaluate(ged.sol, mode='acc'))
print(ged.evaluate(ged.ref_sol))

