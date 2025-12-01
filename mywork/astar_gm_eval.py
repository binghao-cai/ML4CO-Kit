import time
import pandas as pd
from ml4co_kit import GMGenerator
from ml4co_kit.generator.graphset.base import GraphFeatureGenerator, GRAPH_FEATURE_TYPE  

from ml4co_kit.solver.sm import SMSolver
from ml4co_kit.solver.ipfp import IPFPSolver
from ml4co_kit.solver.rrwm import RRWMSolver
from ml4co_kit.solver.ngm import NGMSolver
from ml4co_kit.solver.astar import AStarSolver
from ml4co_kit.solver.genn_astar import GENN_AStarSolver

# solvers
sm = SMSolver()
ipfp = IPFPSolver()
rrwm = RRWMSolver()
ngm = NGMSolver()
astar = AStarSolver()
genn = GENN_AStarSolver()
# feature generator
g1 = GraphFeatureGenerator(feature_type=GRAPH_FEATURE_TYPE.UNIFORM)

# record containers for each graph type
def init_log():
    return {
        "size": [],
        "time_astar": [], "acc_astar": [],
        "time_genn_astar": [], "acc_genn_astar": [],
    }

log_iso = init_log()
log_pert = init_log()

def run_solvers(graph, log, size):
    log["size"].append(size)

    # AStar
    t0 = time.perf_counter()
    astar.solve(graph)
    t = time.perf_counter() - t0
    acc=graph.evaluate(graph.sol)
    print(f"AStar: t={t}, acc={acc}")
    log["time_astar"].append(t)
    log["acc_astar"].append(acc)

    # GENN
    t0 = time.perf_counter()
    genn.batch_solve([graph])
    t = time.perf_counter() - t0
    acc=graph.evaluate(graph.sol)
    print(f"GENN_AStar: t={t}, acc={acc}")
    log["time_genn_astar"].append(t)
    log["acc_genn_astar"].append(acc)


# -------- main loop --------
for i in range(4, 11):
    size = i
    print(f"===== round {i} =====")
    iso_gen = GMGenerator(
        nodes_num_scale=(size, size),
        nodes_feat_dim_scal=(36, 36),
        edges_feat_dim_scal=(36, 36),
        graph_generate_rule="isomorphic",
        er_prob = 0.5,
        node_feature_gen=g1,
        edge_feature_gen=g1,
    )


    pert_gen = GMGenerator(
        nodes_num_scale=(size, size),
        nodes_feat_dim_scal=(36, 36),
        edges_feat_dim_scal=(36, 36),
        graph_generate_rule="perturbed",
        perturb_node_features=True,
        perturb_edge_features=True,
        node_feature_gen=g1,
        edge_feature_gen=g1,
    )

    iso = iso_gen.generate()
    pert = pert_gen.generate()

    # run and log
    run_solvers(iso, log_iso, size)
    run_solvers(pert, log_pert, size)
    print()
    print()

# save csv
pd.DataFrame(log_iso).to_csv("astar_results_iso.csv", index=False)
pd.DataFrame(log_pert).to_csv("astar_results_pert.csv", index=False)

print("已保存：astar_results_iso.csv, astar_results_pert.csv")
