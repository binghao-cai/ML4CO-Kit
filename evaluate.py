import time
import pandas as pd
from ml4co_kit import GMGenerator
from ml4co_kit.generator.graphset.base import GraphFeatureGenerator, GRAPH_FEATURE_TYPE  

from ml4co_kit.solver.sm import SMSolver
from ml4co_kit.solver.ipfp import IPFPSolver
from ml4co_kit.solver.rrwm import RRWMSolver
from ml4co_kit.solver.ngm import NGMSolver

# solvers
sm = SMSolver()
ipfp = IPFPSolver()
rrwm = RRWMSolver()
ngm = NGMSolver()

# feature generator
g1 = GraphFeatureGenerator(feature_type=GRAPH_FEATURE_TYPE.UNIFORM)

# record containers for each graph type
def init_log():
    return {
        "size": [],
        "time_sm": [], "acc_sm": [],
        "time_ipfp": [], "acc_ipfp": [],
        "time_rrwm": [], "acc_rrwm": [],
        "time_ngm": [], "acc_ngm": [],
    }

log_iso = init_log()
log_ind = init_log()
log_pert = init_log()

def run_solvers(graph, log, size):
    log["size"].append(size)

    # SM
    t0 = time.perf_counter()
    sm.solve(graph)
    t = time.perf_counter() - t0
    acc=graph.evaluate(graph.sol)
    print(f"SM: t={t}, acc={acc}")
    log["time_sm"].append(t)
    log["acc_sm"].append(acc)

    # IPFP
    t0 = time.perf_counter()
    ipfp.solve(graph)
    t = time.perf_counter() - t0
    acc=graph.evaluate(graph.sol)
    print(f"IPFP: t={t}, acc={acc}")
    log["time_ipfp"].append(t)
    log["acc_ipfp"].append(acc)

    # RRWM
    t0 = time.perf_counter()
    rrwm.solve(graph)
    t = time.perf_counter() - t0
    acc=graph.evaluate(graph.sol)
    print(f"RRWM: t={t}, acc={acc}")
    log["time_rrwm"].append(t)
    log["acc_rrwm"].append(acc)

    # NGM
    t0 = time.perf_counter()
    ngm.batch_solve([graph])
    t = time.perf_counter() - t0
    acc=graph.evaluate(graph.sol)
    print(f"NGM: t={t}, acc={acc}")
    log["time_ngm"].append(t)
    log["acc_ngm"].append(acc)

# -------- main loop --------
for i in range(1, 8):
    size = 10 * i

    iso_gen = GMGenerator(
        nodes_num_scale=(size, size),
        nodes_feat_dim_scal=(8, 8),
        edges_feat_dim_scal=(8, 8),
        graph_generate_rule="isomorphic",
        node_feature_gen=g1,
        edge_feature_gen=g1,
    )

    ind_gen = GMGenerator(
        nodes_num_scale=(size, size),
        nodes_feat_dim_scal=(8, 8),
        edges_feat_dim_scal=(8, 8),
        graph_generate_rule="induced_subgraph",
        keep_ratio=0.7,
        node_feature_gen=g1,
        edge_feature_gen=g1,
    )

    pert_gen = GMGenerator(
        nodes_num_scale=(size, size),
        nodes_feat_dim_scal=(8, 8),
        edges_feat_dim_scal=(8, 8),
        graph_generate_rule="perturbed",
        perturb_node_features=True,
        perturb_edge_features=True,
        node_feature_gen=g1,
        edge_feature_gen=g1,
    )

    iso = iso_gen.generate()
    ind = ind_gen.generate()
    pert = pert_gen.generate()

    # run and log
    run_solvers(iso, log_iso, size)
    run_solvers(ind, log_ind, size)
    run_solvers(pert, log_pert, size)
    print(f"round{i}")

# save csv
pd.DataFrame(log_iso).to_csv("results_iso.csv", index=False)
pd.DataFrame(log_ind).to_csv("results_ind.csv", index=False)
pd.DataFrame(log_pert).to_csv("results_pert.csv", index=False)

print("已保存：results_iso.csv, results_ind.csv, results_pert.csv")
