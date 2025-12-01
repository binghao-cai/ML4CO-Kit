from ml4co_kit import GEDTask, GMTask
from ml4co_kit import GMGenerator, GEDGenerator
from ml4co_kit import GMWrapper, GEDWrapper
from pathlib import Path
from ml4co_kit import RRWMSolver
from ml4co_kit import AStarSolver

def build_task(gen, data_dir: Path):
    """
    Build a GEDTask from the given data directory.

    Args:
        data_dir (Path): The directory containing the dataset files.
    """

    task = gen.generate()
    task.to_pickle(data_dir)

def build_wrapper(wrapper, gen,solver, data_dir: Path, data_dir_txt: Path) :
    """
    Build a GEDWrapper from the given data directory.

    Args:
        data_dir (Path): The directory containing the dataset files.
        data_dir_txt (Path): The directory to save the dataset in txt format.

    Returns:
        GEDWrapper: The constructed GEDWrapper.
    """
    wrapper.generate(generator=gen,solver=solver,num_samples=4,num_threads=1,batch_size=1,show_time=True) 
    wrapper.to_pickle(data_dir)
    wrapper.to_txt(data_dir_txt)
   
   
gm_task_large = GMGenerator(
    nodes_num_scale=(30, 30),
    node_feat_dim_scale=(16,16),
    edge_feat_dim_scale=(16,16),
)
gm_task_small = GMGenerator(
    nodes_num_scale=(10, 10),
    node_feat_dim_scale=(36,36),
    edge_feat_dim_scale=(36,36),
)

ged_task = GEDGenerator(
    nodes_num_scale=(10, 10),
    node_feat_dim_scale=(36,36),
    edge_feat_dim_scale=(36,36),
)

gm_wrapper_large = GMWrapper()
gm_wrapper_small = GMWrapper()
ged_wrapper = GEDWrapper()

task_gm_large = Path("mydata/gm_large.pkl")
task_gm_small = Path("mydata/gm_small.pkl")
task_ged = Path("mydata/ged.pkl")
wrapper_gm_large = Path("mydata/wrapper_gm_large.pkl")
wrapper_gm_small = Path("mydata/wrapper_gm_small.pkl")
wrapper_ged = Path("mydata/wrapper_ged.pkl")
wrapper_gm_large_txt = Path("mydata/wrapper_gm_large.txt")
wrapper_gm_small_txt = Path("mydata/wrapper_gm_small.txt")
wrapper_ged_txt = Path("mydata/wrapper_ged.txt")

rrwm = RRWMSolver()
astar = AStarSolver()

build_task(gm_task_large, task_gm_large)
build_task(gm_task_small, task_gm_small)
build_task(ged_task, task_ged)

build_wrapper(gm_wrapper_large, gm_task_large, solver=rrwm, data_dir=wrapper_gm_large, data_dir_txt=wrapper_gm_large_txt)
build_wrapper(gm_wrapper_small, gm_task_small, solver=rrwm, data_dir=wrapper_gm_small, data_dir_txt=wrapper_gm_small_txt)
build_wrapper(ged_wrapper, ged_task, solver=astar, data_dir=wrapper_ged, data_dir_txt=wrapper_ged_txt)