from mpi4py import MPI
import plspm.config as c
import pandas as pd
import numpy as np
import plspm.inner_model as im
import plspm.outer_model as om
from plspm.weights import WeightsCalculatorFactory
from plspm.estimator import Estimator
from multiprocessing import Process, Queue, cpu_count
from queue import Empty
import time

def _create_summary(data: pd.DataFrame, original):
    """Generate summary statistics for bootstrap results."""
    summary = pd.DataFrame(0.0, index=data.columns, columns=["original", "mean", "std.error", "perc.025", "perc.975", "t stat."])
    summary.loc[:, "mean"] = data.mean(axis=0)
    summary.loc[:, "std.error"] = data.std(axis=0)
    summary.loc[:, "perc.025"] = data.quantile(0.025, axis=0)
    summary.loc[:, "perc.975"] = data.quantile(0.975, axis=0)
    summary.loc[:, "original"] = original
    summary.loc[:, "t stat."] = original / data.std(axis=0)
    return summary


class BootstrapWorker(Process):
    """Worker process that runs bootstrap iterations and sends results via a queue."""
    def __init__(self, queue: Queue, config: c.Config, data: pd.DataFrame, inner_model: im.InnerModel,
                 calculator: WeightsCalculatorFactory, iterations: int):
        super(BootstrapWorker, self).__init__()
        self.queue = queue
        self.config = config
        self.data = data
        self.inner_model = inner_model
        self.calculator = calculator
        self.iterations = iterations

    def run(self):
        """Execute bootstrap iterations in this worker process."""
        weights = pd.DataFrame(columns=self.data.columns, dtype="float")
        r_squared = pd.DataFrame(columns=self.inner_model.r_squared().index, dtype="float")
        total_effects = pd.DataFrame(columns=self.inner_model.effects().index, dtype="float")
        paths = pd.DataFrame(columns=self.inner_model.effects().index, dtype="float")
        loadings = pd.DataFrame(columns=self.data.columns, dtype="float")

        observations = self.data.shape[0]
        estimator = Estimator(self.config)

        for _ in range(self.iterations):
            try:
                boot_observations = np.random.randint(0, observations, size=observations)
                _final_data, _scores, _weights = estimator.estimate(self.calculator, self.data.iloc[boot_observations, :])
                inner_model_boot = im.InnerModel(self.config.path(), _scores)

                weights = pd.concat([weights, _weights.T], ignore_index=True)
                r_squared = pd.concat([r_squared, inner_model_boot.r_squared().to_frame().T], ignore_index=True)
                total_effects = pd.concat([total_effects, inner_model_boot.effects().loc[:, "total"].to_frame().T], ignore_index=True)
                paths = pd.concat([paths, inner_model_boot.effects().loc[:, "direct"].to_frame().T], ignore_index=True)
                loadings = pd.concat([loadings,
                                      (_scores.apply(lambda s: _final_data.corrwith(s)) * self.config.odm(self.config.path())).sum(axis=1).to_frame().T], ignore_index=True)
            except:
                pass

        results = {"weights": weights, "r_squared": r_squared, "total_effects": total_effects, "paths": paths, "loadings": loadings}
        self.queue.put(results)


class Bootstrap_Multi_Nodes:
    """Manages the execution of bootstrap validation across multiple nodes and cores."""
    def __init__(self, config: c.Config, data: pd.DataFrame, inner_model: im.InnerModel, outer_model: om.OuterModel,
                 calculator: WeightsCalculatorFactory, iterations: int, num_processes: int = None):
        
        # MPI Initialization
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Split bootstrap iterations across MPI nodes
        base_iterations_per_node = iterations // size
        extra_iterations = iterations % size

        iterations_per_rank = base_iterations_per_node + (1 if rank < extra_iterations else 0)
        print(f"Node {rank}: Assigned {iterations_per_rank} iterations out of {iterations} total.")


        # Get the number of CPU cores on the current node
        num_cores = cpu_count() if num_processes is None else num_processes
        print(f"Node {rank}: Using {num_cores} CPU cores. Distributing {iterations_per_rank} iterations across workers.")


        # Initialize storage for results
        weights = pd.DataFrame(columns=data.columns, dtype="float")
        r_squared = pd.DataFrame(columns=inner_model.r_squared().index, dtype="float")
        total_effects = pd.DataFrame(columns=inner_model.effects().index, dtype="float")
        paths = pd.DataFrame(columns=inner_model.effects().index, dtype="float")
        loadings = pd.DataFrame(columns=data.columns, dtype="float")

        base_iterations = iterations_per_rank // num_cores
        extra_iterations = iterations_per_rank % num_cores

        queue = Queue()
        processes = []

        # Start worker processes within the node
        for i in range(0, num_cores):
            worker_iterations = base_iterations + (1 if i < extra_iterations else 0)
            process = BootstrapWorker(queue, config, data, inner_model, calculator, worker_iterations)
            process.start()
            processes.append(process)

        # Collect results from all processes in the queue
        running = list(processes)
        while running:
            try:
                while not queue.empty()::
                    results = queue.get(False)
                    weights = pd.concat([weights, results["weights"]])
                    r_squared = pd.concat([r_squared, results["r_squared"]])
                    total_effects = pd.concat([total_effects, results["total_effects"]])
                    paths = pd.concat([paths, results["paths"]])
                    loadings = pd.concat([loadings, results["loadings"]])
            except Empty:
                pass
            time.sleep(1)
            if not queue.empty():
                continue
            running = [process for process in running if process.is_alive()]

        # MPI: Gather results at Rank 0
        try:
            all_results = comm.gather({"weights": weights, "r_squared": r_squared, 
                               "total_effects": total_effects, "paths": paths, "loadings": loadings}, root=0)
        except Exception as e:
            print(f"Node {MPI.COMM_WORLD.Get_rank()} failed during gather: {e}")
            comm.Abort(1)  # Exit cleanly if MPI fails

        if rank == 0:
            # Merge results from all nodes
            self.__weights = _create_summary(pd.concat([r["weights"] for r in all_results]), outer_model.model().loc[:, "weight"])
            self.__r_squared = _create_summary(pd.concat([r["r_squared"] for r in all_results]), inner_model.r_squared()).loc[inner_model.endogenous(), :]
            self.__total_effects = _create_summary(pd.concat([r["total_effects"] for r in all_results]), inner_model.effects().loc[:, "total"])
            self.__paths = _create_summary(pd.concat([r["paths"] for r in all_results]), inner_model.effects().loc[:, "direct"])
            self.__loading = _create_summary(pd.concat([r["loadings"] for r in all_results]), outer_model.model().loc[:, "loading"])
        else:
            self.__weights = self.__r_squared = self.__total_effects = self.__paths = self.__loading = None  # Other ranks don't store results

    def weights(self) -> pd.DataFrame:
        """Outer weights calculated from bootstrap validation."""
        return self.__weights

    def r_squared(self) -> pd.DataFrame:
        """R squared for latent variables calculated from bootstrap validation."""
        return self.__r_squared

    def total_effects(self) -> pd.DataFrame:
        """Total effects for paths calculated from bootstrap validation."""
        return self.__total_effects

    def paths(self) -> pd.DataFrame:
        """Direct effects for paths calculated from bootstrap validation."""
        return self.__paths[self.__paths["mean"] != 0]

    def loading(self) -> pd.DataFrame:
        """Loadings of manifest variables calculated from bootstrap validation."""
        return self.__loading
