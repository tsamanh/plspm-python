from mpi4py import MPI
import plspm.config as c, pandas as pd, cupy as cp, plspm.inner_model as im, plspm.outer_model as om, time
from plspm.weights import WeightsCalculatorFactory
from plspm.estimator import Estimator

def _create_summary(data: pd.DataFrame, original):
    summary = pd.DataFrame(0.0, index=data.columns, columns=["original", "mean", "std.error", "perc.025", "perc.975", "t stat."])
    summary.loc[:, "mean"] = data.mean(axis=0)
    summary.loc[:, "std.error"] = data.std(axis=0)
    summary.loc[:, "perc.025"] = data.quantile(0.025, axis=0)
    summary.loc[:, "perc.975"] = data.quantile(0.975, axis=0)
    summary.loc[:, "original"] = original
    summary.loc[:, "t stat."] = original / data.std(axis=0)
    return summary


class Bootstrap_Multi_Nodes:
    """Performs bootstrap validation across multiple MPI nodes."""

    def __init__(self, config: c.Config, data: pd.DataFrame, inner_model: im.InnerModel, outer_model: om.OuterModel,
                 calculator: WeightsCalculatorFactory, iterations: int):
        
        # Initialize MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        # Split bootstrap iterations across MPI nodes
        iterations_per_rank = iterations // size

        weights = pd.DataFrame(columns=data.columns, dtype="float")
        r_squared = pd.DataFrame(columns=inner_model.r_squared().index, dtype="float")
        total_effects = pd.DataFrame(columns=inner_model.effects().index, dtype="float")
        paths = pd.DataFrame(columns=inner_model.effects().index, dtype="float")
        loadings = pd.DataFrame(columns=data.columns, dtype="float")

        observations = data.shape[0]
        estimator = Estimator(config)

        # Each rank runs its portion of bootstrap iterations
        for i in range(iterations_per_rank):
            try:
                boot_observations = cp.random.randint(observations, size=observations)
                _final_data, _scores, _weights = estimator.estimate(calculator, data.iloc[boot_observations, :])
                weights = pd.concat([weights, _weights.T], ignore_index=True)
                inner_model_boot = im.InnerModel(config.path(), _scores)
                r_squared = pd.concat([r_squared, inner_model_boot.r_squared().to_frame().T], ignore_index=True)
                total_effects = pd.concat([total_effects, inner_model_boot.effects().loc[:, "total"].to_frame().T], ignore_index=True)
                paths = pd.concat([paths, inner_model_boot.effects().loc[:, "direct"].to_frame().T], ignore_index=True)
                loadings = pd.concat([loadings,
                                      (_scores.apply(lambda s: _final_data.corrwith(s)) * config.odm(config.path())).sum(axis=1).to_frame().T], ignore_index=True)
            except:
                pass  # Continue even if an error occurs

        # Gather results at Rank 0
        results = {
            "weights": weights,
            "r_squared": r_squared,
            "total_effects": total_effects,
            "paths": paths,
            "loadings": loadings
        }

        if rank == 0:
            all_results = [results]
            for r in range(1, size):
                recv_data = comm.recv(source=r)
                all_results.append(recv_data)

            # Merge results from all nodes
            self.__weights = _create_summary(pd.concat([r["weights"] for r in all_results]), outer_model.model().loc[:, "weight"])
            self.__r_squared = _create_summary(pd.concat([r["r_squared"] for r in all_results]), inner_model.r_squared()).loc[inner_model.endogenous(), :]
            self.__total_effects = _create_summary(pd.concat([r["total_effects"] for r in all_results]), inner_model.effects().loc[:, "total"])
            self.__paths = _create_summary(pd.concat([r["paths"] for r in all_results]), inner_model.effects().loc[:, "direct"])
            self.__loading = _create_summary(pd.concat([r["loadings"] for r in all_results]), outer_model.model().loc[:, "loading"])
        else:
            comm.send(results, dest=0)  # Other ranks send results to Rank 0

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
