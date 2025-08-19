import plspm.config as c, pandas as pd, numpy as np, plspm.inner_model as im, plspm.outer_model as om, time
from multiprocessing import Process, Queue, cpu_count
from queue import Empty
from plspm.weights import WeightsCalculatorFactory
from plspm.estimator import Estimator
import secrets
import statsmodels.api as sm
from collections import defaultdict
import plspm.sign_change as sgch
import copy


def _create_summary(data: pd.DataFrame, original):
    summary = pd.DataFrame(0.0, index=data.columns, columns=["original", "mean", "std.error", "perc.025", "perc.975", "t stat."])
    summary.loc[:, "mean"] = data.mean(axis=0)
    summary.loc[:, "std.error"] = data.std(axis=0)
    summary.loc[:, "perc.025"] = data.quantile(0.025, axis=0)
    summary.loc[:, "perc.975"] = data.quantile(0.975, axis=0)
    summary.loc[:, "original"] = original
    summary.loc[:, "t stat."] = original / data.std(axis=0)
    return summary


class BootstrapProcess(Process):
    def __init__(self, queue: Queue, config: c.Config, data: pd.DataFrame, inner_model: im.InnerModel, outer_model: om.OuterModel, calculator: WeightsCalculatorFactory, iterations: int, seed: int, sign_change: bool = False,):
        super(BootstrapProcess, self).__init__()
        self.__queue = queue
        self.__config = config
        self.__data = data
        self.__inner_model = inner_model
        self.__calculator = calculator
        self.__iterations = iterations
        self.__seed = seed
        self.__original_outer_model = outer_model.model()
        self.__sign_change = sign_change

    def run(self):
        weights = pd.DataFrame(columns=self.__data.columns, dtype="float")
        r_squared = pd.DataFrame(columns=self.__inner_model.r_squared().index, dtype="float")
        total_effects = pd.DataFrame(columns=self.__inner_model.effects().index, dtype="float")
        paths = pd.DataFrame(columns=self.__inner_model.effects().index, dtype="float")
        loadings = pd.DataFrame(columns=self.__data.columns, dtype="float")
        raw_scores_list = []
        final_data_list = []
        inner_model_effects = []

        if self.__sign_change:
            #naive sign change objects
            nc_loadings = []
            nc_weights = []
            nc_path_coef = []

            #construct-level change objects
            cl_loadings = []
            cl_scores = []
            cl_weights = []
            cl_path_coef = []
            cl_effects = []
            cl_path_coef_recalc = []
            cl_effects_recalc = []
            
            #dominant indicators objects
            di_loadings = []
            di_scores = []
            di_weights = []
            di_path_coef = []
            di_effects = []
            di_path_coef_recalc = []
            di_effects_recalc = []
            
            #construct score objects
            cs_scores = []
            cs_loadings = []
            cs_weights = []
            cs_path_coef = []
            cs_effects = []
            cs_path_coef_recalc = []
            cs_effects_recalc = []

        original_weights = self.__original_outer_model.loc[:, "weight"].copy()
        original_loadings = self.__original_outer_model.loc[:, "loading"].copy()
        original_path = self.__inner_model.path_coefficients().copy()
        observations = self.__data.shape[0]
        estimator = Estimator(self.__config)

        rng = np.random.default_rng(self.__seed)

        for i in range(0, self.__iterations):
            try:
                boot_observations = rng.integers(0, observations, size=observations)
                _final_data, _scores, _weights = estimator.estimate(self.__calculator, self.__data.iloc[boot_observations, :])
                weights = pd.concat([weights, _weights.T], ignore_index = True)
                inner_model = im.InnerModel(self.__config.path(), _scores)
                r_squared = pd.concat([r_squared, inner_model.r_squared().to_frame().T], ignore_index=True)
                total_effects = pd.concat([total_effects,
                                           inner_model.effects().loc[:, "total"].to_frame().T], ignore_index=True)
                paths = pd.concat([paths,
                                   inner_model.effects().loc[:, "direct"].to_frame().T], ignore_index=True)
                boot_loadings = (_scores.apply(lambda s: _final_data.corrwith(s)) * self.__config.odm(self.__config.path())).sum(axis=1).to_frame().T
                loadings = pd.concat([loadings, boot_loadings], ignore_index=True)
                raw_scores_list.append(_scores)
                final_data_list.append(_final_data)
                inner_model_effects.append(inner_model.effects())

                if self.__sign_change:
                    for cl_df, boot_cl_df in zip(
                        [cl_scores, cl_weights, cl_loadings, cl_path_coef, cl_effects, cl_path_coef_recalc, cl_effects_recalc],
                        sgch._boot_construct_level_change(self.__config, inner_model, _weights.T, boot_loadings, _scores, self.__original_outer_model)):
                        cl_df.append(boot_cl_df)

                    for di_df, boot_di_df in zip(
                        [di_scores, di_weights, di_loadings, di_path_coef, di_effects, di_path_coef_recalc, di_effects_recalc],
                        sgch._boot_dominant_indicator_change(self.__config, inner_model, _weights.T, boot_loadings, _scores, self.__original_outer_model)):
                        di_df.append(boot_di_df)

                    for cs_df, boot_cs_df in zip(
                        [cs_scores, cs_weights, cs_loadings, cs_path_coef, cs_effects, cs_path_coef_recalc, cs_effects_recalc],
                        sgch._boot_construct_scores_change(
                            self.__config, inner_model, _weights.T, boot_loadings, _scores, _final_data, self.__original_outer_model)):
                        cs_df.append(boot_cs_df)

                    for nc_df, boot_nc_df in zip(
                        [nc_weights, nc_loadings, nc_path_coef],
                        sgch._boot_naive_sign_change(_weights, boot_loadings.T, inner_model.path_coefficients(), original_weights, original_loadings, original_path)):
                        nc_df.append(boot_nc_df)

            except:
                pass
        results = {}
        results["weights"] = weights
        results["r_squared"] = r_squared
        results["total_effects"] = total_effects
        results["paths"] = paths
        results["loadings"] = loadings
        results["scores"] = raw_scores_list
        results["final_data"] = final_data_list
        results["inner_model"] = inner_model_effects 

        if self.__sign_change:
            #nc
            results["nc_loadings"] = nc_loadings
            results["nc_weights"] = nc_weights
            results["nc_path_coef"] = nc_path_coef

            #cl
            results["cl_scores"] = cl_scores
            results["cl_loadings"] = cl_loadings
            results["cl_weights"] = cl_weights
            results["cl_path_coef"] = cl_path_coef
            results["cl_effects"] = cl_effects
            results["cl_effects_recalc"] = cl_effects_recalc
            results["cl_path_coef_recalc"] = cl_path_coef_recalc
            #di
            results["di_scores"] = di_scores
            results["di_loadings"] = di_loadings
            results["di_weights"] = di_weights
            results["di_path_coef"] = di_path_coef
            results["di_effects"] = di_effects
            results["di_effects_recalc"] = di_effects_recalc
            results["di_path_coef_recalc"] = di_path_coef_recalc

            #cs
            results["cs_scores"] = cs_scores
            results["cs_loadings"] = cs_loadings
            results["cs_weights"] = cs_weights
            results["cs_path_coef"] = cs_path_coef
            results["cs_effects"] = cs_effects
            results["cs_effects_recalc"] = cs_effects_recalc
            results["cs_path_coef_recalc"] = cs_path_coef_recalc


        self.__queue.put(results)


class Bootstrap:
    """Performs bootstrap validation to determine the statistical significance of the model.

    Setting ``bootstrap=True`` when constructing :class:`.Plspm` will perform bootstrap validation. Calling :meth:`~.Plspm.bootstrap` on :class:`.Plspm` will return an instance of this class, from which the bootstrapping results can be retrieved by calling the methods listed below.
    """
    def __init__(self, config: c.Config, data: pd.DataFrame, inner_model: im.InnerModel, outer_model: om.OuterModel,
                 calculator: WeightsCalculatorFactory, iterations: int, num_processes: int, seed: int = None, sign_change: bool = False ):
        self.__original_inner_model = copy.deepcopy(inner_model)
        weights = pd.DataFrame(columns=data.columns, dtype="float")
        r_squared = pd.DataFrame(columns=inner_model.r_squared().index, dtype="float")
        total_effects = pd.DataFrame(columns=inner_model.effects().index, dtype="float")
        paths = pd.DataFrame(columns=inner_model.effects().index, dtype="float")
        loadings = pd.DataFrame(columns=data.columns, dtype="float")
        raw_scores = []
        final_data = []
        boot_inner_model = []
        
        if sign_change:
            nc_loadings = []
            nc_weights = []
            nc_path_coef = []

            cl_effects = []
            cl_path_coef = []
            cl_effects_recalc = []
            cl_path_coef_recalc = []
            cl_scores = []
            cl_loadings = []
            cl_weights = []

            di_effects = []
            di_path_coef = []
            di_effects_recalc = []
            di_path_coef_recalc = []
            di_scores = []
            di_loadings = []
            di_weights = []

            cs_effects = []
            cs_path_coef = []
            cs_effects_recalc = []
            cs_path_coef_recalc = []
            cs_scores = []
            cs_loadings = []
            cs_weights = []

        queue = Queue()
        processes = []

        base_seed = seed if seed is not None else secrets.randbits(64)
        num_cores = cpu_count() if num_processes is None else num_processes
        print(f"Using {num_cores} CPU cores")


        for t in range(0, num_cores):
            process_seed = base_seed + t
            base_iteration = iterations // num_cores
            extra_iteration = iterations % num_cores
            worker_iterations = base_iteration + (1 if t < extra_iteration else 0)
            process = BootstrapProcess(queue, config, data, inner_model, outer_model, calculator, worker_iterations, process_seed, sign_change)
            process.start()
            processes.append(process)

        running = list(processes)
        while running:
            try:
                while True:
                    results = queue.get(False)
                    weights = pd.concat([weights, results["weights"]])
                    r_squared = pd.concat([r_squared, results["r_squared"]])
                    total_effects = pd.concat([total_effects, results["total_effects"]])
                    paths = pd.concat([paths, results["paths"]])
                    loadings = pd.concat([loadings, results["loadings"]])
                    raw_scores.extend(results["scores"])
                    final_data.extend(results["final_data"])
                    boot_inner_model.extend(results["inner_model"])
                    
                    if sign_change:
                        # naive sign change
                        nc_loadings.extend(results["nc_loadings"])
                        nc_weights.extend(results["nc_weights"])
                        nc_path_coef.extend(results["nc_path_coef"])

                        # construct-level sign change
                        cl_effects.extend(results["cl_effects"])
                        cl_path_coef.extend(results["cl_path_coef"])
                        cl_effects_recalc.extend(results["cl_effects_recalc"])
                        cl_path_coef_recalc.extend(results["cl_path_coef_recalc"])
                        cl_scores.extend(results["cl_scores"])
                        cl_loadings.extend(results["cl_loadings"])
                        cl_weights.extend(results["cl_weights"])

                        # dominant indicators sign change
                        di_effects.extend(results["di_effects"])
                        di_path_coef.extend(results["di_path_coef"])
                        di_effects_recalc.extend(results["di_effects_recalc"])
                        di_path_coef_recalc.extend(results["di_path_coef_recalc"])
                        di_scores.extend(results["di_scores"])
                        di_loadings.extend(results["di_loadings"])
                        di_weights.extend(results["di_weights"])

                        # construct scores sign change
                        cs_scores.extend(results["cs_scores"])
                        cs_loadings.extend(results["cs_loadings"])
                        cs_weights.extend(results["cs_weights"])
                        cs_path_coef.extend(results["cs_path_coef"])
                        cs_effects.extend(results["cs_effects"])
                        cs_effects_recalc.extend(results["cs_effects_recalc"])
                        cs_path_coef_recalc.extend(results["cs_path_coef_recalc"])
            except Empty:
                pass
            time.sleep(1)
            if not queue.empty():
                continue
            running = [process for process in running if process.is_alive()]

        self.__weights = _create_summary(weights, outer_model.model().loc[:, "weight"])
        self.__r_squared = _create_summary(r_squared, self.__original_inner_model.r_squared()).loc[self.__original_inner_model.endogenous(), :]
        self.__r_squared_list = r_squared
        self.__total_effects = _create_summary(total_effects, self.__original_inner_model.effects().loc[:, "total"])
        self.__paths = _create_summary(paths, self.__original_inner_model.effects().loc[:, "direct"])
        self.__loading = _create_summary(loadings, outer_model.model().loc[:, "loading"])
        self.__seed = base_seed
        self.__raw_scores = {f"scores_{i}": raw_scores[i] for i in range(len(raw_scores))}
        self.__final_data = {f"final_data_{i}": final_data[i] for i in range(len(final_data))}
        self.__raw_weights = weights
        self.__raw_loadings = loadings
        self.__boot_inner_model = boot_inner_model
        self.__sign_change = sign_change
        
        if sign_change:
            # sign changes objects
            self.__nc_loadings = {f"nc_loadings_{i}": nc_loadings[i] for i in range(len(nc_loadings))}
            self.__nc_weights = {f"nc_weights_{i}": nc_weights[i] for i in range(len(nc_weights))}
            self.__nc_path_coef = {f"nc_path_coef_{i}": nc_path_coef[i] for i in range(len(nc_path_coef))}
            
            self.__cl_path_coef = {f"cl_path{i}": cl_path_coef[i] for i in range(len(cl_path_coef))}
            self.__cl_effects = {f"cl_effect_{i}": cl_effects[i] for i in range(len(cl_effects))}
            self.__cl_path_coef_recalc = {f"cl_path_recalc_{i}": cl_path_coef_recalc[i] for i in range(len(cl_path_coef_recalc))}
            self.__cl_effects_recalc = {f"boot_effect_recalc_{i}": cl_effects_recalc[i] for i in range(len(cl_effects_recalc))}
            self.__cl_scores = {f"cl_scores_{i}": cl_scores[i] for i in range(len(cl_scores))}
            self.__cl_loadings = {f"cl_loadings_{i}": cl_loadings[i] for i in range(len(cl_loadings))}
            self.__cl_weights = {f"cl_weights_{i}": cl_weights[i] for i in range(len(cl_weights))}
            
            self.__di_path_coef = {f"di_path{i}": di_path_coef[i] for i in range(len(di_path_coef))}
            self.__di_effects = {f"di_effect_{i}": di_effects[i] for i in range(len(di_effects))}
            self.__di_path_coef_recalc = {f"di_path_recalc_{i}": di_path_coef_recalc[i] for i in range(len(di_path_coef_recalc))}
            self.__di_effects_recalc = {f"boot_effect_recalc_{i}": di_effects_recalc[i] for i in range(len(di_effects_recalc))}
            self.__di_scores = {f"di_scores_{i}": di_scores[i] for i in range(len(di_scores))}
            self.__di_loadings = {f"di_loadings_{i}": di_loadings[i] for i in range(len(di_loadings))}
            self.__di_weights = {f"di_weights_{i}": di_weights[i] for i in range(len(di_weights))}
            
            self.__cs_scores = {f"cs_scores_{i}": cs_scores[i] for i in range(len(cs_scores))}
            self.__cs_loadings = {f"cs_loadings_{i}": cs_loadings[i] for i in range(len(cs_loadings))}
            self.__cs_weights = {f"cs_weights_{i}": cs_weights[i] for i in range(len(cs_weights))}
            self.__cs_path_coef = {f"cs_path{i}": cs_path_coef[i] for i in range(len(cs_path_coef))}
            self.__cs_effects = {f"cs_effects_{i}": cs_effects[i] for i in range(len(cs_effects))}
            self.__cs_effects_recalc = {f"cs_effects_recalc_{i}": cs_effects_recalc[i] for i in range(len(cs_effects_recalc))}
            self.__cs_path_coef_recalc = {f"cs_path_recalc_{i}": cs_path_coef_recalc[i] for i in range(len(cs_path_coef_recalc))}



    def require_sign_change(func):
        def wrapper(self, *args, **kwargs):
            if not self.__sign_change:
                raise ValueError(f"Sign change is not enabled. Cannot call {func.__name__}")
            return func(self, *args, **kwargs)
        return wrapper

    def weights(self) -> pd.DataFrame:
        """Outer weights calculated from bootstrap validation."""
        return self.__weights

    def r_squared(self) -> pd.DataFrame:
        """R squared for latent variables calculated from bootstrap validation."""
        return self.__r_squared
    
    def r_squared_list(self) -> pd.DataFrame:
        """R squared for latent variables calculated from bootstrap validation."""
        return self.__r_squared_list


    def total_effects(self) -> pd.DataFrame:
        """Total effects for paths calculated from bootstrap validation."""
        return self.__total_effects

    def paths(self) -> pd.DataFrame:
        """Direct effects for paths calculated from bootstrap validation."""
        return self.__paths

    def loading(self) -> pd.DataFrame:
        """Loadings of manifest variables calculated from bootstrap validation."""
        return self.__loading

    def seed(self) -> int:
        """The seed used for random number generatory"""
        return self.__seed

    def boot_scores(self) -> dict:
        """The bootstrap scores"""
        return self.__raw_scores
    
    def boot_final_data(self) -> dict:
        """The final data"""
        return self.__final_data

    def boot_raw_weights(self):
        return self.__raw_weights

    def boot_raw_loadings(self):
        return self.__raw_loadings

    def boot_inner_model(self):
        return self.__boot_inner_model
    
    @require_sign_change
    def boot_nc_weights(self):
        return self.__nc_weights
    
    @require_sign_change
    def boot_nc_loadings(self):
        return self.__nc_loadings   
    
    @require_sign_change
    def boot_nc_path_coef(self):
        return self.__nc_path_coef
    
    @require_sign_change
    def boot_cl_scores(self):
        return self.__cl_scores
    
    @require_sign_change
    def boot_cl_loadings(self):
        return self.__cl_loadings
    
    @require_sign_change
    def boot_cl_weights(self):
        return self.__cl_weights
    
    @require_sign_change
    def boot_cl_path_coef(self):
        return self.__cl_path_coef
    
    @require_sign_change
    def boot_cl_effects(self):
        return self.__cl_effects
    
    @require_sign_change
    def boot_cl_path_coef_recalc(self):
        return self.__cl_path_coef_recalc
    
    @require_sign_change
    def boot_cl_effects_recalc(self):
        return self.__cl_effects_recalc
    @require_sign_change
    def boot_di_weights(self):
        if not self.__sign_change:
            raise ValueError("Sign change is not enabled. Cannot return sign change objects.")
        return self.__di_weights

    @require_sign_change        
    def boot_di_loadings(self):
        return self.__di_loadings
    
    @require_sign_change
    def boot_di_scores(self):
        return self.__di_scores
    
    @require_sign_change
    def boot_di_effects(self):
        return self.__di_effects
    
    @require_sign_change
    def boot_di_path_coef(self):
        return self.__di_path_coef
    
    @require_sign_change
    def boot_di_effects_recalc(self):
        return self.__di_effects_recalc
    
    @require_sign_change
    def boot_di_path_coef_recalc(self):
        return self.__di_path_coef_recalc
    
    @require_sign_change
    def boot_cs_scores(self):
        return self.__cs_scores
    
    @require_sign_change
    def boot_cs_loadings(self):
        return self.__cs_loadings
    
    @require_sign_change
    def boot_cs_weights(self):
        return self.__cs_weights
    
    @require_sign_change
    def boot_cs_path_coef(self):
        return self.__cs_path_coef


