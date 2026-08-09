from collections import defaultdict
import statsmodels.api as sm
import plspm.config as c
import pandas as pd, numpy as np
import plspm.inner_model as im, plspm.outer_model as om

def _effects(path: pd.DataFrame):
    indirect_paths = pd.DataFrame(0, index=path.index, columns=path.columns)
    effects = pd.DataFrame({"from":     pd.Series(dtype="str"),
                            "to":       pd.Series(dtype="str"),
                            "direct":   pd.Series(dtype="float"),
                            "indirect": pd.Series(dtype="float"),
                            "total":    pd.Series(dtype="float")})
    num_lvs = len(list(path))
    if (num_lvs == 2):
        total_paths = path
    else:
        path_effects = {}
        path_effects[0] = path
        for i in range(1, num_lvs):
            path_effects[i] = path_effects[i - 1].dot(path)
            indirect_paths = indirect_paths + path_effects[i]
        total_paths = path + indirect_paths
    for from_lv in list(path):
        for to_lv in list(path):
            if from_lv != to_lv and total_paths.loc[to_lv, from_lv] != 0:
                effect = pd.DataFrame([{
                        "from": from_lv,
                        "to": to_lv,
                        "direct": path.loc[to_lv, from_lv],
                        "indirect": indirect_paths.loc[to_lv, from_lv],
                        "total": total_paths.loc[to_lv, from_lv]
                    }], index=[from_lv + " -> " + to_lv])
                effects = pd.concat([effects, effect])
    return effects

def _boot_naive_sign_change(boot_weights, boot_loadings, boot_path, original_weights, original_loadings, original_path) -> tuple[pd.Series, pd.Series, pd.DataFrame]: 

    sch_boot_weights = boot_weights.copy(deep=True)
    sch_boot_loadings = boot_loadings.copy(deep=True)
    sch_boot_path = boot_path.copy(deep=True)

    try:
        # Compare the original and the bootstrapped
        pairs = [(sch_boot_weights, original_weights), (sch_boot_loadings, original_loadings), (sch_boot_path, original_path)]
        for i, (df_boot, df_original) in enumerate(pairs, start = 1):
            if isinstance(df_boot, pd.Series):
                aligned_boot, aligned_original = df_boot.align(df_original, join="inner")
                df_boot.loc[aligned_boot.index] = np.abs(aligned_boot) * np.sign(aligned_original)

            elif isinstance(df_boot, pd.DataFrame):
                # Align both index and columns before operating
                df_boot_aligned, df_original_aligned = df_boot.align(df_original, join="inner", axis=0)
                df_boot_aligned, df_original_aligned = df_boot_aligned.align(df_original_aligned, join="inner", axis=1)

                # Use aligned versions here
                boot_sign = np.sign(df_boot_aligned)
                original_sign = np.sign(df_original_aligned)

                sign_mismatch = boot_sign != original_sign

                if sign_mismatch.any().any():
                    flipped = np.abs(df_boot_aligned) * np.sign(original_sign)
                    df_boot.loc[:, :] = flipped.values  # set values safely

        
        return sch_boot_weights, sch_boot_loadings, sch_boot_path
    except Exception as e:
        return boot_weights, boot_loadings, boot_path

def _boot_construct_level_change(config: c.Config, boot_inner_model: im.InnerModel, boot_weights: pd.DataFrame,
                                 boot_loadings: pd.DataFrame, boot_scores: pd.DataFrame, original_outer_weights: pd.DataFrame):
    """
    Adjust construct-level sign changes for a bootstrapped model
    """
    boot_cl_weights = boot_weights.copy(deep=True)
    boot_cl_loadings = boot_loadings.copy(deep=True)
    boot_cl_scores = boot_scores.copy(deep=True)
    try:
        ## Create a dictionary to store the LV and its indicators
        measurement_model = {}
        items_df = config.odm(config.path())
        
        for lv in items_df.columns:
            items = items_df.index[items_df[lv] == 1].tolist()
            measurement_model[lv] = items

        ## Flag df to keep track of which LV flips
        flag = pd.DataFrame(columns=items_df.columns, dtype="bool", index=["flag"])

        for lv, indi in measurement_model.items():
            sum_weights = boot_weights.loc[:,indi] + original_outer_weights.loc[indi, "weight"]
            diff_weights = boot_weights.loc[:,indi] - original_outer_weights.loc[indi, "weight"]

            if abs(sum_weights.sum(axis=1).item()) < abs(diff_weights.sum(axis=1).item()):
                #Multiple the bootstrapped loadings, scores, weights by -1
                boot_cl_weights.loc[:,indi] = boot_weights.loc[:,indi] * (-1)
                boot_cl_loadings.loc[:,indi] = boot_loadings.loc[:,indi] * (-1)
                boot_cl_scores.loc[:,lv] = boot_scores.loc[:,lv] * (-1)
                flag[lv] = True
            else:
                flag[lv] = False
                
        # Check each pair of construct to see if one flip
        boot_cl_path_coefficients = boot_inner_model.path_coefficients().copy(deep=True)
        
        effects_df = boot_inner_model.effects()
        for path in effects_df.index:
            endo = effects_df.at[path, "to"]
            exo = effects_df.at[path, "from"]
            if flag.at["flag", endo] ^ flag.at["flag", exo]:
                boot_cl_path_coefficients.at[endo,exo] = boot_cl_path_coefficients.at[endo,exo] * (-1)
        
        boot_cl_effects = _effects(boot_cl_path_coefficients)

    
        ##Recalculate the structural path using the final scores
        boot_cl_path_coefficients_recalc = boot_inner_model.path_coefficients().copy(deep=True)
        grouped_dv_iv = defaultdict(list)

        endo = boot_inner_model.inner_model().loc[:,"to"].tolist()
        exo = boot_inner_model.inner_model().loc[:,"from"].tolist()

        for dv, iv in zip(endo, exo):
            grouped_dv_iv[dv].append(iv)

        for dv, ivs in grouped_dv_iv.items():
            exogenous = sm.add_constant(boot_cl_scores.loc[:, ivs])
            regression = sm.OLS(boot_cl_scores.loc[:, dv], exogenous).fit()
            boot_cl_path_coefficients_recalc.loc[dv, ivs] = regression.params
        boot_cl_effects_recalc = _effects(boot_cl_path_coefficients_recalc)

        return boot_cl_scores, boot_cl_weights, boot_cl_loadings, boot_cl_path_coefficients, boot_cl_effects, boot_cl_path_coefficients_recalc, boot_cl_effects_recalc
    except:
        path = boot_inner_model.path_coefficients()
        effects = _effects(path)
        return boot_scores, boot_weights, boot_loadings, path, effects, path, effects


def _boot_dominant_indicator_change(config: c.Config, boot_inner_model: im.InnerModel, boot_weights: pd.DataFrame,
                                 boot_loadings: pd.DataFrame, boot_scores: pd.DataFrame, original_outer_model: om.OuterModel):
    """
    Adjust sign for a bootstrap model using the dominant indicator logic
    """
    boot_di_weights = boot_weights.copy(deep=True)
    boot_di_loadings = boot_loadings.copy(deep=True)
    boot_di_scores = boot_scores.copy(deep=True)
    try:
        original_loadings = original_outer_model.loc[:, "loading"].copy(deep=True)

        
        # Create a dictionary to store the LV and its indicators
        measurement_model = {}
        items_df = config.odm(config.path())
        
        # Flag to check which LV flips
        flag = pd.DataFrame(False, index=["flag"], columns=items_df.columns)

        for lv in items_df.columns:
            items = items_df.index[items_df[lv] == 1].tolist()
            measurement_model[lv] = items

        #Get the original highest loadings, compare and flip if needed
        for lv, items in measurement_model.items():
            lding = original_loadings.loc[items]
            dominant = lding.abs().idxmax()
            original_val = float(lding.loc[dominant])
            boot_val = float(boot_di_loadings.loc[:, dominant].iloc[0])
            if np.sign(original_val) != np.sign(boot_val):
                #Multiple the bootstrapped loadings, scores, weights by -1
                boot_di_weights.loc[:,items] = boot_weights.loc[:,items] * (-1)
                boot_di_loadings.loc[:,items] = boot_loadings.loc[:,items] * (-1)
                boot_di_scores.loc[:,lv] = boot_scores.loc[:,lv] * (-1)
                flag[lv] = True

                
        #Check each pair of construct to see if one flip
        boot_di_path_coefficients = boot_inner_model.path_coefficients().copy(deep=True)
        
        effects_df = boot_inner_model.effects()
        for path in effects_df.index:
            endo = effects_df.at[path, "to"]
            exo = effects_df.at[path, "from"]
            if flag.at["flag", endo] ^ flag.at["flag", exo]:
                boot_di_path_coefficients.at[endo,exo] = boot_di_path_coefficients.at[endo,exo] * (-1)
        
        boot_di_effects = _effects(boot_di_path_coefficients)

    
        #Recalculate the structural path using the final scores
        boot_di_path_coefficients_recalc = boot_inner_model.path_coefficients().copy(deep=True)
        grouped_dv_iv = defaultdict(list)

        endo = boot_inner_model.inner_model().loc[:,"to"].tolist()
        exo = boot_inner_model.inner_model().loc[:,"from"].tolist()

        for dv, iv in zip(endo, exo):
            grouped_dv_iv[dv].append(iv)

        for dv, ivs in grouped_dv_iv.items():
            exogenous = sm.add_constant(boot_di_scores.loc[:, ivs])
            regression = sm.OLS(boot_di_scores.loc[:, dv], exogenous).fit()
            boot_di_path_coefficients_recalc.loc[dv, ivs] = regression.params
        boot_di_effects_recalc = _effects(boot_di_path_coefficients_recalc)
        return boot_di_scores, boot_di_weights, boot_di_loadings, boot_di_path_coefficients, boot_di_effects, boot_di_path_coefficients_recalc, boot_di_effects_recalc
    except:
        path = boot_inner_model.path_coefficients()
        effects = _effects(path)
        return boot_scores, boot_weights, boot_loadings, path, effects, path, effects



def _boot_construct_scores_change(config: c.Config, boot_inner_model: im.InnerModel, boot_weights: pd.DataFrame,
                                 boot_loadings: pd.DataFrame, boot_scores: pd.DataFrame, boot_data: pd.DataFrame, original_outer_model: om.OuterModel):
    """
    Adjust the signs of the PLS parameters by comparing the original-weights calculated scores and the actual bootstrapped scores.  
    """
    boot_cs_weights = boot_weights.copy(deep=True)
    boot_cs_loadings = boot_loadings.copy(deep=True)
    boot_cs_scores = boot_scores.copy(deep=True)
    try:
        ## Get the original weights
        original_outer_weights = original_outer_model.loc[:, "weight"]


        ## Create a dictionary to store the LV and its indicators
        measurement_model = {}
        items_df = config.odm(config.path())
        
        for lv in items_df.columns:
            items = items_df.index[items_df[lv] == 1].tolist()
            measurement_model[lv] = items

        ## Flag df to keep track of which LV flips
        flag = pd.DataFrame(columns=items_df.columns, dtype="bool", index=["flag"])

        for lv, indi in measurement_model.items():
            w0 = original_outer_weights.loc[indi]
            new_scores = (boot_data[indi] @ w0).to_numpy()
            score_boot = boot_cs_scores[lv].to_numpy()
            # sum_of_scores = new_scores + score_boot
            # diff_of_scores = new_scores - score_boot

            # if abs(sum_of_scores.sum()) < abs(diff_of_scores.sum()):
            #     #Multiply the bootstrapped loadings, scores, weights by -1
            #     boot_cs_weights.loc[:,indi] = boot_weights.loc[:,indi] * (-1)
            #     boot_cs_loadings.loc[:,indi] = boot_loadings.loc[:,indi] * (-1)
            #     boot_cs_scores.loc[:,lv] = boot_scores.loc[:,lv] * (-1)
            #     flag[lv] = True
            # else:
            #     flag[lv] = False
            r = np.corrcoef(new_scores, score_boot)[0,1]

            if (np.isnan(r)):
                r = np.sign(np.dot(new_scores - new_scores.mean(), score_boot - score_boot.mean()))
            
            if (r < 0):
                boot_cs_weights.loc[:,indi] = boot_weights.loc[:,indi] * (-1)
                boot_cs_loadings.loc[:,indi] = boot_loadings.loc[:,indi] * (-1)
                boot_cs_scores.loc[:,lv] = boot_scores.loc[:,lv] * (-1)
                flag[lv] = True
            else:
                flag[lv] = False

                
        # Check each pair of construct to see if one flip
        boot_cs_path_coefficients = boot_inner_model.path_coefficients().copy(deep=True)
        
        effects_df = boot_inner_model.effects()
        for path in effects_df.index:
            endo = effects_df.at[path, "to"]
            exo = effects_df.at[path, "from"]
            if flag.at["flag", endo] ^ flag.at["flag", exo]:
                boot_cs_path_coefficients.at[endo,exo] = boot_cs_path_coefficients.at[endo,exo] * (-1)


        boot_cs_effects = _effects(boot_cs_path_coefficients)

        ##Recalculate the structural path using the final scores
        boot_cs_path_coefficients_recalc = boot_inner_model.path_coefficients().copy(deep=True)
        grouped_dv_iv = defaultdict(list)

        endo = boot_inner_model.inner_model().loc[:,"to"].tolist()
        exo = boot_inner_model.inner_model().loc[:,"from"].tolist()

        for dv, iv in zip(endo, exo):
            grouped_dv_iv[dv].append(iv)

        for dv, ivs in grouped_dv_iv.items():
            exogenous = sm.add_constant(boot_cs_scores.loc[:, ivs])
            regression = sm.OLS(boot_cs_scores.loc[:, dv], exogenous).fit()
            boot_cs_path_coefficients_recalc.loc[dv, ivs] = regression.params
        boot_cs_effects_recalc = _effects(boot_cs_path_coefficients_recalc)
        return boot_cs_scores, boot_cs_weights, boot_cs_loadings, boot_cs_path_coefficients, boot_cs_effects, boot_cs_path_coefficients_recalc, boot_cs_effects_recalc
    except:
        path = boot_inner_model.path_coefficients()
        effects = _effects(path)
        boot_cs_scores = pd.DataFrame("E", index=boot_cs_scores.index, columns=boot_cs_scores.columns)
        return boot_cs_scores, boot_weights, boot_loadings, path, effects, path, effects

