
import os
os.environ["USE_JAX"] = "1"

import numpy as np
import jax
from jax import numpy as jnp
import pandas as pd
import scipy as sp
import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

input_path = "/Users/ctsl28/PD/ConCR/concr_cosmology/scripts/cancer/dataset/outputs/least_squares_fits_drug_subset.csv"
output_dir = "/Users/ctsl28/PD/ConCR/concr_cosmology/scripts/cancer/dataset/outputs/random_forest"
RANDOM_STATE = 42

DRUG_NAMES = {1003: 'Camptothecin', 1007: 'Docetaxel', 1073: '5-Fluorouracil', 1017: 'Olaparib'}

"""
Define methods
"""

def select_sample(input_df, SNR_cut=5.0, GoF_cut=3.0):
    return input_df[(input_df['success'] == True) & (input_df['fun'] < GoF_cut) & (np.abs(input_df['logic50'])/input_df['logic50_sigma'] > SNR_cut)]

def mean_across_drugsetid(input_df):
    mean_df = pd.DataFrame()
    for drug_id in np.unique(input_df['DRUG_ID']):
        res_df = input_df[input_df['DRUG_ID'] == drug_id]
        for cosmic_id in res_df['COSMIC_ID'].unique():
            cosmic_data = res_df[res_df['COSMIC_ID'] == cosmic_id]
            
            weights = 1 / (cosmic_data['logic50_sigma'])**2
            weighted_mean = np.sum(cosmic_data['logic50'] * weights) / np.sum(weights)
            weighted_std = np.sqrt(1 / np.sum(weights))
            
            weights_n = 1 / (cosmic_data['n_sigma'])**2
            weighted_mean_n = np.sum(cosmic_data['n_sigma'] * weights) / np.sum(weights)
            weighted_std_n = np.sqrt(1 / np.sum(weights_n))
            
            weights_base = 1 / (cosmic_data['base_sigma'])**2
            weighted_mean_base = np.sum(cosmic_data['base'] * weights) / np.sum(weights)
            weighted_std_base = np.sqrt(1 / np.sum(weights_base))  
            
            mean_df = pd.concat([mean_df, pd.DataFrame({
                'latent_logRNAseq': cosmic_data['latent_logRNAseq'],
                'latent_unimol_cls_repr': cosmic_data['latent_unimol_cls_repr'],
                'SMILES': cosmic_data['SMILES'],
                'DRUG_ID': drug_id,
                'COSMIC_ID': cosmic_id,
                'weighted_logic50_mean': weighted_mean,
                'weighted_logic50_std': weighted_std,
                'weighted_n_mean': weighted_mean_n,
                'weighted_n_std': weighted_std_n,
                'weighted_base_mean': weighted_mean_base,
                'weighted_base_std': weighted_std_base,
            })], ignore_index=True)
    return mean_df

def _parse_latent(v):
    if isinstance(v, np.ndarray):
        return v.astype(float, copy=False)
    if isinstance(v, (list, tuple)):
        return np.asarray(v, dtype=float)
    if isinstance(v, str):
        s = v.strip().strip('[]')
        sep = ',' if ',' in s else ' '
        arr = np.fromstring(s, sep=sep)
        return arr if arr.size else None
    return None


def build_feature_matrix(df):
    parsed = [_parse_latent(v) for v in df['latent_logRNAseq'].to_numpy()]
    sizes = {a.shape[0] for a in parsed if a is not None}
    if not sizes:
        return df.iloc[0:0].reset_index(drop=True), np.empty((0, 0))
    target = max(sizes)
    keep = np.array([
        a is not None and a.shape == (target,) and np.all(np.isfinite(a))
        for a in parsed
    ])
    df = df.loc[keep].reset_index(drop=True)
    X = np.stack([parsed[i] for i in np.flatnonzero(keep)])
    return df, X


def train_drug_efficiency_model_sklearn(X, y, y_weight, cosmic_ids):
    X_train, X_test, y_train, y_test, w_train, w_test, c_train, c_test = train_test_split(
        X, y, y_weight, cosmic_ids, test_size=0.2, random_state=RANDOM_STATE
    )

    rf = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    rf.fit(X_train, y_train, sample_weight=w_train)
    y_test_pred = rf.predict(X_test)

    r2 = r2_score(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print(f"Model Performance:")
    print(f"R^2 Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    return {
        "model": rf,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "w_train": w_train, "w_test": w_test,
        "cosmic_train": c_train, "cosmic_test": c_test,
        "test_r2": float(r2), "test_rmse": float(rmse),
    }


def fit_and_save(df, drug_id, variant, output_dir):
    df_i, X = build_feature_matrix(df)
    if len(df_i) < 10:
        print(f"Skipping drug {int(drug_id)} ({variant}): only {len(df_i)} usable rows")
        return
    y = df_i['weighted_logic50_mean'].to_numpy()
    w = np.ones_like(y)
    cosmic = df_i['COSMIC_ID'].to_numpy()
    print(f"Drug {int(drug_id)} ({variant}): training on {len(y)} samples")
    bundle = train_drug_efficiency_model_sklearn(X, y, w, cosmic)
    bundle.update({
        "drug_id": int(drug_id),
        "drug_name": DRUG_NAMES.get(int(drug_id), str(int(drug_id))),
        "variant": variant,
        "feature_spec": "latent_logRNAseq",
        "random_state": RANDOM_STATE,
    })
    out_path = os.path.join(output_dir, f"rf_{variant}_drug_{int(drug_id)}.joblib")
    joblib.dump(bundle, out_path)
    print(f"  Saved {out_path}")

"""
Read in data
"""
input_fits = pd.read_csv(input_path)


"""
Apply selection cut to data
"""
selected_fits = select_sample(input_fits)


"""
Take the mean of the logIC50 for a given DRUG_ID & COSMIC_ID
"""
mean_fits = mean_across_drugsetid(input_fits)
mean_selected_fits = mean_across_drugsetid(selected_fits)

"""
Fit a random forest per drug for each selection variant, and persist the
model + train/test splits + COSMIC_ID metadata so the validation plots
(actual-vs-predicted scatter, per-drug R^2/RMSE bars) can be reproduced
later from the saved bundles.
"""

os.makedirs(output_dir, exist_ok=True)

for drug_id in np.unique(mean_fits['DRUG_ID']):
    fit_and_save(
        mean_fits[mean_fits['DRUG_ID'] == drug_id],
        drug_id, "all", output_dir,
    )
    fit_and_save(
        mean_selected_fits[mean_selected_fits['DRUG_ID'] == drug_id],
        drug_id, "selected", output_dir,
    )

