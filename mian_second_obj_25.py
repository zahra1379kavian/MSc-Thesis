# Combine Run 1 & 2
import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from os.path import join
from empca.empca.empca import empca
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, ticker
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import plotting
from scipy import sparse
from scipy.io import loadmat
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.stats import t as student_t
import pingouin as pg

LOG_FILE = "run_metrics_log.jsonl"


def _array_summary(array):
    flat = np.asarray(array, dtype=np.float64).ravel()
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return {"min": np.nan, "max": np.nan, "mean": np.nan, "var": np.nan}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "var": float(np.var(finite)),
    }


def _matrix_norm_summary(matrix):
    mat = np.asarray(matrix, dtype=np.float64)
    finite = np.isfinite(mat)
    if not finite.any():
        return {"fro": np.nan, "mean_abs": np.nan, "max_abs": np.nan}
    safe_mat = np.where(finite, mat, 0.0)
    fro_norm = float(np.linalg.norm(safe_mat, ord="fro"))
    abs_vals = np.abs(safe_mat[finite])
    return {"fro": fro_norm, "mean_abs": float(np.mean(abs_vals)), "max_abs": float(np.max(abs_vals))}


def _sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.floating, np.integer, float, int)):
        value = float(obj)
        if not math.isfinite(value):
            return None
        return value
    if isinstance(obj, str) or obj is None:
        return obj
    return str(obj)


def _append_run_log(entry, path=LOG_FILE):
    payload = dict(entry)
    payload["timestamp"] = datetime.utcnow().isoformat() + "Z"
    payload = _sanitize_for_json(payload)
    with open(path, "a") as log_file:
        log_file.write(json.dumps(payload) + "\n")


def synchronize_beta_voxels(beta_run1, beta_run2, mask_run1, mask_run2):
    valid_run1, valid_run2 = ~mask_run1, ~mask_run2
    shared_valid = valid_run1 & valid_run2
    shared_mask_run1 = shared_valid[np.flatnonzero(valid_run1)] #indices of nonzero elements in mask_run1
    shared_mask_run2 = shared_valid[np.flatnonzero(valid_run2)]

    synced_run1, synced_run2 = beta_run1[shared_mask_run1], beta_run2[shared_mask_run2]
    removed_run1 = beta_run1.shape[0] - synced_run1.shape[0]
    removed_run2 = beta_run2.shape[0] - synced_run2.shape[0]
    print(f"Removed {removed_run1} run1 voxels and {removed_run2} run2 voxels to enforce a shared voxel set.", flush=True)

    shared_flat_indices = np.flatnonzero(shared_valid)
    combined_nan_mask = mask_run1 | mask_run2
    return synced_run1, synced_run2, combined_nan_mask, shared_flat_indices

def combine_filtered_betas(beta_run1, beta_run2, nan_mask_flat, flat_indices):
    union_nan_mask = np.isnan(beta_run1) | np.isnan(beta_run2)
    if np.any(union_nan_mask):
        beta_run1[union_nan_mask] = np.nan
        beta_run2[union_nan_mask] = np.nan
    beta_combined = np.concatenate((beta_run1, beta_run2), axis=1)
    valid_voxel_mask = ~np.all(np.isnan(beta_combined), axis=1)
    removed_voxels = int(valid_voxel_mask.size - np.count_nonzero(valid_voxel_mask))
    if removed_voxels > 0:
        beta_combined = beta_combined[valid_voxel_mask]
        dropped_flat = flat_indices[~valid_voxel_mask]
        nan_mask_flat[dropped_flat] = True
    print(f"Combined filtered beta shape: {beta_combined.shape}", flush=True)
    print(f"Removed voxels with all-NaN trials: {removed_voxels}", flush=True)
    return beta_combined, nan_mask_flat

def combine_active_run_data(coords_run1, coords_run2, bold_run1, bold_run2, volume_shape, shared_nan_mask_flat):
    coords_run1 = tuple(axis for axis in coords_run1)
    coords_run2 = tuple(axis for axis in coords_run2)
    # Collapse (x, y, z) coordinates into single flat indices for practical set operations.
    flat_run1 = np.ravel_multi_index(coords_run1, volume_shape)
    flat_run2 = np.ravel_multi_index(coords_run2, volume_shape)

    # Only keep voxels that exist in both runs and are not masked out by NaNs.
    shared_flat = np.intersect1d(flat_run1, flat_run2, assume_unique=False)
    shared_flat = shared_flat[~shared_nan_mask_flat[shared_flat]]
    shared_coords = np.unravel_index(shared_flat, volume_shape)
    shared_coords = tuple(axis for axis in shared_coords)

    # Sorting once lets np.searchsorted perform vectorized index lookup back into each run.
    sorter1 = np.argsort(flat_run1)
    sorter2 = np.argsort(flat_run2)
    sorted_flat_run1 = flat_run1[sorter1]
    sorted_flat_run2 = flat_run2[sorter2]
    idx_run1 = sorter1[np.searchsorted(sorted_flat_run1, shared_flat)]
    idx_run2 = sorter2[np.searchsorted(sorted_flat_run2, shared_flat)]
    # Concatenate matching voxel time courses from the two runs to build the joint dataset.
    combined_bold = np.concatenate((bold_run1[idx_run1], bold_run2[idx_run2]), axis=1)
    return shared_flat, shared_coords, combined_bold

def mask_anatomical_image(anat_img, mask_flat, volume_shape):
    anat_data = anat_img.get_fdata()
    mask_flat = mask_flat.ravel()
    mask_3d = mask_flat.reshape(volume_shape)
    anat_data = np.where(mask_3d, np.nan, anat_data)
    return anat_data

def compute_distance_weighted_adjacency(coords, affine, radius_mm=3.0, sigma_mm=None):
    if sigma_mm is None:
        sigma_mm = radius_mm / 2.0
    coord_array = np.column_stack(coords).astype(np.float32, copy=False)
    coords_mm = nib.affines.apply_affine(affine, coord_array)
    tree = cKDTree(coords_mm)
    dist_matrix = tree.sparse_distance_matrix(tree, max_distance=radius_mm, output_type="coo_matrix")
    rows, cols, data = dist_matrix.row, dist_matrix.col, dist_matrix.data
    if data.size:
        off_diag = rows != cols
        rows, cols, data = rows[off_diag], cols[off_diag], data[off_diag]
    weights = np.exp(-(data ** 2) / (sigma_mm ** 2))
    adjacency = sparse.csr_matrix((weights, (rows, cols)), shape=(coords_mm.shape[0], coords_mm.shape[0]))
    if adjacency.nnz:
        adjacency = adjacency.maximum(adjacency.T)
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    degree_matrix = sparse.diags(degrees).tocsr()  # CSR keeps diagonal structure but allows slicing.
    return adjacency, degree_matrix
# %% 
ses = 1
sub = "09"
num_trials = 180
trial_len = 9
behave_indice = 1 #1/RT
trials_per_run = num_trials // 2

base_path = f"/scratch/st-mmckeown-1/zkavian/fmri_models/MSc-Thesis/sub{sub}"
brain_mask_path = f"{base_path}/sub-pd0{sub}_ses-{ses}_T1w_brain_mask.nii.gz"
csf_mask_path = f"{base_path}/sub-pd0{sub}_ses-{ses}_T1w_brain_pve_0.nii.gz"
gray_mask_path = f"{base_path}/sub-pd0{sub}_ses-{ses}_T1w_brain_pve_1.nii.gz"

# %%
anat_img = nib.load(f"{base_path}/sub-pd0{sub}_ses-{ses}_T1w_brain.nii.gz")
brain_mask = nib.load(brain_mask_path).get_fdata().astype(np.float16)
csf_mask = nib.load(csf_mask_path).get_fdata().astype(np.float16)
gray_mask = nib.load(gray_mask_path).get_fdata().astype(np.float16)

glm_dict = np.load(f"{base_path}/TYPED_FITHRF_GLMDENOISE_RR_sub{sub}.npy", allow_pickle=True).item()
beta_glm = glm_dict["betasmd"]

nan_mask_flat_run1 = np.load(f"sub{sub}/nan_mask_flat_sub{sub}_ses{ses}_run1.npy")
nan_mask_flat_run2 = np.load(f"sub{sub}/nan_mask_flat_sub{sub}_ses{ses}_run2.npy")
active_coords_run1 = np.load(f"sub{sub}/active_coords_sub{sub}_ses{ses}_run1.npy", allow_pickle=True)
active_coords_run2 = np.load(f"sub{sub}/active_coords_sub{sub}_ses{ses}_run2.npy", allow_pickle=True)
# Each `active_coords` array is the tuple of (x, y, z) indices saved during preprocessing for the voxels that stayed active after the statistical and Hampel filtering steps.

beta_volume_filtered_run1 = np.load(f"{base_path}/cleaned_beta_volume_sub{sub}_ses{ses}_run1.npy")
beta_volume_filtered_run2 = np.load(f"{base_path}/cleaned_beta_volume_sub{sub}_ses{ses}_run2.npy")
beta_filtered_run1  = np.load(f"sub{sub}/beta_volume_filter_sub{sub}_ses{ses}_run1.npy") 
beta_filtered_run2  = np.load(f"sub{sub}/beta_volume_filter_sub{sub}_ses{ses}_run2.npy") 
beta_filtered_run1, beta_filtered_run2, shared_nan_mask_flat, shared_flat_indices = synchronize_beta_voxels(beta_filtered_run1, beta_filtered_run2, nan_mask_flat_run1, nan_mask_flat_run2)
beta_clean, nan_mask_flat = combine_filtered_betas(beta_filtered_run1, beta_filtered_run2, shared_nan_mask_flat.ravel(), shared_flat_indices)

bold_data_run1 = np.load(join(base_path, f'fmri_sub{sub}_ses{ses}_run1.npy'))
bold_data_run2 = np.load(join(base_path, f'fmri_sub{sub}_ses{ses}_run2.npy'))
clean_active_bold_run1 = np.load(f"sub{sub}/active_bold_sub{sub}_ses{ses}_run1.npy")
clean_active_bold_run2 = np.load(f"sub{sub}/active_bold_sub{sub}_ses{ses}_run2.npy")
volume_shape = bold_data_run1.shape[:3]
active_flat_idx, active_coords, bold_clean = combine_active_run_data(active_coords_run1, active_coords_run2, clean_active_bold_run1, clean_active_bold_run2, bold_data_run1.shape[:3], shared_nan_mask_flat)
active_coords = np.asarray(active_coords)
print(f"Combined active_flat_idx: {active_flat_idx.shape}", flush=True)
print(f"Combined clean_active_bold: {bold_clean.shape}", flush=True)
print(f"Combined active_coords: {active_coords.shape}", flush=True)

# Mask anatomical volume to the functional voxel set and compute adjacency/degree matrices.
masked_anat = mask_anatomical_image(anat_img, shared_nan_mask_flat, volume_shape)
masked_active_anat = masked_anat.ravel()[active_flat_idx]
finite_anat = masked_active_anat[np.isfinite(masked_active_anat)]
anat_range = (float(np.min(finite_anat)), float(np.max(finite_anat))) if finite_anat.size else (np.nan, np.nan)
print(f"Masked anatomical volume shape: {masked_anat.shape}, active voxels: {masked_active_anat.size}", flush=True)
print(f"Masked anatomical intensity range (finite): [{anat_range[0]}, {anat_range[1]}]", flush=True)

adjacency_radius_mm = 3.0
adjacency_sigma_mm = adjacency_radius_mm / 2.0
adjacency_matrix, degree_matrix = compute_distance_weighted_adjacency(active_coords, anat_img.affine, radius_mm=adjacency_radius_mm, sigma_mm=adjacency_sigma_mm)
adj_data = adjacency_matrix.data
adj_range = (float(adj_data.min()), float(adj_data.max())) if adj_data.size else (0.0, 0.0)
degree_values = degree_matrix.data.ravel()
degree_range = (float(degree_values.min()), float(degree_values.max())) if degree_values.size else (0.0, 0.0)
print(f"Adjacency matrix (distance-weighted) shape: {adjacency_matrix.shape}, nnz={adjacency_matrix.nnz}, range: [{adj_range[0]}, {adj_range[1]}]", flush=True)
print(f"Degree matrix shape: {degree_matrix.shape}, range: [{degree_range[0]}, {degree_range[1]}]", flush=True)
laplacian_matrix = degree_matrix - adjacency_matrix

behav_data = loadmat(f"{base_path}/PSPD0{sub}_OFF_behav_metrics.mat" if ses == 1 else f"{base_path}/PSPD0{sub}_ON_behav_metrics.mat")
behav_metrics = behav_data["behav_metrics"]
behav_block = np.stack(behav_metrics[0], axis=0)
_, _, num_metrics = behav_block.shape
behavior_matrix = behav_block.reshape(-1, num_metrics)
print(f"Behavior matrix shape: {behavior_matrix.shape}")

# %%
def standardize_matrix(matrix):
    sym_array = 0.5 * (matrix + matrix.T)
    trace_value = np.trace(sym_array)
    if not np.isfinite(trace_value) or trace_value == 0:
        trace_value = 1.0
    normalized_mat = sym_array / trace_value * sym_array.shape[0]
    return normalized_mat

def select_empca_components(model, variance_threshold=0.8):
    explained = np.var(model.coeff, axis=0)
    total = explained.sum()
    if not np.isfinite(total) or total <= 0:
        print("EMPCA variance check failed; keeping all components.", flush=True)
        return model

    explained_ratio = explained / total
    cumulative_ratio = np.cumsum(explained_ratio)
    for idx, (var, frac, cum_frac) in enumerate(zip(explained, explained_ratio, cumulative_ratio), start=1):
        print(f"EMPCA component {idx:3d}: variance={var:.6f}, ratio={frac*100:.2f}%, cumulative={cum_frac*100:.2f}%", flush=True)

    target_components = int(np.searchsorted(cumulative_ratio, variance_threshold) + 1)
    target_components = min(target_components, model.nvec)
    print(f"Using {target_components} EMPCA components for {variance_threshold*100:.0f}% variance.", flush=True)

    model.nvec = target_components
    model.eigvec = model.eigvec[:target_components]
    model.coeff = model.coeff[:, :target_components]
    model.solve_model()
    model.dof = model.data[model._unmasked].size - model.eigvec.size - model.nvec * model.nobs
    return model

def apply_empca(bold_clean):
    def prepare_for_empca(data):
        W = np.isfinite(data)
        Y = np.where(np.isfinite(data), data, 0.0)
        row_weight = W.sum(axis=0, keepdims=True).astype(np.float64)
        mean = np.divide((Y * W).sum(axis=0, keepdims=True), row_weight, out=np.zeros_like(row_weight, dtype=np.float64), where=row_weight > 0)
        centered = Y - mean
        var = np.divide((W * centered**2).sum(axis=0, keepdims=True), row_weight, out=np.zeros_like(row_weight, dtype=np.float64), where=row_weight > 0)
        scale = np.sqrt(var)
        z = np.divide(centered, np.maximum(scale, 1e-6), out=np.zeros_like(centered), where=row_weight > 0)
        return z, W
    
    X_reshap = bold_clean.reshape(bold_clean.shape[0], -1)
    Yc, W = prepare_for_empca(X_reshap.T)
    Yc = Yc.T
    W = W.T
    print("begin empca...", flush=True)
    m = empca(Yc, W, nvec=250, niter=15)
    np.save(f'sub{sub}/empca_model_sub{sub}_ses{ses}.npy', m)
    return m

def load_or_fit_empca_model(bold_clean):
    model_path = f"sub{sub}/empca_model_sub{sub}_ses{ses}.npy"
    if os.path.exists(model_path):
        print(f"Loading existing EMPCA model from {model_path}", flush=True)
        m = np.load(model_path, allow_pickle=True).item()
        m = select_empca_components(m, variance_threshold=0.80)
        return m
    return apply_empca(bold_clean)

def build_pca_dataset(bold_clean, beta_clean, behavioral_matrix, nan_mask_flat, active_coords, active_flat_indices, trial_length, num_trials):
    pca_model = load_or_fit_empca_model(bold_clean)
    bold_pca_components = pca_model.eigvec
    bold_pca_trials = bold_pca_components.reshape(bold_pca_components.shape[0], num_trials, trial_length)
    
    coeff_pinv = np.linalg.pinv(pca_model.coeff)
    beta_pca_full = coeff_pinv @ np.nan_to_num(beta_clean)

    normalization = _compute_global_normalization(behavioral_matrix, beta_pca_full, behave_indice)

    return {"pca_model": pca_model, "bold_pca_trials": bold_pca_trials, "coeff_pinv": coeff_pinv, "beta_pca_full": beta_pca_full, 
            "bold_clean": bold_clean, "beta_clean": beta_clean, "behavior_matrix": behavioral_matrix, "nan_mask_flat": nan_mask_flat, 
            "active_coords": active_coords, "active_flat_indices": active_flat_indices, **normalization}

def _compute_global_normalization(behavior_matrix, beta_pca_full, behavior_index):
    # behavior_vector = behavior_matrix[:, behavior_index].ravel()
    raw_behavior = behavior_matrix[:, behavior_index].ravel()
    inv_behavior = np.divide(1.0, raw_behavior, out=np.full_like(raw_behavior, np.nan, dtype=np.float64),
                             where=np.isfinite(raw_behavior) & (raw_behavior != 0))
    inv_max = np.nanmax(inv_behavior) if np.any(np.isfinite(inv_behavior)) else 0.0
    behavior_vector = inv_behavior - inv_max
    behavior_finite_mask = np.isfinite(behavior_vector)

    behavior_centered_full = np.full_like(behavior_vector, np.nan, dtype=np.float64)
    behavior_mean = np.nanmean(behavior_vector[behavior_finite_mask])
    behavior_centered_full[behavior_finite_mask] = behavior_vector[behavior_finite_mask] - behavior_mean
    behavior_norm = np.linalg.norm(behavior_centered_full[behavior_finite_mask])
    if not np.isfinite(behavior_norm) or behavior_norm <= 0:
        behavior_norm = 1.0

    behavior_normalized_full = np.full_like(behavior_vector, np.nan, dtype=np.float64)
    behavior_normalized_full[behavior_finite_mask] = behavior_centered_full[behavior_finite_mask] / behavior_norm

    behavior_mask = behavior_finite_mask[None, :]
    beta_valid_mask = np.isfinite(beta_pca_full) & behavior_mask
    beta_counts = beta_valid_mask.sum(axis=1, keepdims=True)
    beta_sum = np.nansum(np.where(beta_valid_mask, beta_pca_full, 0.0), axis=1, keepdims=True)
    beta_mean = np.divide(beta_sum, beta_counts, out=np.zeros_like(beta_sum, dtype=np.float64), where=beta_counts > 0)
    beta_centered_full = beta_pca_full - beta_mean

    return {"behavior_vector_full": behavior_vector, "behavior_centered_full": behavior_centered_full,
            "behavior_normalized_full": behavior_normalized_full, "behavior_mean": behavior_mean,
            "behavior_norm": behavior_norm, "beta_centered_full": beta_centered_full, "beta_mean_full": beta_mean}

def compute_fold_normalization(projection_data, train_indices, behavior_index):
    """Center/scale behaviors (and beta means) using only the training trials, then apply the scale to all trials."""
    behavior_vector_full = projection_data["behavior_vector_full"]
    beta_pca_full = projection_data["beta_pca_full"]

    total_trials = behavior_vector_full.size
    train_mask = np.zeros(total_trials, dtype=bool)
    train_mask[np.asarray(train_indices, dtype=int)] = True
    behavior_mask = np.isfinite(behavior_vector_full)
    behavior_train_mask = train_mask & behavior_mask

    behavior_mean = float(np.nanmean(behavior_vector_full[behavior_train_mask])) if np.any(behavior_train_mask) else 0.0
    behavior_centered_full = np.full_like(behavior_vector_full, np.nan, dtype=np.float64)
    behavior_centered_full[behavior_mask] = behavior_vector_full[behavior_mask] - behavior_mean
    behavior_norm = float(np.linalg.norm(behavior_centered_full[behavior_train_mask])) if np.any(behavior_train_mask) else 1.0
    if not np.isfinite(behavior_norm) or behavior_norm <= 0:
        behavior_norm = 1.0

    behavior_normalized_full = np.full_like(behavior_vector_full, np.nan, dtype=np.float64)
    behavior_normalized_full[behavior_mask] = behavior_centered_full[behavior_mask] / behavior_norm

    beta_valid_train = np.isfinite(beta_pca_full) & behavior_train_mask[None, :]
    beta_counts = beta_valid_train.sum(axis=1, keepdims=True)
    beta_sum = np.nansum(np.where(beta_valid_train, beta_pca_full, 0.0), axis=1, keepdims=True)
    beta_mean = np.divide(beta_sum, beta_counts, out=np.zeros_like(beta_sum, dtype=np.float64), where=beta_counts > 0)
    beta_centered_full = beta_pca_full - beta_mean

    return {
        "behavior_vector_full": behavior_vector_full,
        "behavior_centered_full": behavior_centered_full,
        "behavior_normalized_full": behavior_normalized_full,
        "behavior_mean": behavior_mean,
        "behavior_norm": behavior_norm,
        "beta_centered_full": beta_centered_full,
        "beta_mean_full": beta_mean,
    }

def build_loss_corr_term(beta_components, normalized_behaviors, ridge=1e-6):
    beta_components = np.nan_to_num(beta_components, nan=0.0, posinf=0.0, neginf=0.0)
    behavior_vector = normalized_behaviors.reshape(1, -1)
    behavior_projection = beta_components @ behavior_vector.T
    C_b = behavior_projection @ behavior_projection.T
    C_d = beta_components @ beta_components.T
    skew_b = C_b - C_b.T
    skew_d = C_d - C_d.T
    # print(
    #     f"C_b symmetry deviation: fro_norm={np.linalg.norm(skew_b, ord='fro'):.6e}, max_abs={np.max(np.abs(skew_b)):.6e}",
    #     flush=True,
    # )
    # print(
    #     f"C_d symmetry deviation: fro_norm={np.linalg.norm(skew_d, ord='fro'):.6e}, max_abs={np.max(np.abs(skew_d)):.6e}",
    #     flush=True,
    # )
    C_b = 0.5 * (C_b + C_b.T)
    C_d = 0.5 * (C_d + C_d.T)
    eye = np.eye(beta_components.shape[0])
    C_b = C_b + ridge * eye
    C_d = C_d + ridge * eye
    return C_b, C_d

def _safe_pearsonr(x, y, return_p=False):
    finite_mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite_mask], y[finite_mask]
    sample_size = x.size
    x -= x.mean()
    y -= y.mean()
    denom = np.sqrt(np.dot(x, x) * np.dot(y, y))
    correlation = np.dot(x, y) / denom
    if not return_p:
        return correlation
    r = float(np.clip(correlation, -0.999999, 0.999999))
    dof = sample_size - 2
    t_statistic = r * np.sqrt(dof) / np.sqrt(max(1e-12, 1.0 - r**2))
    p_value = student_t.sf(abs(t_statistic), dof)
    return correlation, float(p_value)

def _aggregate_trials(active_bold):
    # downsample bold data by reducer metrice
    num_voxels, num_trials, trial_length = active_bold.shape
    reshaped = active_bold.reshape(-1, trial_length)
    finite_counts = np.count_nonzero(np.isfinite(reshaped), axis=1)
    valid_mask = finite_counts > 0
    reduced_flat = np.full(reshaped.shape[0], np.nan)

    if np.any(valid_mask):
        valid_values = reshaped[valid_mask]
        reduced_values = np.nanmedian(valid_values, axis=-1)
        reduced_flat[valid_mask] = reduced_values
    return reduced_flat.reshape(num_voxels, num_trials)

def _compute_matrix_icc(data):
    """ICC(A,1) reliability across folds (raters) and voxels (targets) using pingouin."""
    n_folds, n_vox = data.shape
    fold_idx, voxel_idx = np.meshgrid(np.arange(n_folds, dtype=int), np.arange(n_vox, dtype=int), indexing="ij")
    df = pd.DataFrame({"targets": voxel_idx.ravel(), "raters": fold_idx.ravel(), "ratings": data.ravel()})
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["ratings"])
    icc_table = pg.intraclass_corr(data=df, targets="targets", raters="raters", ratings="ratings")
    icc2 = icc_table.loc[icc_table["Type"] == "ICC2", "ICC"]
    if icc2.empty or not np.isfinite(icc2.iloc[0]):
        return np.nan
    return icc2.iloc[0]

def save_brain_map(correlations, active_coords, volume_shape, anat_img, file_prefix, result_prefix="active_bold_corr", map_title=None):
    coord_arrays = tuple(np.asarray(axis, dtype=int) for axis in active_coords)
    display_values = np.abs(correlations.ravel())
    volume = np.full(volume_shape, np.nan)
    colorbar_max = None
    if coord_arrays and coord_arrays[0].size:
        volume[coord_arrays] = display_values
        finite_values = display_values[np.isfinite(display_values)]
        colorbar_max = float(np.percentile(finite_values, 99.5))

    corr_img = nib.Nifti1Image(volume, anat_img.affine, anat_img.header)
    volume_path = f"{result_prefix}_{file_prefix}.nii.gz"
    nib.save(corr_img, volume_path)

    if np.any(np.isfinite(display_values)):
        display_volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
        display_volume = np.clip(display_volume, 0.0, colorbar_max)
        display_img = nib.Nifti1Image(display_volume, anat_img.affine, anat_img.header)
        display = plotting.view_img(display_img, bg_img=anat_img, colorbar=True, symmetric_cmap=False, cmap='jet', title=map_title)
        display.save_as_html(f"{result_prefix}_{file_prefix}.html")
    return

def _load_projection_series(series_path):
    if not os.path.exists(series_path):
        return {}
    loaded = np.load(series_path, allow_pickle=True)
    if isinstance(loaded, np.ndarray):
        data = loaded.item()
    else:
        data = dict(loaded)
    series = {}
    for key, values in data.items():
        gamma_key = float(key)
        series[gamma_key] = values
    return series

def _merge_projection_series(existing_series, new_series):
    merged = dict(existing_series)
    for gamma_value, projection in new_series:
        if projection is None:
            continue
        merged[float(gamma_value)] = np.asarray(projection, dtype=np.float64).ravel()
    return merged

#%% 
def calcu_penalty_terms(run_data, alpha_task, alpha_bold, alpha_beta, alpha_smooth):
    beta_centered = run_data["beta_centered"]
    n_components = beta_centered.shape[0]

    task_penalty = alpha_task * run_data["C_task"]
    bold_penalty = alpha_bold * run_data["C_bold"]
    beta_penalty = alpha_beta * run_data["C_beta"]
    smooth_matrix = run_data.get("C_smooth")
    skew_smooth = smooth_matrix - smooth_matrix.T
    # print(f"C_smooth symmetry deviation: fro_norm={np.linalg.norm(skew_smooth, ord='fro'):.6e}, max_abs={np.max(np.abs(skew_smooth)):.6e}",flush=True)
    smooth_penalty = alpha_smooth * smooth_matrix

    total_penalty = task_penalty + bold_penalty + beta_penalty + smooth_penalty
    total_penalty = 0.5 * (total_penalty + total_penalty.T)
    total_penalty = total_penalty + 1e-6 * np.eye(n_components)
    return {"task": task_penalty, "bold": bold_penalty, "beta": beta_penalty, "smooth": smooth_penalty}, total_penalty

# def _corr_ratio_and_components(run_data, weights, eps=1e-8):
#     weights = np.asarray(weights, dtype=np.float64)
#     beta_centered = run_data["beta_centered"]
#     behavior_vec = run_data["normalized_behaviors"]
#     y = beta_centered.T @ weights
#     corr_num = float(np.dot(y, behavior_vec))
#     corr_den = float(np.dot(y, y) + eps)
#     corr_ratio = (corr_num ** 2) / corr_den if np.isfinite(corr_den) and corr_den > 0 else np.inf
#     return corr_ratio, corr_num, corr_den, y

# def compute_total_loss(run_data, weights, total_penalty, gamma_value, penalty_terms=None, label=None, eps=1e-8):
#     weights = np.asarray(weights, dtype=np.float64)
#     corr_ratio, corr_num, corr_den, _ = _corr_ratio_and_components(run_data, weights, eps=eps)
#     penalty_value = float(weights.T @ total_penalty @ weights)
#     if penalty_terms:
#         label_prefix = f"[{label}] " if label else ""
#         for term_label, matrix in penalty_terms.items():
#             contribution = float(weights.T @ matrix @ weights)
#             print(f"{label_prefix}{term_label}_penalty: {contribution:.6f}", flush=True)
#     total_loss = gamma_value * penalty_value - corr_ratio
#     return total_loss, penalty_value, corr_ratio, corr_num, corr_den

def _build_objective_matrices(total_penalty, corr_num, corr_den, gamma_value, eps=1e-8):
    A_mat = gamma_value * total_penalty - corr_num
    A_mat = 0.5 * (A_mat + A_mat.T)
    B_mat = 0.5 * (corr_den + corr_den.T)
    eye = np.eye(A_mat.shape[0], dtype=np.float64)
    return A_mat + eps * eye, B_mat + eps * eye

def _total_loss_from_penalty(weights, total_penalty, gamma_value, corr_num=None, corr_den=None, penalty_terms=None, label=None, A_mat=None, B_mat=None):
    weights = np.where(np.isfinite(weights), weights, 0.0)
    A_mat, B_mat = _build_objective_matrices(total_penalty, corr_num, corr_den, gamma_value)
    numerator = float(weights.T @ A_mat @ weights)
    denominator = float(weights.T @ B_mat @ weights)
    if penalty_terms:
        label_prefix = f"[{label}] " if label else ""
        for term_label, matrix in penalty_terms.items():
            contribution = float(weights.T @ matrix @ weights)
            print(f"{label_prefix}{term_label}_penalty: {contribution:.6f}", flush=True)
    return numerator / denominator

def evaluate_projection_corr(data, weights):
    #corr(W*beta, behave)
    beta_centered = data["beta_centered"]
    behavior_centered = data["behavior_centered"]
    normalized_behaviors = data["normalized_behaviors"]

    Y = beta_centered.T @ weights
    finite_mask = np.isfinite(Y) & np.isfinite(behavior_centered)
    Y = Y[finite_mask]
    behavior_centered = behavior_centered[finite_mask]
    normalized_behaviors = normalized_behaviors[finite_mask]

    metrics = {"pearson": np.nan, "pearson_p": np.nan}
    metrics["pearson"], metrics["pearson_p"] = _safe_pearsonr(Y, behavior_centered, return_p=True)
    return metrics

def _compute_projection(weights, mat):
    if mat.ndim != 2:
        mat = mat.reshape(mat.shape[0], -1)
    finite_mask = np.isfinite(mat)
    weighted = mat * weights[:, None]
    projection = np.where(np.any(finite_mask, axis=0), np.nansum(weighted, axis=0), np.nan)
    return projection

def _compute_bold_projection(voxel_weights, data):
    active_bold = data.get("active_bold")
    bold_matrix = active_bold.reshape(active_bold.shape[0], -1).copy()
    voxel_means = np.nanmean(bold_matrix, axis=1, keepdims=True)
    voxel_means = np.where(np.isfinite(voxel_means), voxel_means, 0.0)
    bold_matrix -= voxel_means
    projection = _compute_projection(voxel_weights, bold_matrix)
    return projection, bold_matrix

def evaluate_beta_bold_projection_corr(voxel_weights, data):
    #corr(w*beta, bold)
    active_beta = data.get("beta_clean")
    active_bold = data.get("active_bold")
    bold_trial_metric = _aggregate_trials(active_bold)
    projection = _compute_projection(voxel_weights, active_beta)
    correlations = np.full(active_beta.shape[0], np.nan)
    projection_finite = np.isfinite(projection)

    for voxel_idx, voxel_series in enumerate(bold_trial_metric):
        voxel_finite = np.isfinite(voxel_series)
        joint_mask = projection_finite & voxel_finite
        if np.count_nonzero(joint_mask) < 2:
            continue
        correlations[voxel_idx] = _safe_pearsonr(projection[joint_mask], voxel_series[joint_mask])
    return projection, correlations

def evaluate_bold_bold_projection_corr(voxel_weights, data):
    # corr(weight * bold, bold)
    projection, bold_matrix = _compute_bold_projection(voxel_weights, data)
    correlations = np.full(bold_matrix.shape[0], np.nan)
    projection_finite = np.isfinite(projection)

    for voxel_idx, voxel_series in enumerate(bold_matrix):
        voxel_finite = np.isfinite(voxel_series)
        joint_mask = projection_finite & voxel_finite
        if np.count_nonzero(joint_mask) < 2:
            continue
        correlations[voxel_idx] = _safe_pearsonr(projection[joint_mask], voxel_series[joint_mask])
    return projection, correlations

#%%
def build_custom_kfold_splits(total_trials, num_folds=5, trials_per_run=None):
    if trials_per_run is None:
        trials_per_run = total_trials // 2
    base_fold = trials_per_run // num_folds
    remainder = trials_per_run % num_folds
    fold_sizes = np.full(num_folds, base_fold, dtype=int)
    if remainder > 0:
        fold_sizes[:remainder] += 1

    run1_folds = []
    run2_folds = []
    run_start = 0
    for block_size in fold_sizes:
        run1_start = run_start
        run1_end = run1_start + block_size
        run2_start = trials_per_run + run1_start
        run2_end = run2_start + block_size

        run1_folds.append(np.arange(run1_start, run1_end, dtype=np.int64))
        run2_folds.append(np.arange(run2_start, run2_end, dtype=np.int64))
        run_start = run1_end

    folds = []
    all_trials = np.arange(total_trials, dtype=np.int64)
    fold_id = 1
    for run1_fold_idx, test_run1 in enumerate(run1_folds, start=1):
        for run2_fold_idx, test_run2 in enumerate(run2_folds, start=1):
            test_indices = np.concatenate((test_run1, test_run2))
            train_mask = np.ones(total_trials, dtype=bool)
            train_mask[test_indices] = False
            train_indices = all_trials[train_mask]
            folds.append({"fold_id": fold_id, "train_indices": train_indices, "test_indices": test_indices, 
                          "test_run1": test_run1, "test_run2": test_run2, "run1_fold": run1_fold_idx, "run2_fold": run2_fold_idx})
            fold_id += 1
    return folds

def calcu_matrices_func(beta_pca, bold_pca, behave_mat, behave_indice, trial_len, num_trials, trial_indices, run_boundary):
    bold_pca_reshape = bold_pca.reshape(bold_pca.shape[0], num_trials, trial_len)
    behavior_selected = behave_mat[:, behave_indice]

    counts = np.count_nonzero(np.isfinite(beta_pca), axis=-1)
    sums = np.nansum(np.abs(beta_pca), axis=-1)
    mean_beta = np.zeros(beta_pca.shape[0])
    mask = counts > 0
    mean_beta[mask] = (sums[mask] / counts[mask])
    C_task = np.zeros_like(mean_beta)
    valid = np.abs(mean_beta) > 0
    C_task[valid] = (1.0 / mean_beta[valid])
    C_task = np.diag(C_task)

    C_bold = np.zeros((bold_pca_reshape.shape[0], bold_pca_reshape.shape[0]))
    bold_transition_count = 0
    for i in range(num_trials - 1):
        idx_current = trial_indices[i]
        idx_next = trial_indices[i + 1]
        if (idx_next - idx_current) != 1:
            continue
        if (idx_current < run_boundary) and (idx_next >= run_boundary):
            continue
        x1 = bold_pca_reshape[:, i, :]
        x2 = bold_pca_reshape[:, i + 1, :]
        C_bold += (x1 - x2) @ (x1 - x2).T
        bold_transition_count += 1
    if bold_transition_count > 0:
        C_bold /= bold_transition_count

    C_beta = np.zeros((beta_pca.shape[0], beta_pca.shape[0]))
    beta_transition_count = 0
    for i in range(num_trials - 1):
        idx_current = trial_indices[i]
        idx_next = trial_indices[i + 1]
        if (idx_next - idx_current) != 1:
            continue
        if (idx_current < run_boundary) and (idx_next >= run_boundary):
            continue
        x1 = beta_pca[:, i]
        x2 = beta_pca[:, i + 1]
        diff = x1 - x2
        C_beta += np.outer(diff, diff)
        beta_transition_count += 1
    if beta_transition_count > 0:
        C_beta /= beta_transition_count

    return C_task, C_bold, C_beta, behavior_selected

def prepare_data_func(projection_data, trial_indices, trial_length, run_boundary, normalization_info):
    effective_num_trials = trial_indices.size
    behavior_subset = projection_data["behavior_matrix"][trial_indices]
    behavior_vector_full = normalization_info["behavior_vector_full"]
    behavior_centered_full_all = normalization_info["behavior_centered_full"]
    behavior_normalized_full_all = normalization_info["behavior_normalized_full"]
    beta_centered_full_all = normalization_info["beta_centered_full"]
    beta_pca_full = projection_data["beta_pca_full"]

    behavior_vector = behavior_vector_full[trial_indices]
    behavior_centered_full = behavior_centered_full_all[trial_indices]
    behavior_normalized_full = behavior_normalized_full_all[trial_indices]
    beta_centered_full = beta_centered_full_all
    beta_pca = beta_pca_full[:, trial_indices]
    beta_centered_subset = beta_centered_full[:, trial_indices]
    bold_pca_trials = projection_data["bold_pca_trials"][:, trial_indices, :]
    bold_pca_components = bold_pca_trials.reshape(bold_pca_trials.shape[0], effective_num_trials * trial_length)

    (C_task, C_bold, C_beta, _) = calcu_matrices_func(beta_pca, bold_pca_components, behavior_subset, behave_indice,
                                                      trial_len=trial_length, num_trials=effective_num_trials,
                                                      trial_indices=trial_indices, run_boundary=run_boundary)
    skew_task = C_task - C_task.T
    skew_bold = C_bold - C_bold.T
    skew_beta = C_beta - C_beta.T
    # print(f"C_task symmetry deviation: fro_norm={np.linalg.norm(skew_task, ord='fro'):.6e}, max_abs={np.max(np.abs(skew_task)):.6e}",flush=True)
    # print(f"C_bold symmetry deviation: fro_norm={np.linalg.norm(skew_bold, ord='fro'):.6e}, max_abs={np.max(np.abs(skew_bold)):.6e}",flush=True)
    # print(f"C_beta symmetry deviation: fro_norm={np.linalg.norm(skew_beta, ord='fro'):.6e}, max_abs={np.max(np.abs(skew_beta)):.6e}",flush=True)
    C_task, C_bold, C_beta = standardize_matrix(C_task), standardize_matrix(C_bold), standardize_matrix(C_beta)
    
    trial_mask = np.isfinite(behavior_vector)
    behavior_observed = behavior_vector[trial_mask]
    behavior_centered = behavior_centered_full[trial_mask]
    normalized_behaviors = behavior_normalized_full[trial_mask]

    beta_observed = beta_pca[:, trial_mask]
    beta_centered = beta_centered_subset[:, trial_mask]
    # Correlation term (r^2) — keep the ratio intact, but rescale both matrices by a shared scalar
    # so their magnitude is comparable to the (standardized) penalty matrices. Using a common scale
    # preserves w^T C_corr_num w / w^T C_corr_den w while preventing tiny denominators.
    C_corr_num, C_corr_den = build_loss_corr_term(beta_centered, normalized_behaviors)
    corr_scale = float(np.trace(C_corr_den)) / max(1, C_corr_den.shape[0])
    if not np.isfinite(corr_scale) or corr_scale <= 0:
        corr_scale = 1.0
    C_corr_num = C_corr_num / corr_scale
    C_corr_den = C_corr_den / corr_scale

    beta_subset = projection_data["beta_clean"][:, trial_indices]
    bold_subset = projection_data["bold_clean"][:, trial_indices, :]

    return {"nan_mask_flat": projection_data["nan_mask_flat"],
        "active_coords": tuple(np.array(coord) for coord in projection_data["active_coords"]),
        "active_flat_indices": projection_data["active_flat_indices"],
        "coeff_pinv": projection_data["coeff_pinv"], "beta_centered": beta_centered,
        "behavior_centered": behavior_centered, "normalized_behaviors": normalized_behaviors,
        "behavior_observed": behavior_observed, "beta_observed": beta_observed,
        "beta_clean": beta_subset, "C_task": C_task, "C_bold": C_bold, "C_beta": C_beta,
        "C_smooth": projection_data.get("C_smooth"),
        "C_corr_num": C_corr_num, "C_corr_den": C_corr_den,
        "active_bold": bold_subset, "trial_indices": trial_indices, "num_trials": effective_num_trials}

def solve_soc_problem(run_data, alpha_task, alpha_bold, alpha_beta, alpha_smooth, gamma):
    beta_centered = run_data["beta_centered"]
    n_components = beta_centered.shape[0]
    penalty_matrices, total_penalty = calcu_penalty_terms(run_data, alpha_task, alpha_bold, alpha_beta, alpha_smooth)

    corr_num, corr_den = run_data["C_corr_num"], run_data["C_corr_den"]
    A_mat, B_mat = _build_objective_matrices(total_penalty, corr_num, corr_den, gamma)

    def _objective_with_grad(weights):
        numerator = weights.T @ A_mat @ weights
        denominator = weights.T @ B_mat @ weights
        grad_num = 2.0 * (A_mat @ weights)
        grad_den = 2.0 * (B_mat @ weights)
        objective_value = numerator / denominator
        gradient = (grad_num * denominator - grad_den * numerator) / (denominator**2)
        return objective_value, gradient
    def _objective(weights):
        value, _ = _objective_with_grad(weights)
        return value
    def _objective_grad(weights):
        _, grad = _objective_with_grad(weights)
        return grad

    constraints = [{"type": "eq", "fun": lambda w: np.linalg.norm(w) - 1.0, "jac": lambda w: w / (np.linalg.norm(w) + 1e-12)}]
    initial_weights = np.full(n_components, 1.0 / np.sqrt(n_components), dtype=np.float64)
    result = minimize(_objective, initial_weights, jac=_objective_grad, constraints=constraints, 
                      method="SLSQP", options={"maxiter": 1000, "ftol": 1e-8, "disp": True})
    if not result.success:
        raise RuntimeError(f"Fractional optimizer failed: {result.message}")
    solution_weights = np.asarray(result.x)
    denominator_value = solution_weights.T @ B_mat @ solution_weights
    if denominator_value < 1e-6:
        raise RuntimeError("Optimized weights violate the correlation denominator constraint.")

    contributions = {label: solution_weights.T @ matrix @ solution_weights for label, matrix in penalty_matrices.items()}
    y = beta_centered.T @ solution_weights
    penalty_value = float(solution_weights.T @ total_penalty @ solution_weights)
    gamma_penalty_value = float(gamma * penalty_value)
    gamma_penalty_ratio = gamma_penalty_value / denominator_value
    numerator_value = solution_weights.T @ A_mat @ solution_weights
    correlation_numerator = solution_weights.T @ corr_num @ solution_weights
    total_loss = _total_loss_from_penalty(solution_weights, total_penalty, gamma, corr_num=corr_num, corr_den=corr_den, A_mat=A_mat, B_mat=B_mat)
    fractional_objective = numerator_value / denominator_value
    correlation_ratio = correlation_numerator / denominator_value

    return {"weights": solution_weights, "total_loss": total_loss, "Y": y, "fractional_objective": fractional_objective,
            "numerator": numerator_value, "denominator": denominator_value, "penalty_contributions": contributions, "corr_ratio": correlation_ratio,
            "penalty_value": penalty_value, "gamma_penalty": gamma_penalty_value, "gamma_penalty_ratio": gamma_penalty_ratio}

#%%

def plot_projection_bold(y_trials, task_alpha, bold_alpha, beta_alpha, gamma_value, output_path,
                         max_trials=10, series_label=None, trial_indices_array=None, trial_windows=None):
    num_trials_total, trial_length = y_trials.shape
    time_axis = np.arange(trial_length)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    if trial_windows is None:
        window_span = 10
        first_start = 10 if num_trials_total > 20 else max(0, num_trials_total // 4)
        second_start = num_trials_total // 2
        def _build_window(start_idx):
            start_idx = int(max(0, min(start_idx, max(num_trials_total - 1, 0))))
            end_idx = min(start_idx + window_span, num_trials_total)
            if end_idx <= start_idx:
                end_idx = min(num_trials_total, start_idx + max(1, window_span))
            return start_idx, end_idx
        first_window = _build_window(first_start)
        second_window = _build_window(max(second_start, first_window[1]))
        trial_windows = [first_window, second_window]

    def _get_light_colors(count):
        cmap = plt.cm.get_cmap("tab20", count)
        raw_colors = cmap(np.linspace(0, 1, count))
        pastel = raw_colors * 0.55 + 0.45
        return np.clip(pastel, 0.0, 1.0)

    for ax, (start, end) in zip(axes, trial_windows):
        start_idx = min(start, num_trials_total)
        end_idx = min(end, num_trials_total)
        if start_idx >= end_idx:
            ax.axis("off")
            continue
        window_trials = y_trials[start_idx:end_idx]
        trials_to_plot = window_trials[:max_trials]
        colors = _get_light_colors(trials_to_plot.shape[0])
        for trial_offset, (trial_values, color) in enumerate(zip(trials_to_plot, colors)):
            if trial_indices_array is not None:
                global_idx = int(trial_indices_array[start_idx + trial_offset]) + 1
            else:
                global_idx = start_idx + trial_offset + 1
            label = f"Trial {global_idx}"
            ax.plot(time_axis, trial_values, linewidth=1.0, color=color, alpha=0.9, label=label)

        ax.set_ylabel("BOLD projection")
        if trial_indices_array is not None:
            label_start = int(trial_indices_array[start_idx]) + 1
            label_end = int(trial_indices_array[end_idx - 1]) + 1
        else:
            label_start = start_idx + 1
            label_end = end_idx
        ax.set_title(f"Trials {label_start}-{label_end}")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend(loc="upper right", fontsize=8, ncol=2)

    axes[-1].set_xlabel("Time point")
    if series_label:
        figure_title = f"{series_label} BOLD projection"
    else:
        figure_title = "BOLD projection"
    hyperparam_label = f"alpha_task={task_alpha:g}, alpha_bold={bold_alpha:g}, alpha_beta={beta_alpha:g}, gamma={gamma_value:g}"
    fig.suptitle(f"{figure_title}\n{hyperparam_label}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path

def plot_projection_beta_sweep(gamma_projection_series, task_alpha, bold_alpha, beta_alpha, output_path, series_label=None, trial_indices=None):
    sorted_series = sorted(gamma_projection_series, key=lambda item: item[0])
    max_trials = max(np.asarray(series, dtype=np.float64).ravel().size for _, series in sorted_series)
    fig_height = 5.0 if max_trials <= 120 else 6.5
    trial_indices_array = None
    if trial_indices is not None:
        trial_indices_array = np.asarray(trial_indices, dtype=np.int64).ravel()
    axis_reference = None
    if trial_indices_array is not None and trial_indices_array.size:
        axis_reference = trial_indices_array[:min(trial_indices_array.size, max_trials)] + 1
    use_global_axis = axis_reference is not None
    cmap = plt.cm.get_cmap("tab10", len(sorted_series))

    fig, ax = plt.subplots(figsize=(10, fig_height))
    for idx, (gamma_value, projection) in enumerate(sorted_series):
        y_trials = np.asarray(projection, dtype=np.float64).ravel()
        finite_vals = y_trials[np.isfinite(y_trials)]
        cv_val = np.nanstd(finite_vals) / abs(np.nanmean(finite_vals))
        if trial_indices_array is not None:
                axis_values = np.arange(y_trials.size)
                axis_values = trial_indices_array[: y_trials.size] + 1
        else:
            axis_values = np.arange(y_trials.size)
        variability_label = f"gamma={gamma_value:g} (cv={cv_val})"
        ax.plot(axis_values, y_trials, linewidth=1.2, alpha=0.9, label=variability_label, color=cmap(idx))

    axis_label = "Trial (global index)" if use_global_axis else "Trial index"
    ax.set_xlabel(axis_label)
    ax.set_ylabel("Voxel-space projection")
    label_suffix = f" [{series_label}]" if series_label else ""
    gamma_labels = ", ".join(f"{gamma:g}" for gamma, _ in sorted_series)
    ax.set_title(f"y_beta across gammas ({gamma_labels}) | alpha_task={task_alpha:g}, alpha_bold={bold_alpha:g}, alpha_beta={beta_alpha:g}{label_suffix}")
    if max_trials > 1:
        approx_ticks = 10
        if use_global_axis:
            tick_count = min(approx_ticks, axis_reference.size)
            tick_indices = np.linspace(0, axis_reference.size - 1, num=tick_count, dtype=int)
            xtick_positions = axis_reference[tick_indices]
            ax.set_xticks(xtick_positions)
        else:
            tick_step = max(1, max_trials // approx_ticks)
            xtick_positions = np.arange(0, max_trials, tick_step, dtype=int)
            if xtick_positions[-1] != max_trials - 1:
                xtick_positions = np.append(xtick_positions, max_trials - 1)
            ax.set_xticks(xtick_positions)
            ax.set_xticklabels([str(pos + 1) for pos in xtick_positions])
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path

def plot_fold_metric_box(fold_values, title, ylabel, output_path, highlight_threshold=None):
    sorted_values = sorted(fold_values, key=lambda item: item[0])
    fold_ids = np.array([int(idx) for idx, _ in sorted_values], dtype=np.int64)
    raw_values = np.array([float(val) if np.isfinite(val) else np.nan for _, val in sorted_values], dtype=np.float64)
    finite_mask = np.isfinite(raw_values)
    finite_values = raw_values[finite_mask]
    finite_ids = fold_ids[finite_mask]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    box = ax.boxplot([finite_values], vert=True, patch_artist=True, widths=0.3,
                     boxprops={"facecolor": "#b5cde0", "color": "#4c72b0", "linewidth": 1.5},
                     medianprops={"color": "#c44e52", "linewidth": 2.0},
                     whiskerprops={"color": "#4c72b0"}, capprops={"color": "#4c72b0"},
                     flierprops={"marker": "o", "markerfacecolor": "#c44e52", "markeredgecolor": "#c44e52", "markersize": 5})
    x_center = 1.0
    if finite_values.size > 1:
        jitter = np.linspace(-0.07, 0.07, finite_values.size)
    else:
        jitter = np.array([0.0])
    scatter_positions = x_center + jitter
    ax.scatter(scatter_positions, finite_values, color="#4c72b0", alpha=0.7, s=25, zorder=3, label="Fold values")
    for fold_id, x_pos, y_val in zip(finite_ids, scatter_positions, finite_values):
        ax.text(x_pos, y_val, f"{fold_id}", fontsize=8, color="#1a1a1a", ha="center", va="bottom", rotation=0, clip_on=True)

    ax.set_xticks([x_center])
    ax.set_xticklabels([f"{finite_values.size} folds"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    use_scalar_formatter = True
    if finite_values.size:
        max_abs = np.nanmax(np.abs(finite_values))
        if np.isfinite(max_abs) and max_abs < 0.1:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
            use_scalar_formatter = False

    if use_scalar_formatter:
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    if highlight_threshold is not None:
        ax.axhline(highlight_threshold, color="#c44e52", linestyle="--", linewidth=1.0, label=f"threshold={highlight_threshold:g}")
    ax.legend(loc="best")

    nan_count = np.count_nonzero(~finite_mask)
    if nan_count:
        ax.annotate(f"{nan_count} fold(s) skipped (NaN)", xy=(0.02, 0.95), xycoords="axes fraction",fontsize=8, color="#c44e52", ha="left", va="top")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path

def plot_train_test_total_loss_box(train_values, test_values, title, ylabel, output_path):
    def _prepare(entries):
        sorted_entries = sorted(entries, key=lambda item: item[0])
        fold_ids = np.array([int(idx) for idx, _ in sorted_entries])
        raw_values = np.array([float(val) if np.isfinite(val) else np.nan for _, val in sorted_entries])
        finite_mask = np.isfinite(raw_values)
        return fold_ids[finite_mask], raw_values[finite_mask], np.count_nonzero(~finite_mask)

    train_ids, train_vals, train_nan = _prepare(train_values)
    test_ids, test_vals, test_nan = _prepare(test_values)

    plot_data = []
    positions = []
    labels = []
    colors = []
    scatter_data = []
    position_cursor = 1

    if train_vals.size:
        plot_data.append(train_vals)
        positions.append(position_cursor)
        labels.append(f"Train ({train_vals.size} folds)")
        colors.append("#4c72b0")
        scatter_data.append((position_cursor, train_ids, train_vals, "#4c72b0"))
        position_cursor += 1

    if test_vals.size:
        plot_data.append(test_vals)
        positions.append(position_cursor)
        labels.append(f"Test ({test_vals.size} folds)")
        colors.append("#dd8452")
        scatter_data.append((position_cursor, test_ids, test_vals, "#dd8452"))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    box = ax.boxplot(plot_data, positions=positions, vert=True, patch_artist=True, widths=0.35)

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("#1a1a1a")
    for median in box["medians"]:
        median.set_color("#1a1a1a")
        median.set_linewidth(2.0)
    for whisker in box["whiskers"]:
        whisker.set_color("#1a1a1a")
    for cap in box["caps"]:
        cap.set_color("#1a1a1a")
    for flier, color in zip(box["fliers"], colors):
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor(color)

    for x_pos, fold_ids, values, color in scatter_data:
        if values.size > 1:
            jitter = np.linspace(-0.07, 0.07, values.size)
        else:
            jitter = np.array([0.0])
        scatter_positions = x_pos + jitter
        ax.scatter(scatter_positions, values, color=color, alpha=0.75, s=25, zorder=3)
        for fold_id, x_coord, y_val in zip(fold_ids, scatter_positions, values):
            ax.text(x_coord, y_val, f"{fold_id}", fontsize=8, color="#1a1a1a",
                    ha="center", va="bottom", rotation=0, clip_on=True)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    all_values = np.concatenate(plot_data) if plot_data else np.array([], dtype=np.float64)
    use_scalar_formatter = True
    if all_values.size:
        max_abs = np.nanmax(np.abs(all_values))
        if np.isfinite(max_abs) and max_abs < 0.1:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
            use_scalar_formatter = False
    if use_scalar_formatter:
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    skipped = train_nan + test_nan
    if skipped:
        ax.annotate(f"{skipped} value(s) skipped (NaN)", xy=(0.02, 0.95), xycoords="axes fraction",
                    fontsize=8, color="#c44e52", ha="left", va="top")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path

def save_projection_outputs(pca_weights, bold_pca_components, trial_length, file_prefix,
                            task_alpha, bold_alpha, beta_alpha, gamma_value, voxel_weights, beta_clean,
                            data=None, bold_projection=None, plot_trials=True):
    bold_projection_signal, _ = _compute_bold_projection(voxel_weights, data)
    num_trials = bold_projection_signal.size // trial_length
    y_trials = bold_projection_signal.reshape(num_trials, trial_length)
    trial_indices = None
    if data is not None:
        trial_indices = data.get("trial_indices")
    if plot_trials:
        plot_path = f"y_projection_trials_{file_prefix}.png"
        plot_projection_bold(y_trials, task_alpha, bold_alpha, beta_alpha, gamma_value, plot_path, series_label="Active BOLD space", trial_indices_array=trial_indices)

    voxel_weights = voxel_weights.ravel()
    beta_matrix = np.nan_to_num(beta_clean, nan=0.0, posinf=0.0, neginf=0.0)
    y_projection_voxel_trials = voxel_weights @ beta_matrix
    y_projection_voxel = y_projection_voxel_trials.ravel()
    return y_projection_voxel

# # %%
ridge_penalty = 1e-3
solver_name = "MOSEK"
# penalty_sweep = [(0.8, 1, 0.5, 1.2), (0, 1, 0.5, 1.2), (0.8, 0, 0.5, 1.2), (0.8, 1, 0, 1.2), (0.8, 1, 0.5, 0)]
# penalty_sweep = [(0.5, 0.7, 0.25, 0.8)]
penalty_sweep = [(0.8, 1, 0.5, 1.2)]
alpha_sweep = [{"task_penalty": task_alpha, "bold_penalty": bold_alpha, "beta_penalty": beta_alpha, "smooth_penalty": smooth_alpha} for task_alpha, bold_alpha, beta_alpha, smooth_alpha in penalty_sweep]
# gamma_sweep = [0, 0.2, 0.8]
gamma_sweep = [2]
SAVE_PER_FOLD_VOXEL_MAPS = False  # disable individual fold voxel-weight plots; averages saved later

projection_data = build_pca_dataset(bold_clean, beta_clean, behavior_matrix, nan_mask_flat, active_coords, active_flat_idx, trial_len, num_trials)
coeff_pinv = projection_data["coeff_pinv"]
projected = coeff_pinv @ (laplacian_matrix @ coeff_pinv.T)
projection_data["C_smooth"] = standardize_matrix(projected)

def run_cross_run_experiment(alpha_settings, gamma_values, fold_splits, projection_data):
    total_folds = len(fold_splits)
    bold_pca_components = projection_data["pca_model"].eigvec
    aggregate_metrics = defaultdict(
        lambda: {
            "train_corr": [],
            "train_p": [],
            "test_corr": [],
            "test_p": [],
            "train_total_loss": [],
            "test_total_loss": [],
            "train_gamma_ratio": [],
            "test_gamma_ratio": [],
        }
    )
    fold_metric_records = defaultdict(lambda: defaultdict(list))
    fold_output_tracker = defaultdict(lambda: {"component_weights": [], "voxel_weights": [], "bold_corr": [], "beta_corr": []})
    metric_plot_configs = {"train_corr": {"title": "Train correlation", "ylabel": "Corr"}, "test_corr": {"title": "Test correlation", "ylabel": "Corr"},
                           "train_p": {"title": "Train correlation p-value", "ylabel": "p-value", "threshold": 0.05},
                           "test_p": {"title": "Test correlation p-value", "ylabel": "p-value", "threshold": 0.05}}
    for fold_idx, split in enumerate(fold_splits, start=1):
        print(f"\n===== Fold {fold_idx}/{total_folds} =====", flush=True)
        test_indices, train_indices = split["test_indices"], split["train_indices"]
        run1_desc = (int(split["test_run1"][0] + 1), int(split["test_run1"][-1] + 1))
        run2_desc = (int(split["test_run2"][0] + 1), int(split["test_run2"][-1] + 1))
        print(f"Test trials (1-based): run1 {run1_desc[0]}-{run1_desc[1]}, run2 {run2_desc[0]}-{run2_desc[1]}", flush=True)
        normalization_info = compute_fold_normalization(projection_data, train_indices, behave_indice)
        train_data = prepare_data_func(projection_data, train_indices, trial_len, trials_per_run, normalization_info)
        test_data = prepare_data_func(projection_data, test_indices, trial_len, trials_per_run, normalization_info)

        train_corr_norms = {"num": _matrix_norm_summary(train_data["C_corr_num"]), "den": _matrix_norm_summary(train_data["C_corr_den"])}
        test_corr_norms = {"num": _matrix_norm_summary(test_data["C_corr_num"]), "den": _matrix_norm_summary(test_data["C_corr_den"])}
        print(f"  Corr matrix norms (train) num_fro={train_corr_norms['num']['fro']:.4f}, den_fro={train_corr_norms['den']['fro']:.4f}; "
            f"(test) num_fro={test_corr_norms['num']['fro']:.4f}, den_fro={test_corr_norms['den']['fro']:.4f}", flush=True)

        for alpha_setting in alpha_settings:
            task_alpha, bold_alpha, beta_alpha, smooth_alpha = alpha_setting["task_penalty"], alpha_setting["bold_penalty"], alpha_setting["beta_penalty"], alpha_setting["smooth_penalty"]
            alpha_prefix = f"fold{fold_idx}_sub{sub}_ses{ses}_task{task_alpha:g}_bold{bold_alpha:g}_beta{beta_alpha:g}_smooth{smooth_alpha:g}"

            for gamma_value in gamma_values:
                metrics_key = (task_alpha, bold_alpha, beta_alpha, smooth_alpha, gamma_value)
                combo_label = (f"task={task_alpha:g}, bold={bold_alpha:g}, beta={beta_alpha:g}, smooth={smooth_alpha:g}, gamma={gamma_value:g}")
                file_prefix = f"{alpha_prefix}_gamma{gamma_value:g}"
                print(f"Fold {fold_idx}: optimization with task={task_alpha}, bold={bold_alpha}, beta={beta_alpha}, smooth={smooth_alpha}, gamma={gamma_value}", flush=True)
                try:
                    solution = solve_soc_problem(train_data, task_alpha, bold_alpha, beta_alpha, smooth_alpha, gamma_value)
                except Exception as exc:
                    print(f"  Fold {fold_idx}: optimization failed for {combo_label} -> {exc}", flush=True)
                    continue
                numerator_value = solution.get("numerator")
                denominator_value = solution.get("denominator")
                print(f"  Objective terms -> numerator: {numerator_value:.6f}, denominator: {denominator_value:.6f}", flush=True)

                penalty_contributions = solution.get("penalty_contributions", {})
                loss_report = (f"C_task: {penalty_contributions.get('task')}, " f"C_bold: {penalty_contributions.get('bold')}, "
                               f"C_beta: {penalty_contributions.get('beta')}, " f"C_smooth: {penalty_contributions.get('smooth')}")
                print(f"  Loss terms -> {loss_report}", flush=True)

                # Preserve the optimized weight signs so correlation metrics match the optimized objective
                component_weights = np.asarray(solution["weights"], dtype=np.float64)
                coeff_pinv = np.asarray(train_data["coeff_pinv"])
                voxel_weights = coeff_pinv.T @ component_weights
                if SAVE_PER_FOLD_VOXEL_MAPS:
                    save_brain_map(voxel_weights * 1000, train_data.get("active_coords"), anat_img.shape[:3], anat_img, file_prefix, result_prefix="voxel_weights")

                projection_signal, voxel_correlations = evaluate_bold_bold_projection_corr(voxel_weights, train_data)
                beta_projection_signal, beta_voxel_correlations = evaluate_beta_bold_projection_corr(voxel_weights, train_data)

                y_projection_voxel = save_projection_outputs(component_weights, bold_pca_components, trial_len, file_prefix,
                                                             task_alpha, bold_alpha, beta_alpha, gamma_value,
                                                             voxel_weights=voxel_weights, beta_clean=train_data["beta_clean"],
                                                             data=train_data, bold_projection=projection_signal, plot_trials=False) 
                fold_outputs = fold_output_tracker[metrics_key]
                fold_outputs["component_weights"].append(np.abs(component_weights))
                fold_outputs["voxel_weights"].append(np.abs(voxel_weights))
                fold_outputs["bold_corr"].append(np.abs(voxel_correlations))
                fold_outputs["beta_corr"].append(np.abs(beta_voxel_correlations))

                train_metrics = evaluate_projection_corr(train_data, component_weights)
                test_metrics = evaluate_projection_corr(test_data, component_weights)

                train_penalties, total_penalty_train = calcu_penalty_terms(train_data, task_alpha, bold_alpha, beta_alpha, smooth_alpha)
                train_A, train_B = _build_objective_matrices(total_penalty_train, train_data["C_corr_num"], train_data["C_corr_den"], gamma_value)
                train_total_loss = solution.get("total_loss")
                train_numerator = float(component_weights.T @ train_A @ component_weights)
                train_denominator = float(component_weights.T @ train_B @ component_weights)
                train_corr_num_quad = float(component_weights.T @ train_data["C_corr_num"] @ component_weights)
                train_gamma_penalty = float(gamma_value * (component_weights.T @ total_penalty_train @ component_weights))
                train_gamma_ratio = train_gamma_penalty / train_denominator if np.isfinite(train_denominator) and train_denominator > 0 else np.nan
                train_corr_ratio = train_corr_num_quad / train_denominator if np.isfinite(train_denominator) and train_denominator > 0 else np.nan

                test_penalties, total_penalty_test = calcu_penalty_terms(test_data, task_alpha, bold_alpha, beta_alpha, smooth_alpha)
                test_total_loss = _total_loss_from_penalty(solution["weights"], total_penalty_test, gamma_value, corr_num=test_data["C_corr_num"], 
                                                           corr_den=test_data["C_corr_den"], penalty_terms=test_penalties, label="test")
                test_A, test_B = _build_objective_matrices(total_penalty_test, test_data["C_corr_num"], test_data["C_corr_den"], gamma_value)
                test_numerator = float(component_weights.T @ test_A @ component_weights)
                test_denominator = float(component_weights.T @ test_B @ component_weights)
                test_corr_num_quad = float(component_weights.T @ test_data["C_corr_num"] @ component_weights)
                test_gamma_penalty = float(gamma_value * (component_weights.T @ total_penalty_test @ component_weights))
                test_gamma_ratio = test_gamma_penalty / test_denominator if np.isfinite(test_denominator) and test_denominator > 0 else np.nan
                test_corr_ratio = test_corr_num_quad / test_denominator if np.isfinite(test_denominator) and test_denominator > 0 else np.nan

                train_metrics["total_loss"] = train_total_loss
                test_metrics["total_loss"] = test_total_loss

                print(f"  Total loss (train objective): {train_total_loss:.6f}", flush=True)
                print(f"  Total loss (test objective):  {test_total_loss:.6f}", flush=True)
                print(f"  Gamma-penalty ratio -> train: {train_gamma_ratio:.6f}, test: {test_gamma_ratio:.6f}", flush=True)
                print(f"  Train metrics -> corr: {train_metrics['pearson']:.4f}", flush=True)
                print(f"  Test metrics  -> corr: {test_metrics['pearson']:.4f},", flush=True)

                input_stats = {"beta_clean": _array_summary(train_data["beta_clean"]),
                    "active_bold": _array_summary(train_data["active_bold"]),
                    "behavior_observed": _array_summary(train_data["behavior_observed"]),
                    "beta_centered": _array_summary(train_data["beta_centered"]),
                    "normalized_behaviors": _array_summary(train_data["normalized_behaviors"])}
                weight_stats = {"component_weights": _array_summary(component_weights), "voxel_weights": _array_summary(voxel_weights)}
                log_entry = {"fold": fold_idx,
                    "run1_test_span": run1_desc,
                    "run2_test_span": run2_desc,
                    "alphas": {"task": task_alpha, "bold": bold_alpha, "beta": beta_alpha, "smooth": smooth_alpha},
                    "gamma": gamma_value,
                    "numerator": numerator_value,
                    "denominator": denominator_value,
                    "fractional_objective": solution.get("fractional_objective"),
                    "gamma_penalty": solution.get("gamma_penalty"),
                    "gamma_penalty_ratio": solution.get("gamma_penalty_ratio"),
                    "train_total_loss": train_total_loss,
                    "test_total_loss": test_total_loss,
                    "train_gamma_ratio": train_gamma_ratio,
                    "test_gamma_ratio": test_gamma_ratio,
                    "train_corr": float(train_metrics.get("pearson", np.nan)),
                    "test_corr": float(test_metrics.get("pearson", np.nan)),
                    "train_numerator": train_numerator,
                    "train_denominator": train_denominator,
                    "train_corr_ratio": train_corr_ratio,
                    "test_numerator": test_numerator,
                    "test_denominator": test_denominator,
                    "test_corr_ratio": test_corr_ratio,
                    "train_gamma_penalty": train_gamma_penalty,
                    "test_gamma_penalty": test_gamma_penalty,
                    "train_corr_num_quad": train_corr_num_quad,
                    "test_corr_num_quad": test_corr_num_quad,
                    "penalty_contributions": {k: float(v) for k, v in penalty_contributions.items()},
                    "train_corr_norms": train_corr_norms,
                    "test_corr_norms": test_corr_norms,
                    "input_stats": input_stats,
                    "weight_stats": weight_stats}
                _append_run_log(log_entry)

                metrics_key = (task_alpha, bold_alpha, beta_alpha, smooth_alpha, gamma_value)
                bucket = aggregate_metrics[metrics_key]
                bucket["train_corr"].append(np.abs(train_metrics["pearson"]))
                bucket["train_p"].append(train_metrics["pearson_p"])
                bucket["test_corr"].append(np.abs(test_metrics["pearson"]))
                bucket["test_p"].append(test_metrics["pearson_p"])
                bucket["train_total_loss"].append(train_total_loss)
                bucket["test_total_loss"].append(test_total_loss)
                bucket["train_gamma_ratio"].append(train_gamma_ratio)
                bucket["test_gamma_ratio"].append(test_gamma_ratio)

                fold_metrics = fold_metric_records[metrics_key]
                fold_metrics["train_corr"].append((fold_idx, np.abs(train_metrics["pearson"])))
                fold_metrics["train_p"].append((fold_idx, train_metrics["pearson_p"]))
                fold_metrics["test_corr"].append((fold_idx, np.abs(test_metrics["pearson"])))
                fold_metrics["test_p"].append((fold_idx, test_metrics["pearson_p"]))
                fold_metrics["train_total_loss"].append((fold_idx, train_total_loss))
                fold_metrics["test_total_loss"].append((fold_idx, test_total_loss))
                fold_metrics["train_gamma_ratio"].append((fold_idx, train_gamma_ratio))
                fold_metrics["test_gamma_ratio"].append((fold_idx, test_gamma_ratio))

    if fold_metric_records:
        print("\n===== Saving fold-wise metric box plots =====", flush=True)
        for metrics_key in sorted(fold_metric_records.keys()):
            task_alpha, bold_alpha, beta_alpha, smooth_alpha, gamma_value = metrics_key
            combo_label = (f"task={task_alpha:g}, bold={bold_alpha:g}, beta={beta_alpha:g}, smooth={smooth_alpha:g}, gamma={gamma_value:g}")
            metrics_prefix = (f"foldmetrics_sub{sub}_ses{ses}_task{task_alpha:g}_bold{bold_alpha:g}_beta{beta_alpha:g}_smooth{smooth_alpha:g}_gamma{gamma_value:g}")
            
            metric_records = fold_metric_records[metrics_key]
            train_loss_entries = metric_records.get("train_total_loss", [])
            test_loss_entries = metric_records.get("test_total_loss", [])
            for metric_name, entries in metric_records.items():
                if metric_name in ("train_total_loss", "test_total_loss", "train_gamma_ratio", "test_gamma_ratio"):
                    continue
                config = metric_plot_configs.get(metric_name)
                plot_title = f"{config['title']} across folds ({combo_label})"
                plot_path = f"{metric_name}_{metrics_prefix}.png"
                created_path = plot_fold_metric_box(entries, plot_title, config["ylabel"], plot_path, highlight_threshold=config.get("threshold"))
                if created_path:
                    print(f"  Saved {metric_name} fold plot for {combo_label} -> {created_path}", flush=True)
            if train_loss_entries or test_loss_entries:
                loss_title = f"Total loss across folds ({combo_label})"
                loss_path = f"total_loss_{metrics_prefix}_train_vs_test.png"
                created_loss_plot = plot_train_test_total_loss_box(train_loss_entries, test_loss_entries, loss_title, "Objective value", loss_path)
                if created_loss_plot:
                    print(f"  Saved train/test total loss fold plot for {combo_label} -> {created_loss_plot}", flush=True)

            train_gamma_entries = metric_records.get("train_gamma_ratio", [])
            test_gamma_entries = metric_records.get("test_gamma_ratio", [])
            if train_gamma_entries or test_gamma_entries:
                gamma_title = f"Gamma-penalty ratio across folds ({combo_label})"
                gamma_path = f"gamma_penalty_ratio_{metrics_prefix}_train_vs_test.png"
                created_gamma_plot = plot_train_test_total_loss_box(
                    train_gamma_entries,
                    test_gamma_entries,
                    gamma_title,
                    "gamma * penalty / corr_den",
                    gamma_path,
                )
                if created_gamma_plot:
                    print(f"  Saved gamma-penalty ratio fold plot for {combo_label} -> {created_gamma_plot}", flush=True)

    if fold_output_tracker:
        print("\n===== Saving fold-averaged spatial maps and projections =====", flush=True)
        active_coords = projection_data.get("active_coords")
        volume_shape = anat_img.shape[:3]
        fold_avg_projection_series = defaultdict(list)

        for metrics_key in sorted(fold_output_tracker.keys()):
            task_alpha, bold_alpha, beta_alpha, smooth_alpha, gamma_value = metrics_key
            fold_outputs = fold_output_tracker[metrics_key]
            avg_prefix = f"foldavg_sub{sub}_ses{ses}_task{task_alpha:g}_bold{bold_alpha:g}_beta{beta_alpha:g}_smooth{smooth_alpha:g}_gamma{gamma_value:g}"

            comp_weights_stack = np.stack(fold_outputs["component_weights"], axis=0)
            comp_weights_icc = _compute_matrix_icc(comp_weights_stack)
            print(f"  ICC (component weights across folds/components): {comp_weights_icc:.4f}" if np.isfinite(comp_weights_icc) else "  ICC (component weights) could not be computed", flush=True)

            weights_stack = np.stack(fold_outputs["voxel_weights"], axis=0)
            weights_avg = np.nanmean(weights_stack, axis=0)
            weights_icc = _compute_matrix_icc(weights_stack)
            print(f"  ICC (weights across folds/voxels): {weights_icc:.4f}" if np.isfinite(weights_icc) else "  ICC (weights) could not be computed", flush=True)
            weights_title = f"ICC={weights_icc:.3f}"
            save_brain_map(weights_avg * 1000, active_coords, volume_shape, anat_img, avg_prefix, result_prefix="voxel_weights_mean", map_title=weights_title)

            bold_corr_stack = np.stack(np.abs(fold_outputs["bold_corr"]), axis=0)
            bold_corr_avg = np.nanmean(bold_corr_stack, axis=0)
            bold_corr_icc = _compute_matrix_icc(bold_corr_stack)
            print(f"  ICC (bold corr across folds/voxels): {bold_corr_icc:.4f}" if np.isfinite(bold_corr_icc) else "  ICC (bold corr) could not be computed", flush=True)
            bold_title = f"ICC={bold_corr_icc:.3f}"
            save_brain_map(bold_corr_avg, active_coords, volume_shape, anat_img, avg_prefix, result_prefix="active_bold_corr_mean", map_title=bold_title)

            beta_corr_stack = np.stack(np.abs(fold_outputs["beta_corr"]), axis=0)
            beta_corr_avg = np.nanmean(beta_corr_stack, axis=0)
            beta_corr_icc = _compute_matrix_icc(beta_corr_stack)
            print(f"  ICC (beta corr across folds/voxels): {beta_corr_icc:.4f}" if np.isfinite(beta_corr_icc) else "  ICC (beta corr) could not be computed", flush=True)
            beta_title = f"ICC={beta_corr_icc:.3f}"
            save_brain_map(beta_corr_avg, active_coords, volume_shape, anat_img, avg_prefix, result_prefix=f"active_beta_corr_mean", map_title=beta_title)

            projection_avg = _compute_projection(weights_avg, projection_data["beta_clean"])
            avg_series_key = (task_alpha, bold_alpha, beta_alpha, smooth_alpha)
            fold_avg_projection_series[avg_series_key].append((gamma_value, projection_avg))

            bold_clean_full = projection_data.get("bold_clean")
            if bold_clean_full is not None:
                bold_projection_signal, _ = _compute_bold_projection(weights_avg, {"active_bold": bold_clean_full})
                num_trials = bold_projection_signal.size // trial_len
                bold_projection_trials= bold_projection_signal.reshape(num_trials, trial_len)
                bold_plot_path = f"y_projection_bold_{avg_prefix}.png"
                plot_projection_bold(bold_projection_trials, task_alpha, bold_alpha, beta_alpha, gamma_value, bold_plot_path, 
                                     series_label="Active BOLD space (weights avg)", trial_indices_array=np.arange(num_trials, dtype=np.int64))

        if fold_avg_projection_series:
            for avg_series_key in sorted(fold_avg_projection_series.keys()):
                task_alpha, bold_alpha, beta_alpha, smooth_alpha = avg_series_key
                gamma_series = fold_avg_projection_series[avg_series_key]
                avg_base_prefix = f"foldavg_sub{sub}_ses{ses}_task{task_alpha:g}_bold{bold_alpha:g}_beta{beta_alpha:g}_smooth{smooth_alpha:g}"
                avg_series_storage = f"gamma_series_voxel_{avg_base_prefix}.npy"
                existing_avg = _load_projection_series(avg_series_storage)
                merged_avg = _merge_projection_series(existing_avg, gamma_series)
                np.save(avg_series_storage, merged_avg, allow_pickle=True)
                aggregate_plot_path = f"y_projection_trials_voxel_{avg_base_prefix}_all_gammas.png"
                merged_avg_sorted = sorted(merged_avg.items(), key=lambda item: item[0])
                plot_projection_beta_sweep(merged_avg_sorted, task_alpha, bold_alpha, beta_alpha, aggregate_plot_path, series_label="Voxel space (fold avg)")

    if aggregate_metrics:
        print("\n===== Cross-fold average metrics =====", flush=True)
        for metrics_key in sorted(aggregate_metrics.keys()):
            task_alpha, bold_alpha, beta_alpha, smooth_alpha, gamma_value = metrics_key
            bucket = aggregate_metrics[metrics_key]

            train_corr_mean = np.nanmean(np.abs(bucket["train_corr"]))
            train_p_mean = np.nanmean(bucket["train_p"])
            test_corr_mean = np.nanmean(np.abs(bucket["test_corr"]))
            test_p_mean = np.nanmean(bucket["test_p"])
            train_loss_mean = np.nanmean(bucket["train_total_loss"])
            test_loss_mean = np.nanmean(bucket["test_total_loss"])
            train_gamma_ratio_mean = np.nanmean(bucket["train_gamma_ratio"])
            test_gamma_ratio_mean = np.nanmean(bucket["test_gamma_ratio"])
            fold_count = len(bucket["test_corr"])

            combo_label = (f"task={task_alpha:g}\n bold={bold_alpha:g}\n beta={beta_alpha:g}\n smooth={smooth_alpha:g}\n gamma={gamma_value:g}")
            print(f"task={task_alpha:g}, bold={bold_alpha:g}, beta={beta_alpha:g}, smooth={smooth_alpha:g}, gamma={gamma_value:g} "
                f"(folds contributing={fold_count}): "
                f"train corr={train_corr_mean:.4f} (p={train_p_mean:.4f}), "
                f"test corr={test_corr_mean:.4f} (p={test_p_mean:.4f}), "
                f"avg loss train={train_loss_mean:.4f}, test={test_loss_mean:.4f}, "
                f"gamma_ratio train={train_gamma_ratio_mean:.4f}, test={test_gamma_ratio_mean:.4f}", flush=True)


DEFAULT_NUM_FOLDS = 5


if __name__ == "__main__":
    combo_idx = None
    combo_env = os.environ.get("FMRI_COMBO_IDX") or os.environ.get("SLURM_ARRAY_TASK_ID")
    if combo_env not in (None, ""):
        combo_idx = int(combo_env)

    selected_alphas = alpha_sweep
    selected_gammas = gamma_sweep
    if combo_idx is not None:
        total = len(alpha_sweep) * len(gamma_sweep)
        alpha_idx, gamma_idx = divmod(combo_idx, len(gamma_sweep))
        selected_alphas = [alpha_sweep[alpha_idx]]
        selected_gammas = [gamma_sweep[gamma_idx]]
        print(f"Running combo_idx={combo_idx} (alpha_idx={alpha_idx}, gamma_idx={gamma_idx})", flush=True)
    else:
        print("Running full alpha/gamma sweep (no combo index provided).", flush=True)

    num_folds_env = os.environ.get("FMRI_NUM_FOLDS", "").strip()
    if num_folds_env:
        num_folds = int(num_folds_env)
    else:
        num_folds = DEFAULT_NUM_FOLDS

    fold_splits = build_custom_kfold_splits(num_trials, num_folds=num_folds, trials_per_run=trials_per_run)
    print(f"Constructed {len(fold_splits)} fold pairs using num_folds={num_folds} per run.", flush=True)
    run_cross_run_experiment(selected_alphas, selected_gammas, fold_splits, projection_data)
