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
from scipy.linalg import eigh
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
    kept_flat_indices = flat_indices[valid_voxel_mask]
    if removed_voxels > 0:
        beta_combined = beta_combined[valid_voxel_mask]
        dropped_flat = flat_indices[~valid_voxel_mask]
        nan_mask_flat[dropped_flat] = True
    print(f"Combined filtered beta shape: {beta_combined.shape}", flush=True)
    print(f"Removed voxels with all-NaN trials: {removed_voxels}", flush=True)
    return beta_combined, nan_mask_flat, kept_flat_indices

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

# Save a quick visualization of the mean beta activation (averaged over trials) as an interactive HTML.


base_path = f"/scratch/st-mmckeown-1/zkavian/fmri_models/MSc-Thesis/sub{sub}"
brain_mask_path = f"{base_path}/sub-pd0{sub}_ses-{ses}_T1w_brain_mask.nii.gz"
csf_mask_path = f"{base_path}/sub-pd0{sub}_ses-{ses}_T1w_brain_pve_0.nii.gz"
gray_mask_path = f"{base_path}/sub-pd0{sub}_ses-{ses}_T1w_brain_pve_1.nii.gz"

# %%
anat_img = nib.load(f"{base_path}/sub-pd0{sub}_ses-{ses}_T1w_brain.nii.gz")
brain_mask = nib.load(brain_mask_path).get_fdata().astype(np.float16)
csf_mask = nib.load(csf_mask_path).get_fdata().astype(np.float16)
gray_mask = nib.load(gray_mask_path).get_fdata().astype(np.float16)

# glm_dict = np.load(f"{base_path}/TYPED_FITHRF_GLMDENOISE_RR_sub{sub}_ses{ses}.npy", allow_pickle=True).item()
# beta_glm = glm_dict["betasmd"]

nan_mask_flat_run1 = np.load(join(base_path, f"nan_mask_flat_sub{sub}_ses{ses}_run1.npy"))
nan_mask_flat_run2 = np.load(join(base_path, f"nan_mask_flat_sub{sub}_ses{ses}_run2.npy"))
active_coords_run1 = np.load(join(base_path, f"active_coords_sub{sub}_ses{ses}_run1.npy"), allow_pickle=True)
active_coords_run2 = np.load(join(base_path, f"active_coords_sub{sub}_ses{ses}_run2.npy"), allow_pickle=True)
# Each `active_coords` array is the tuple of (x, y, z) indices saved during preprocessing for the voxels that stayed active after the statistical and Hampel filtering steps.

beta_volume_filtered_run1 = np.load(f"{base_path}/cleaned_beta_volume_sub{sub}_ses{ses}_run1.npy")
beta_volume_filtered_run2 = np.load(f"{base_path}/cleaned_beta_volume_sub{sub}_ses{ses}_run2.npy")
beta_filtered_run1  = np.load(join(base_path, f"beta_volume_filter_sub{sub}_ses{ses}_run1.npy"))
beta_filtered_run2  = np.load(join(base_path, f"beta_volume_filter_sub{sub}_ses{ses}_run2.npy"))
beta_filtered_run1, beta_filtered_run2, shared_nan_mask_flat, shared_flat_indices = synchronize_beta_voxels(beta_filtered_run1, beta_filtered_run2, nan_mask_flat_run1, nan_mask_flat_run2)
beta_clean, nan_mask_flat, beta_clean_flat_idx = combine_filtered_betas(
    beta_filtered_run1, beta_filtered_run2, shared_nan_mask_flat.ravel(), shared_flat_indices
)

bold_data_run1 = np.load(join(base_path, f'fmri_sub{sub}_ses{ses}_run1.npy'))
bold_data_run2 = np.load(join(base_path, f'fmri_sub{sub}_ses{ses}_run2.npy'))
clean_active_bold_run1 = np.load(join(base_path, f"active_bold_sub{sub}_ses{ses}_run1.npy"))
clean_active_bold_run2 = np.load(join(base_path, f"active_bold_sub{sub}_ses{ses}_run2.npy"))
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
# Quick visualization: mean beta activation overlaid on subject anatomical.
beta_mean = np.nanmean(np.abs(beta_clean), axis=1)
beta_vol = np.full(int(np.prod(volume_shape)), np.nan, dtype=np.float32)
beta_vol[beta_clean_flat_idx] = beta_mean.astype(np.float32, copy=False)
beta_img = nib.Nifti1Image(
    np.nan_to_num(np.abs(beta_vol.reshape(volume_shape)), nan=0.0, posinf=0.0, neginf=0.0),
    anat_img.affine,
    anat_img.header,
)

out_html = join(base_path, f"beta_clean_mean_sub{sub}_ses{ses}.html")
abs_p99 = float(np.nanpercentile(np.abs(beta_mean), 99)) if np.isfinite(beta_mean).any() else None
anat_data = anat_img.get_fdata()
bg_vmin = float(np.nanpercentile(anat_data, 2)) if np.isfinite(anat_data).any() else None
bg_vmax = float(np.nanpercentile(anat_data, 98)) if np.isfinite(anat_data).any() else None
view = plotting.view_img(
    beta_img,
    bg_img=anat_img,
    cmap="jet",
    symmetric_cmap=False,
    vmax=abs_p99,
    threshold=0.0,
    opacity=0.6,
)
view.save_as_html(out_html)
print(f"Saved beta_clean mean overlay to: {out_html}", flush=True)

# %%
# fig, ax
# fig, ax = plt.subplots(2, 2, figsize=(8, 6))
# ax[0,0].plot(beta_clean[])
# beta_mean = np.nanmean(np.abs(beta_clean), axis=1)
# beta_vol = np.full(int(np.prod(volume_shape)), np.nan, dtype=np.float32)
# beta_vol[beta_clean_flat_idx] = beta_mean.astype(np.float32, copy=False)
# trial_idx = 10
for trial_idx in range(90):
    trial_beta = beta_clean[:, trial_idx]
    trial_vol = np.full(int(np.prod(volume_shape)), np.nan, dtype=np.float32)
    trial_vol[beta_clean_flat_idx] = np.nan_to_num(trial_beta, nan=0.0, posinf=0.0, neginf=0.0)
    trial_vol = trial_vol.reshape(volume_shape)
    # trial_vol = np.clip(trial_vol, 0, 4)
    trial_data = np.abs(trial_vol)
    beta_img = nib.Nifti1Image(trial_data, anat_img.affine, anat_img.header)
    out_html = join(base_path, f"beta_clean_mean_sub{sub}_ses{ses}_trial{trial_idx}.html")
    # abs_p99 = np.nanpercentile(trial_beta, 99)
    abs_trial_beta = np.abs(trial_beta)
    vmin = float(np.nanpercentile(abs_trial_beta, 2)) if np.isfinite(abs_trial_beta).any() else None
    vmax = float(np.nanpercentile(abs_trial_beta, 98)) if np.isfinite(abs_trial_beta).any() else None
    view = plotting.view_img(
        nib.Nifti1Image(np.nan_to_num(trial_data, nan=0.0, posinf=0.0, neginf=0.0), anat_img.affine, anat_img.header),
        bg_img=anat_img,
        cmap="jet",
        symmetric_cmap=False,
        threshold=0.0,
        opacity=0.6,
        vmax=vmax,
        vmin=vmin,
    )
    view.save_as_html(out_html)


# trial_idx = 50
# trial_beta = beta_clean[:, trial_idx]
# trial_vol = np.zeros(int(np.prod(volume_shape)), dtype=np.float32)
# trial_vol[beta_clean_flat_idx] = np.nan_to_num(trial_beta, nan=0.0, posinf=0.0, neginf=0.0).astype(
#     np.float32, copy=False
# )
# trial_vol = np.clip(trial_vol, 0, 4)
# beta_img = nib.Nifti1Image(trial_vol.reshape(volume_shape), anat_img.affine, anat_img.header)
# out_html = join(base_path, f"beta_clean_mean_sub{sub}_ses{ses}_trial51.html")
# abs_p99 = float(np.nanpercentile(np.abs(trial_beta), 99)) if np.isfinite(trial_beta).any() else None
# anat_data = anat_img.get_fdata()
# bg_vmin = float(np.nanpercentile(anat_data, 2)) if np.isfinite(anat_data).any() else None
# bg_vmax = float(np.nanpercentile(anat_data, 98)) if np.isfinite(anat_data).any() else None
# view = plotting.view_img(
#     beta_img,
#     bg_img=anat_img,
#     cmap="jet",
#     symmetric_cmap=False,
#     vmax=abs_p99,
#     threshold=0.0,
#     opacity=0.6,
# )
# view.save_as_html(out_html)
# print(f"Saved beta_clean mean overlay to: {out_html}", flush=True)
