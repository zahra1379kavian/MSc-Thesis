# Combine Run 1 & 2
import argparse
import os
from collections import defaultdict
from os.path import join
from empca.empca.empca import empca
import cvxpy as cp
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
from scipy.spatial import cKDTree
from scipy.stats import t as student_t
import pingouin as pg

# %%
def _asarray_filled(data, dtype=np.float32):
    if isinstance(data, np.ma.MaskedArray):
        data = data.filled(np.nan)
    return np.asarray(data, dtype=dtype)

def synchronize_beta_voxels(beta_run1, beta_run2, mask_run1, mask_run2):
    mask_run1 = mask_run1.ravel()
    mask_run2 = mask_run2.ravel()
    valid_run1 = ~mask_run1
    valid_run2 = ~mask_run2
    shared_valid = valid_run1 & valid_run2

    run1_indices = np.flatnonzero(valid_run1)
    run2_indices = np.flatnonzero(valid_run2)
    shared_mask_run1 = shared_valid[run1_indices]
    shared_mask_run2 = shared_valid[run2_indices]

    synced_run1 = beta_run1[shared_mask_run1]
    synced_run2 = beta_run2[shared_mask_run2]
    removed_run1 = beta_run1.shape[0] - synced_run1.shape[0]
    removed_run2 = beta_run2.shape[0] - synced_run2.shape[0]
    if removed_run1 > 0 or removed_run2 > 0:
        print(f"Removed {removed_run1} run1 voxels and {removed_run2} run2 voxels to enforce a shared voxel set.",
            flush=True)

    shared_flat_indices = np.flatnonzero(shared_valid)

    combined_nan_mask = mask_run1 | mask_run2
    return synced_run1, synced_run2, combined_nan_mask, shared_flat_indices

def combine_active_run_data(coords_run1, coords_run2, bold_run1, bold_run2, volume_shape, invalid_flat_mask=None):
    coords_run1 = tuple(np.asarray(axis, dtype=np.int64) for axis in coords_run1)
    coords_run2 = tuple(np.asarray(axis, dtype=np.int64) for axis in coords_run2)

    flat_run1 = np.ravel_multi_index(coords_run1, volume_shape)
    flat_run2 = np.ravel_multi_index(coords_run2, volume_shape)
    shared_flat = np.intersect1d(flat_run1, flat_run2, assume_unique=False)
    if invalid_flat_mask is not None:
        invalid_flat_mask = invalid_flat_mask.ravel()
        shared_flat = shared_flat[~invalid_flat_mask[shared_flat]]

    shared_coords = np.unravel_index(shared_flat, volume_shape)
    shared_coords = tuple(np.asarray(axis, dtype=np.int64) for axis in shared_coords)

    sorter1 = np.argsort(flat_run1)
    sorter2 = np.argsort(flat_run2)
    sorted_flat_run1 = flat_run1[sorter1]
    sorted_flat_run2 = flat_run2[sorter2]
    idx_run1 = sorter1[np.searchsorted(sorted_flat_run1, shared_flat)]
    idx_run2 = sorter2[np.searchsorted(sorted_flat_run2, shared_flat)]
    combined_bold = np.concatenate((bold_run1[idx_run1], bold_run2[idx_run2]), axis=1)

    return shared_flat, shared_coords, combined_bold

def combine_filtered_betas(beta_run1, beta_run2, nan_mask_flat, flat_indices=None):
    union_nan_mask = np.isnan(beta_run1) | np.isnan(beta_run2)
    if np.any(union_nan_mask):
        beta_run1[union_nan_mask] = np.nan
        beta_run2[union_nan_mask] = np.nan
    beta_combined = np.concatenate((beta_run1, beta_run2), axis=1)
    valid_voxel_mask = ~np.all(np.isnan(beta_combined), axis=1)
    removed_voxels = int(valid_voxel_mask.size - np.count_nonzero(valid_voxel_mask))
    if removed_voxels > 0:
        beta_combined = beta_combined[valid_voxel_mask]
        if flat_indices is not None and nan_mask_flat is not None:
            nan_mask_flat = np.asarray(nan_mask_flat, dtype=bool)
            dropped_flat = np.asarray(flat_indices)[~valid_voxel_mask]
            nan_mask_flat[dropped_flat] = True
    print(f"Combined filtered beta shape: {beta_combined.shape}", flush=True)
    print(f"Removed voxels with all-NaN trials: {removed_voxels}", flush=True)
    return beta_combined, nan_mask_flat

def mask_anatomical_image(anat_img, mask_flat, volume_shape):
    """Mask anatomical volume so it matches the functional voxel mask."""
    anat_data = np.asarray(anat_img.get_fdata(), dtype=np.float32)
    if anat_data.shape != volume_shape:
        raise ValueError(f"Anatomical volume shape {anat_data.shape} does not match functional shape {volume_shape}.")
    mask_flat = np.asarray(mask_flat, dtype=bool).ravel()
    if mask_flat.size != np.prod(volume_shape):
        raise ValueError(f"Mask size {mask_flat.size} does not match volume size {np.prod(volume_shape)}.")
    mask_3d = mask_flat.reshape(volume_shape)
    anat_data = np.where(mask_3d, np.nan, anat_data)
    return anat_data

def compute_distance_weighted_adjacency(coords, affine, radius_mm=3.0, sigma_mm=None):
    """Build sparse distance-weighted adjacency and degree matrices for the provided voxel coords."""
    if sigma_mm is None:
        sigma_mm = radius_mm / 2.0
    coord_array = np.column_stack(coords).astype(np.float32, copy=False)
    coords_mm = nib.affines.apply_affine(affine, coord_array)
    tree = cKDTree(coords_mm)
    dist_matrix = tree.sparse_distance_matrix(tree, max_distance=radius_mm, output_type="coo_matrix")
    rows = dist_matrix.row
    cols = dist_matrix.col
    data = dist_matrix.data
    if data.size:
        off_diag = rows != cols
        rows = rows[off_diag]
        cols = cols[off_diag]
        data = data[off_diag]
    weights = np.exp(-(data ** 2) / (sigma_mm ** 2))
    adjacency = sparse.csr_matrix((weights, (rows, cols)), shape=(coords_mm.shape[0], coords_mm.shape[0]))
    if adjacency.nnz:
        adjacency = adjacency.maximum(adjacency.T)
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    degree_matrix = sparse.diags(degrees)
    return adjacency, degree_matrix

# %% 
ses = 1
sub = "04"
num_trials = 180
trial_len = 9
behave_indice = 1 #1/RT
trials_per_run = num_trials // 2

base_path = "/scratch/st-mmckeown-1/zkavian/fmri_models/MSc-Thesis/"
brain_mask_path = f"{base_path}/sub-pd0{sub}_ses-{ses}_T1w_brain_mask.nii.gz"
csf_mask_path = f"{base_path}/sub-pd0{sub}_ses-{ses}_T1w_brain_pve_0.nii.gz"
gray_mask_path = f"{base_path}/sub-pd0{sub}_ses-{ses}_T1w_brain_pve_1.nii.gz"

# %%
anat_img = nib.load(f"{base_path}/sub-pd0{sub}_ses-{ses}_T1w_brain.nii.gz")

brain_mask = nib.load(brain_mask_path).get_fdata().astype(np.float16)
csf_mask = nib.load(csf_mask_path).get_fdata().astype(np.float16)
gray_mask = nib.load(gray_mask_path).get_fdata().astype(np.float16)

print("1")

glm_dict = np.load(f"{base_path}/TYPED_FITHRF_GLMDENOISE_RR_sub{sub}.npy", allow_pickle=True).item()
beta_glm = _asarray_filled(glm_dict["betasmd"], dtype=np.float32)

nan_mask_flat_run1 = np.load(f"nan_mask_flat_sub{sub}_ses{ses}_run1.npy")
nan_mask_flat_run2 = np.load(f"nan_mask_flat_sub{sub}_ses{ses}_run2.npy")

# beta_volume_filtered_run1 = np.load(f"{base_path}/cleaned_beta_volume_sub{sub}_ses{ses}_run1.npy")
# beta_volume_filtered_run2 = np.load(f"{base_path}/cleaned_beta_volume_sub{sub}_ses{ses}_run2.npy")
beta_filtered_run1  = np.load(f"beta_volume_filter_sub{sub}_ses{ses}_run1.npy") 
beta_filtered_run2  = np.load(f"beta_volume_filter_sub{sub}_ses{ses}_run2.npy") 

print("2")

beta_filtered_run1, beta_filtered_run2, shared_nan_mask_flat, shared_flat_indices = synchronize_beta_voxels(
    beta_filtered_run1, beta_filtered_run2, nan_mask_flat_run1, nan_mask_flat_run2)
nan_mask_shape = shared_nan_mask_flat.shape
nan_mask_flat = shared_nan_mask_flat.ravel()
print(f"shared_nan_mask_flat: {shared_nan_mask_flat.shape}", flush=True)
beta_clean, nan_mask_flat = combine_filtered_betas(
    beta_filtered_run1, beta_filtered_run2, nan_mask_flat, shared_flat_indices)
shared_nan_mask_flat = nan_mask_flat.reshape(nan_mask_shape)

print("3")

bold_data_run1 = np.load(join(base_path, f'fmri_sub{sub}_ses{ses}_run1.npy'))
bold_data_run2 = np.load(join(base_path, f'fmri_sub{sub}_ses{ses}_run2.npy'))
clean_active_bold_run1 = np.load(f"active_bold_sub{sub}_ses{ses}_run1.npy")
clean_active_bold_run2 = np.load(f"active_bold_sub{sub}_ses{ses}_run2.npy")
volume_shape = bold_data_run1.shape[:3]

print("4")
# mask_3d = nan_mask_flat.reshape(mask_volume_shape)
# mask_broadcast = np.broadcast_to(mask_3d[..., None], bold_data_run1.shape)
# bold_data_run1 = np.asarray(bold_data_run1, dtype=np.float32, order="C")
# bold_data_run2 = np.asarray(bold_data_run2, dtype=np.float32, order="C")
# np.copyto(bold_data_run1, np.nan, where=mask_broadcast)
# np.copyto(bold_data_run2, np.nan, where=mask_broadcast)
# bold_clean = np.concatenate((bold_data_run1, bold_data_run2), axis=-1)
# print(f"bold_matrix (masked) shape: {bold_clean.shape}", flush=True)

active_coords_run1 = np.load(f"active_coords_sub{sub}_ses{ses}_run1.npy", allow_pickle=True)
active_coords_run2 = np.load(f"active_coords_sub{sub}_ses{ses}_run2.npy", allow_pickle=True)
# Each `active_coords` array is the tuple of (x, y, z) indices saved during preprocessing for the voxels that stayed active after the statistical and Hampel filtering steps.

active_flat_idx, shared_coords, bold_clean = combine_active_run_data(active_coords_run1, active_coords_run2, 
                                                                               clean_active_bold_run1, clean_active_bold_run2,
                                                                               volume_shape, shared_nan_mask_flat)
print("5")
active_coords = np.asarray(shared_coords, dtype=object)
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
adjacency_matrix, degree_matrix = compute_distance_weighted_adjacency(shared_coords, anat_img.affine,
                                                                      radius_mm=adjacency_radius_mm,
                                                                      sigma_mm=adjacency_sigma_mm)
adj_data = adjacency_matrix.data
adj_range = (float(adj_data.min()), float(adj_data.max())) if adj_data.size else (0.0, 0.0)
degree_values = degree_matrix.data.ravel()
degree_range = (float(degree_values.min()), float(degree_values.max())) if degree_values.size else (0.0, 0.0)
print(f"Adjacency matrix (distance-weighted) shape: {adjacency_matrix.shape}, nnz={adjacency_matrix.nnz}, range: [{adj_range[0]}, {adj_range[1]}]", flush=True)
print(f"Degree matrix shape: {degree_matrix.shape}, range: [{degree_range[0]}, {degree_range[1]}]", flush=True)

# Visualize the first 100 voxels for adjacency, degree, and Laplacian matrices.
voxel_subset = min(100, adjacency_matrix.shape[0])
adj_subset = adjacency_matrix[:voxel_subset, :voxel_subset].toarray()
deg_subset = degree_matrix[:voxel_subset, :voxel_subset].toarray()
laplacian_matrix = degree_matrix - adjacency_matrix
lap_subset = laplacian_matrix[:voxel_subset, :voxel_subset].toarray()

def _plot_matrix(matrix, title, filename):
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="viridis", interpolation="nearest")
    plt.title(title)
    plt.xlabel("Voxel index")
    plt.ylabel("Voxel index")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

_plot_matrix(adj_subset, f"Adjacency (first {voxel_subset} voxels)", "adjacency_first100.png")
_plot_matrix(deg_subset, f"Degree (first {voxel_subset} voxels)", "degree_first100.png")
_plot_matrix(lap_subset, f"Laplacian (first {voxel_subset} voxels)", "laplacian_first100.png")
