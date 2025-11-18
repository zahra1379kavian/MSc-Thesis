# Combine Run 1 & 2
import argparse
import os
from collections import defaultdict
from itertools import product
from os.path import join
from empca.empca.empca import empca
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import plotting
from scipy.io import loadmat
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests

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

glm_dict = np.load(f"{base_path}/TYPED_FITHRF_GLMDENOISE_RR_sub{sub}.npy", allow_pickle=True).item()
beta_glm = _asarray_filled(glm_dict["betasmd"], dtype=np.float32)

nan_mask_flat_run1 = np.load(f"nan_mask_flat_sub{sub}_ses{ses}_run1.npy")
nan_mask_flat_run2 = np.load(f"nan_mask_flat_sub{sub}_ses{ses}_run2.npy")

beta_volume_filtered_run1 = np.load(f"{base_path}/cleaned_beta_volume_sub{sub}_ses{ses}_run1.npy")
beta_volume_filtered_run2 = np.load(f"{base_path}/cleaned_beta_volume_sub{sub}_ses{ses}_run2.npy")
beta_filtered_run1  = np.load(f"beta_volume_filter_sub{sub}_ses{ses}_run1.npy") 
beta_filtered_run2  = np.load(f"beta_volume_filter_sub{sub}_ses{ses}_run2.npy") 

beta_filtered_run1, beta_filtered_run2, shared_nan_mask_flat, shared_flat_indices = synchronize_beta_voxels(
    beta_filtered_run1, beta_filtered_run2, nan_mask_flat_run1, nan_mask_flat_run2)
nan_mask_shape = shared_nan_mask_flat.shape
nan_mask_flat = shared_nan_mask_flat.ravel()
print(f"shared_nan_mask_flat: {shared_nan_mask_flat.shape}", flush=True)
beta_clean, nan_mask_flat = combine_filtered_betas(
    beta_filtered_run1, beta_filtered_run2, nan_mask_flat, shared_flat_indices)
shared_nan_mask_flat = nan_mask_flat.reshape(nan_mask_shape)

bold_data_run1 = np.load(join(base_path, f'fmri_sub{sub}_ses{ses}_run1.npy'))
bold_data_run2 = np.load(join(base_path, f'fmri_sub{sub}_ses{ses}_run2.npy'))
clean_active_bold_run1 = np.load(f"active_bold_sub{sub}_ses{ses}_run1.npy")
clean_active_bold_run2 = np.load(f"active_bold_sub{sub}_ses{ses}_run2.npy")


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
                                                                               bold_data_run1.shape[:3], shared_nan_mask_flat)

active_coords = np.asarray(shared_coords, dtype=object)
print(f"Combined active_flat_idx: {active_flat_idx.shape}", flush=True)
print(f"Combined clean_active_bold: {bold_clean.shape}", flush=True)
print(f"Combined active_coords: {active_coords.shape}", flush=True)

# %%
behav_data = loadmat(f"{base_path}/PSPD0{sub}_OFF_behav_metrics.mat" if ses == 1 else f"{base_path}/PSPD0{sub}_ON_behav_metrics.mat")
behav_metrics = behav_data["behav_metrics"]
behav_block = np.stack(behav_metrics[0], axis=0)
_, _, num_metrics = behav_block.shape
behavior_matrix = behav_block.reshape(-1, num_metrics)
print(f"Behavior matrix shape: {behavior_matrix.shape}")
# %%
def calcu_matrices_func(beta_pca, bold_pca, behave_mat, behave_indice, trial_len=trial_len,num_trials=180, trial_indices=None, run_boundary=None):
    bold_pca_reshape = bold_pca.reshape(bold_pca.shape[0], num_trials, trial_len)
    behavior_selected = behave_mat[:, behave_indice]
    if trial_indices is None:
        trial_indices = np.arange(num_trials, dtype=np.int64)
    else:
        trial_indices = np.asarray(trial_indices, dtype=np.int64)
    if run_boundary is None:
        run_boundary = num_trials // 2

    # print("L_task...", flush=True)
    counts = np.count_nonzero(np.isfinite(beta_pca), axis=-1)
    sums = np.nansum(np.abs(beta_pca), axis=-1, dtype=np.float32)
    mean_beta = np.zeros(beta_pca.shape[0], dtype=np.float32)
    mask = counts > 0
    mean_beta[mask] = (sums[mask] / counts[mask]).astype(np.float32)

    C_task = np.zeros_like(mean_beta, dtype=np.float32)
    valid = np.abs(mean_beta) > 0
    C_task[valid] = (1.0 / mean_beta[valid]).astype(np.float32)
    C_task = np.diag(C_task)

    # print("L_var...", flush=True)
    C_bold = np.zeros((bold_pca_reshape.shape[0], bold_pca_reshape.shape[0]), dtype=np.float32)
    bold_transition_count = 0
    for i in range(num_trials - 1):
        idx_current = trial_indices[i]
        idx_next = trial_indices[i + 1]
        if (idx_current < run_boundary) and (idx_next >= run_boundary):
            continue
        x1 = bold_pca_reshape[:, i, :]
        x2 = bold_pca_reshape[:, i + 1, :]
        C_bold += (x1 - x2) @ (x1 - x2).T
        bold_transition_count += 1
    if bold_transition_count > 0:
        C_bold /= bold_transition_count

    # print("L_var...", flush=True)
    C_beta = np.zeros((beta_pca.shape[0], beta_pca.shape[0]), dtype=np.float32)
    beta_transition_count = 0
    for i in range(num_trials - 1):
        idx_current = trial_indices[i]
        idx_next = trial_indices[i + 1]
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

def standardize_matrix(matrix):
    sym_array = 0.5 * (matrix + matrix.T)
    trace_value = np.trace(sym_array)
    if not np.isfinite(trace_value) or trace_value == 0:
        trace_value = 1.0
    normalized = sym_array / trace_value * sym_array.shape[0]
    # print(f"range after: {np.min(normalized)}, {np.max(normalized)}")
    return normalized.astype(matrix.dtype, copy=False)

def _reshape_trials(bold_matrix, num_trials, trial_length, error_label="BOLD"):
    num_voxels, num_timepoints = bold_matrix.shape
    bold_by_trial = np.full((num_voxels, num_trials, trial_length), np.nan, dtype=np.float32)

    start_idx = 0
    for trial_idx in range(num_trials):
        end_idx = start_idx + trial_length
        if end_idx > num_timepoints:
            raise ValueError(f"{error_label} data does not contain enough timepoints for all trials.")
        bold_by_trial[:, trial_idx, :] = bold_matrix[:, start_idx:end_idx]
        start_idx += trial_length
        if start_idx in (270, 560):
            start_idx += 20
    return bold_by_trial

# def _extract_active_bold_trials(bold_timeseries, coords, num_trials, trial_length):
#     coord_arrays = tuple(np.asarray(axis, dtype=int) for axis in coords)
#     bold_active = bold_timeseries[coord_arrays[0], coord_arrays[1], coord_arrays[2], :]
#     return _reshape_trials(bold_active, num_trials, trial_length, error_label="Active BOLD")

def apply_empca(bold_clean):
    print("PCA...", flush=True)
    def prepare_for_empca(data):
        W = np.isfinite(data).astype(float)
        Y = np.where(np.isfinite(data), data, 0.0)

        row_weight = W.sum(axis=0, keepdims=True)
        print(row_weight.shape)
        mean = np.divide((Y * W).sum(axis=0, keepdims=True), row_weight, out=np.zeros_like(row_weight), where=row_weight > 0)
        centered = Y - mean

        var = np.divide((W * centered**2).sum(axis=0, keepdims=True), row_weight, out=np.zeros_like(row_weight), where=row_weight > 0)
        scale = np.sqrt(var)
        print(mean.shape, scale.shape)

        z = np.divide(centered, np.maximum(scale, 1e-6), out=np.zeros_like(centered), where=row_weight > 0)
        return z, W
    X_reshap = bold_clean.reshape(bold_clean.shape[0], -1)
    print(f"bold_by_trial: {X_reshap.shape}")
    print(f"epca func, X_reshap: {X_reshap.shape}")
    Yc, W = prepare_for_empca(X_reshap.T)
    Yc = Yc.T
    W = W.T
    print(Yc.shape, W.shape)

    print("begin empca...", flush=True)
    m = empca(Yc.astype(np.float32, copy=False), W.astype(np.float32, copy=False), nvec=700, niter=15)
    np.save(f'empca_model_sub{sub}_ses{ses}.npy', m)
    return m

def load_or_fit_empca_model(bold_clean):
    model_path = f"empca_model_sub{sub}_ses{ses}.npy"
    if os.path.exists(model_path):
            print(f"Loading existing EMPCA model from {model_path}", flush=True)
            return np.load(model_path, allow_pickle=True).item()
    return apply_empca(bold_clean)

def build_projection_dataset(bold_clean, beta_clean, behavioral_matrix, nan_mask_flat, active_coords, active_flat_indices, trial_length, num_trials):
    pca_model = load_or_fit_empca_model(bold_clean)
    bold_pca_components = pca_model.eigvec
    expected_points = num_trials * trial_length
    if bold_pca_components.shape[1] != expected_points:
        raise ValueError(f"PCA components ({bold_pca_components.shape[1]}) do not match expected trial length ({expected_points}).")
    bold_pca_trials = bold_pca_components.reshape(bold_pca_components.shape[0], num_trials, trial_length)
    coeff_pinv = np.linalg.pinv(pca_model.coeff)
    beta_pca_full = coeff_pinv @ np.nan_to_num(beta_clean)

    return {"pca_model": pca_model, "bold_pca_trials": bold_pca_trials, "coeff_pinv": coeff_pinv, "beta_pca_full": beta_pca_full, 
            "bold_clean": bold_clean, "beta_clean": beta_clean, "behavior_matrix": behavioral_matrix, "nan_mask_flat": nan_mask_flat, 
            "active_coords": active_coords, "active_flat_indices": active_flat_indices}

def prepare_data_func(projection_data, trial_indices, trial_length, run_boundary):
    trial_indices = np.asarray(trial_indices, dtype=np.int64)
    effective_num_trials = trial_indices.size
    behavior_subset = projection_data["behavior_matrix"][trial_indices]
    beta_pca_full = projection_data["beta_pca_full"]
    beta_pca = beta_pca_full[:, trial_indices]

    bold_pca_trials = projection_data["bold_pca_trials"][:, trial_indices, :]
    bold_pca_components = bold_pca_trials.reshape(bold_pca_trials.shape[0], effective_num_trials * trial_length)
    print(f"bold_pca_components (subset): {bold_pca_components.shape}")
    print(f"beta_pca (subset): {beta_pca.shape}")

    (C_task, C_bold, C_beta, behavior_vector) = calcu_matrices_func(beta_pca, bold_pca_components, behavior_subset, behave_indice,
                                                                    trial_len=trial_length, num_trials=effective_num_trials,
                                                                    trial_indices=trial_indices, run_boundary=run_boundary)
    C_task = standardize_matrix(C_task)
    C_bold = standardize_matrix(C_bold)
    C_beta = standardize_matrix(C_beta)
    print(f"shape: {C_task.shape}, {C_bold.shape}, {C_beta.shape}", flush=True)
    print(f"Range: [{np.min(C_task)}, {np.max(C_task)}], [{np.min(C_bold)}, {np.max(C_bold)}],[{np.min(C_beta)}, {np.max(C_beta)}]", flush=True)

    trial_mask = np.isfinite(behavior_vector)
    behavior_observed = behavior_vector[trial_mask]
    behavior_mean = np.mean(behavior_observed)
    behavior_centered = behavior_observed - behavior_mean
    behavior_norm = np.linalg.norm(behavior_centered)
    print(f"behavior_norm: {behavior_norm}")
    if not np.isfinite(behavior_norm) or behavior_norm <= 0:
        print("behavior_norm is zero or non-finite; using 1.0 to avoid division errors.", flush=True)
        behavior_norm = 1.0
    normalized_behaviors = behavior_centered / behavior_norm

    beta_observed = beta_pca[:, trial_mask]
    beta_centered = beta_observed - np.mean(beta_observed, axis=1, keepdims=True)

    beta_subset = projection_data["beta_clean"][:, trial_indices]
    bold_subset = projection_data["bold_clean"][:, trial_indices, :]

    return {"nan_mask_flat": projection_data["nan_mask_flat"],
        "active_coords": tuple(np.array(coord) for coord in projection_data["active_coords"]),
        "active_flat_indices": projection_data["active_flat_indices"],
        "coeff_pinv": projection_data["coeff_pinv"], "beta_centered": beta_centered,
        "behavior_centered": behavior_centered, "normalized_behaviors": normalized_behaviors,
        "behavior_observed": behavior_observed, "beta_observed": beta_observed,
        "beta_clean": beta_subset, "C_task": C_task, "C_bold": C_bold, "C_beta": C_beta,
        "active_bold": bold_subset, "trial_indices": trial_indices, "num_trials": effective_num_trials}

def calcu_penalty_terms(run_data, alpha_task, alpha_bold, alpha_beta):
    beta_centered = run_data["beta_centered"]
    n_components = beta_centered.shape[0]

    task_penalty = alpha_task * run_data["C_task"]
    bold_penalty = alpha_bold * run_data["C_bold"]
    beta_penalty = alpha_beta * run_data["C_beta"]
    total_penalty = task_penalty + bold_penalty + beta_penalty
    total_penalty = 0.5 * (total_penalty + total_penalty.T)
    total_penalty = total_penalty + 1e-6 * np.eye(n_components)

    return {"task": task_penalty, "bold": bold_penalty, "beta": beta_penalty}, total_penalty

def solve_soc_problem(run_data, alpha_task, alpha_bold, alpha_beta, solver_name, phro): # I should change this function
    beta_centered = run_data["beta_centered"]
    normalized_behaviors = run_data["normalized_behaviors"]
    n_components = beta_centered.shape[0]

    penalty_matrices, total_penalty = calcu_penalty_terms(run_data, alpha_task, alpha_bold, alpha_beta)

    weights_positive = cp.Variable(n_components)
    weights_negative = cp.Variable(n_components)
    projections_positive = beta_centered.T @ weights_positive
    projections_negative = beta_centered.T @ weights_negative

    constraints_positive = [weights_positive >= 0, cp.sum(weights_positive) == 1,
        cp.SOC((1.0 / phro) * (normalized_behaviors @ projections_positive), projections_positive)]
    problem_positive = cp.Problem(cp.Minimize(cp.quad_form(weights_positive, cp.psd_wrap(total_penalty))), constraints_positive)


    constraints_negative = [weights_negative >= 0, cp.sum(weights_negative) == 1,
        cp.SOC((1.0 / phro) * (-normalized_behaviors @ projections_negative), projections_negative)]
    problem_negative = cp.Problem(cp.Minimize(cp.quad_form(weights_negative, cp.psd_wrap(total_penalty))), constraints_negative)

    objective_positive = np.inf
    objective_negative = np.inf
    problem_positive.solve(solver=solver_name, warm_start=True)
    problem_negative.solve(solver=solver_name, warm_start=True)

    lam_nonneg   = constraints_positive[0].dual_value   # vector (same size as w) for w >= 0
    lam_sum1     = constraints_positive[1].dual_value   # scalar for sum(w) == 1
    lam_soc_tau, lam_soc_vec = constraints_positive[2].dual_value
    # print(f"lambda Positive: {np.sum(lam_nonneg)}, {lam_sum1}, {lam_soc_tau}, {np.sum(lam_soc_vec)}")

    lam_nonneg   = constraints_negative[0].dual_value   # vector (same size as w) for w >= 0
    lam_sum1     = constraints_negative[1].dual_value   # scalar for sum(w) == 1
    lam_soc_tau, lam_soc_vec = constraints_negative[2].dual_value
    # print(f"lambda Negative: {np.sum(lam_nonneg)}, {lam_sum1}, {lam_soc_tau}, {np.sum(lam_soc_vec)}")

    positive_valid = weights_positive.value is not None and problem_positive.status in ("optimal", "optimal_inaccurate")
    negative_valid = weights_negative.value is not None and problem_negative.status in ("optimal", "optimal_inaccurate")

    if positive_valid: objective_positive = problem_positive.value
    if negative_valid:objective_negative = problem_negative.value

    solution_branch = None
    solution_weights = None

    if positive_valid and (not negative_valid or objective_positive <= objective_negative):
        solution_branch = "positive"
        solution_weights = np.array(weights_positive.value, dtype=np.float64).ravel()
    elif negative_valid:
        solution_branch = "negative"
        solution_weights = np.array(weights_negative.value, dtype=np.float64).ravel()

    if solution_weights is None or solution_weights.size != n_components:
        raise RuntimeError(f"SOC solver failed to find a feasible solution "
                           f"(positive status={problem_positive.status}, negative status={problem_negative.status}).")

    contributions = {label: float(solution_weights.T @ matrix @ solution_weights) for label, matrix in penalty_matrices.items()}

    y = beta_centered.T @ solution_weights
    total_loss = float(solution_weights.T @ total_penalty @ solution_weights)

    return {"weights": solution_weights, "branch": solution_branch, "total_loss": total_loss, "Y": y,
            "objective_positive": objective_positive, "objective_negative": objective_negative, "penalty_contributions": contributions}

def evaluate_projection(data, weights):
    beta_centered = data["beta_centered"]
    behavior_centered = data["behavior_centered"]
    normalized_behaviors = data["normalized_behaviors"]

    projection = beta_centered.T @ weights
    finite_mask = np.isfinite(projection) & np.isfinite(behavior_centered)
    projection = projection[finite_mask]
    behavior_centered = behavior_centered[finite_mask]
    normalized_behaviors = normalized_behaviors[finite_mask]

    metrics = {"pearson": np.nan, "r2": np.nan, "mse": np.nan, "soc_slack": np.nan}
    metrics["pearson"] = _safe_pearsonr(projection, behavior_centered)
    residuals = behavior_centered - projection
    metrics["mse"] = np.mean(residuals**2)
    sst = np.sum(behavior_centered**2)
    metrics["r2"] = 1 - np.sum(residuals**2) / sst

    return metrics

def summarize_bootstrap(values, reference=0.0, direction="less"):
    values_array = _asarray_filled(values, dtype=np.float64)
    finite_mask = np.isfinite(values_array)
    values_array = values_array[finite_mask]

    summary = {"mean": np.nan, "std": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "p_value": np.nan, "num_samples": int(values_array.size)}
    summary["mean"] = np.mean(values_array)
    summary["std"] = np.std(values_array, ddof=1) if values_array.size > 1 else 0.0
    summary["ci_lower"], summary["ci_upper"] = (float(x) for x in np.percentile(values_array, [2.5, 97.5]))

    if direction == "greater":
        summary["p_value"] = (np.count_nonzero(values_array >= reference) + 1) / (values_array.size + 1)
    else:
        summary["p_value"] = (np.count_nonzero(values_array <= reference) + 1) / (values_array.size + 1)
    return summary

def perform_weight_shuffle_tests(train_data, test_data, weights, task_alpha, bold_alpha, beta_alpha, rho_value, num_samples):
    weights = weights.ravel()
    rng = np.random.default_rng()
    penalty_matrices, total_penalty = calcu_penalty_terms(train_data, task_alpha, bold_alpha, beta_alpha)
    baseline_penalties = {label: float(weights.T @ matrix @ weights) for label, matrix in penalty_matrices.items()}
    baseline_loss = float(weights.T @ total_penalty @ weights)
    baseline_train_corr = evaluate_projection(train_data, weights)["pearson"]
    baseline_test_corr = evaluate_projection(test_data, weights)["pearson"]

    penalty_samples = {label: [] for label in penalty_matrices}
    total_loss_samples = []
    test_corr_samples = []
    train_corr_samples = []

    weights_std = float(np.std(weights))
    if not np.isfinite(weights_std) or weights_std < 0:
        weights_std = 0.0
    noise_std = weights_std * 0.1
    if not np.isfinite(noise_std) or noise_std <= 0:
        noise_std = 1e-8

    print(f"Noise Dist, mean: {np.mean(weights)}, std: {noise_std}")
    for iteration in range(num_samples):
        noise = rng.normal(loc=np.mean(weights), scale=noise_std, size=weights.shape)
        noisy_weights = weights + noise
        noisy_weights = np.abs(noisy_weights)
        sum_noisy = np.sum(noisy_weights)
        if not np.isfinite(sum_noisy) or sum_noisy <= 0:
            noisy_weights = np.full_like(noisy_weights, 1.0 / noisy_weights.size)
        else:
            noisy_weights /= sum_noisy

        penalty_values = {}
        for label, matrix in penalty_matrices.items():
            penalty_value = float(noisy_weights.T @ matrix @ noisy_weights)
            penalty_samples[label].append(penalty_value)
            penalty_values[label] = penalty_value

        total_loss = float(noisy_weights.T @ total_penalty @ noisy_weights)
        total_loss_samples.append(total_loss)

        train_metrics = evaluate_projection(train_data, noisy_weights)
        test_metrics = evaluate_projection(test_data, noisy_weights)

        train_corr_samples.append(train_metrics["pearson"])
        test_corr_samples.append(test_metrics["pearson"])

    penalty_arrays = {label: np.asarray(values, dtype=np.float64) for label, values in penalty_samples.items()}
    total_loss_array = np.asarray(total_loss_samples, dtype=np.float64)
    train_corr_array = np.asarray(train_corr_samples, dtype=np.float64)
    test_corr_array = np.asarray(test_corr_samples, dtype=np.float64)

    penalty_summary = {label: summarize_bootstrap(values, reference=baseline_penalties.get(label, 0.0), direction="less") 
                       for label, values in penalty_arrays.items()}
    total_loss_summary = summarize_bootstrap(total_loss_array, reference=baseline_loss, direction = "less")
    train_direction = "greater" if baseline_train_corr >= 0 else "less"
    test_direction = "greater" if baseline_test_corr >= 0 else "less"
    correlation_summary = {"train": summarize_bootstrap(train_corr_array, reference=baseline_train_corr, direction=train_direction),
                           "test": summarize_bootstrap(test_corr_array, reference=baseline_test_corr, direction=test_direction)}

    # num_successful = int(np.count_nonzero(np.isfinite(test_corr_array)))
    # num_failed = int(num_samples - num_successful)

    return {"penalties": penalty_summary, "total_loss": total_loss_summary, "correlation": correlation_summary, 
            "penalty_samples": penalty_arrays, "total_loss_samples": total_loss_array, "correlation_samples": {"train": train_corr_array, "test": test_corr_array}}

def plot_bootstrap_distribution(samples, actual_value, p_value, title, xlabel, output_path):
    finite_samples = samples[np.isfinite(samples)]

    num_bins = max(10, min(50, int(np.sqrt(finite_samples.size))))
    sample_min = float(np.min(finite_samples))
    sample_max = float(np.max(finite_samples))

    lower_edge = float(min(sample_min, actual_value)) if np.isfinite(actual_value) else sample_min
    upper_edge = float(max(sample_max, actual_value)) if np.isfinite(actual_value) else sample_max
    span_margin = 0.05 * max(sample_max - sample_min, 1e-9)
    range_lower = lower_edge - span_margin
    range_upper = upper_edge + span_margin

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(finite_samples, bins=num_bins, color="#4c72b0", alpha=0.75, edgecolor="black", range=(range_lower, range_upper))
    
    annotation_parts = []
    ax.axvline(actual_value, color="#c44e52", linestyle="--", linewidth=2.0, label="Actual")
    annotation_parts.append(f"actual={actual_value:.4g}")
    annotation_parts.append(f"p={p_value:.4g}")
    annotation_text = "\n".join(annotation_parts)
    ax.text(0.98, 0.95, annotation_text, transform=ax.transAxes, ha="right", va="top",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.set_xlim(range_lower, range_upper)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def _compute_weighted_active_bold_projection(voxel_weights, data):
    active_bold = data.get("active_bold")
    active_flat_indices = data.get("active_flat_indices")
    nan_mask_flat = data.get("nan_mask_flat")

    voxel_weights = voxel_weights.ravel()
    active_flat_indices = active_flat_indices.ravel()
    valid_flat_indices = np.flatnonzero(~np.asarray(nan_mask_flat, dtype=bool).ravel())
    positions = np.searchsorted(valid_flat_indices, active_flat_indices)

    active_voxel_weights = voxel_weights[positions]
    active_bold = _asarray_filled(active_bold, dtype=np.float64)
    bold_matrix = active_bold.reshape(active_bold.shape[0], -1).copy()
    with np.errstate(invalid="ignore"):
        voxel_means = np.nanmean(bold_matrix, axis=1, keepdims=True)

    voxel_means = np.where(np.isfinite(voxel_means), voxel_means, 0.0)
    bold_matrix -= voxel_means
    timepoints = bold_matrix.shape[1]
    projection = np.full(timepoints, np.nan, dtype=np.float64)

    finite_mask = np.isfinite(bold_matrix)
    for t_idx in range(timepoints):
        voxel_mask = finite_mask[:, t_idx]
        if not np.any(voxel_mask):
            continue
        projection[t_idx] = np.dot(active_voxel_weights[voxel_mask], bold_matrix[voxel_mask, t_idx])
    return projection, bold_matrix

def _resolve_trial_metric(metric):
    if metric is None:
        return np.nanmean
    if callable(metric):
        return metric
    metric_key = str(metric).strip().lower()
    if metric_key in ("", "mean", "nanmean"):
        return np.nanmean
    if metric_key in ("median", "nanmedian"):
        return np.nanmedian

def _describe_trial_metric(metric):
    if metric is None:
        return "mean"
    if isinstance(metric, str):
        metric_key = metric.strip().lower()
        return metric_key or "mean"
    return getattr(metric, "__name__", "custom_metric")

def _safe_pearsonr(x, y):
    x = _asarray_filled(x, dtype=np.float64).ravel()
    y = _asarray_filled(y, dtype=np.float64).ravel()
    if x.size != y.size or x.size < 2:
        return np.nan
    finite_mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite_mask) < 2:
        return np.nan
    x = x[finite_mask]
    y = y[finite_mask]
    x -= x.mean()
    y -= y.mean()
    denom = np.sqrt(np.dot(x, x) * np.dot(y, y))
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    return float(np.dot(x, y) / denom)

def _aggregate_trials(active_bold, reducer):
    # downsample bold data by reducer metrice
    bold_trials = _asarray_filled(active_bold, dtype=np.float64)
    num_voxels, num_trials, trial_length = bold_trials.shape
    reshaped = bold_trials.reshape(-1, trial_length)
    finite_counts = np.count_nonzero(np.isfinite(reshaped), axis=1)
    valid_mask = finite_counts > 0
    reduced_flat = np.full(reshaped.shape[0], np.nan, dtype=np.float64)

    if np.any(valid_mask):
        valid_values = reshaped[valid_mask]
        reduced_values = reducer(valid_values, axis=-1)
        reduced_flat[valid_mask] = np.asarray(reduced_values, dtype=np.float64).ravel()
    return reduced_flat.reshape(num_voxels, num_trials)

def compute_component_active_beta_correlation(voxel_weights, data, aggregation="median"):
    # corr(weight * beta, bold)
    active_beta = data.get("beta_clean")
    active_bold = data.get("active_bold")

    active_flat_indices = data.get("active_flat_indices")
    nan_mask_flat = data.get("nan_mask_flat")
    active_flat_indices = np.asarray(active_flat_indices).ravel()
    voxel_weights = np.asarray(voxel_weights, dtype=np.float64).ravel()
    valid_flat_indices = np.flatnonzero(~np.asarray(nan_mask_flat, dtype=bool).ravel())
    positions = np.searchsorted(valid_flat_indices, active_flat_indices)

    active_voxel_weights = voxel_weights[positions]
    beta_matrix = _asarray_filled(active_beta, dtype=np.float64)

    reducer = _resolve_trial_metric(aggregation)
    bold_trial_metric = _aggregate_trials(active_bold, reducer)

    num_trials = beta_matrix.shape[1]
    projection = np.full(num_trials, np.nan, dtype=np.float64)
    finite_mask = np.isfinite(beta_matrix)

    for trial_idx in range(num_trials):
        voxel_mask = finite_mask[:, trial_idx]
        if not np.any(voxel_mask):
            continue
        projection[trial_idx] = np.dot(active_voxel_weights[voxel_mask], beta_matrix[voxel_mask, trial_idx])

    correlations = np.full(beta_matrix.shape[0], np.nan, dtype=np.float32)
    projection_finite = np.isfinite(projection)
    for voxel_idx, voxel_series in enumerate(bold_trial_metric):
        voxel_finite = np.isfinite(voxel_series)
        joint_mask = projection_finite & voxel_finite
        if np.count_nonzero(joint_mask) < 2:
            continue
        correlations[voxel_idx] = _safe_pearsonr(projection[joint_mask], voxel_series[joint_mask])
    return projection, correlations

def compute_component_active_bold_correlation(voxel_weights, data):
    # corr(weight * bold, bold)
    projection, bold_matrix = _compute_weighted_active_bold_projection(voxel_weights, data)
    correlations = np.full(bold_matrix.shape[0], np.nan, dtype=np.float32)
    projection_finite = np.isfinite(projection)
    for voxel_idx, voxel_series in enumerate(bold_matrix):
        voxel_finite = np.isfinite(voxel_series)
        joint_mask = projection_finite & voxel_finite
        if np.count_nonzero(joint_mask) < 2:
            continue
        correlations[voxel_idx] = _safe_pearsonr(projection[joint_mask], voxel_series[joint_mask])
    return projection, correlations

def save_active_bold_correlation_map(correlations, active_coords, volume_shape, anat_img, file_prefix, 
                                     result_prefix="active_bold_corr"):
    coord_arrays = tuple(np.asarray(axis, dtype=int) for axis in active_coords)
    display_values = np.abs(correlations.ravel())
    # corr_path = f"{result_prefix}_{file_prefix}.npy"
    # np.save(corr_path, correlations.astype(np.float32))

    volume = np.full(volume_shape, np.nan, dtype=np.float32)
    colorbar_max = None
    if coord_arrays and coord_arrays[0].size:
        volume[coord_arrays] = display_values
        finite_values = display_values[np.isfinite(display_values)]
        colorbar_max = float(np.percentile(finite_values, 95))

    corr_img = nib.Nifti1Image(volume, anat_img.affine, anat_img.header)
    volume_path = f"{result_prefix}_{file_prefix}.nii.gz"
    nib.save(corr_img, volume_path)

    if np.any(np.isfinite(display_values)):
        display_volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
        display_volume = np.clip(display_volume, 0.0, colorbar_max)
        display_img = nib.Nifti1Image(display_volume, anat_img.affine, anat_img.header)
        display = plotting.view_img(display_img, bg_img=anat_img, colorbar=True, symmetric_cmap= False, cmap='jet')
        display.save_as_html(f"{result_prefix}_{file_prefix}.html")
    return

def plot_projection_bold(y_trials, task_alpha, bold_alpha, beta_alpha, rho_value, output_path, max_trials=10, series_label=None):
    num_trials_total, trial_length = y_trials.shape
    time_axis = np.arange(trial_length)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    trial_windows = [(10, 20), (40, 50)]
    window_titles = ["Trials 10-20", "Trials 40-50"]

    def _get_light_colors(count):
        cmap = plt.cm.get_cmap("tab20", count)
        raw_colors = cmap(np.linspace(0, 1, count))
        pastel = raw_colors * 0.55 + 0.45
        return np.clip(pastel, 0.0, 1.0)

    for ax, (start, end), window_title in zip(axes, trial_windows, window_titles):
        start_idx = min(start, num_trials_total)
        end_idx = min(end, num_trials_total)
        window_trials = y_trials[start_idx:end_idx]
        trials_to_plot = window_trials[:max_trials]
        colors = _get_light_colors(trials_to_plot.shape[0])
        for trial_offset, (trial_values, color) in enumerate(zip(trials_to_plot, colors)):
            label = f"Trial {start_idx + trial_offset + 1}"
            ax.plot(time_axis, trial_values, linewidth=1.0, color=color, alpha=0.9, label=label)

        ax.set_ylabel("BOLD projection")
        ax.set_title(window_title)
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend(loc="upper right", fontsize=8, ncol=2)

    axes[-1].set_xlabel("Time point")
    label_suffix = f" [{series_label}]" if series_label else ""
    fig.suptitle(f"y | alpha_task={task_alpha:g}, alpha_bold={bold_alpha:g}, alpha_beta={beta_alpha:g}, rho={rho_value:g}{label_suffix}",fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path

def plot_projection_beta_sweep(rho_projection_series, task_alpha, bold_alpha, beta_alpha, output_path, series_label=None):
    if not rho_projection_series:
        return None
    sorted_series = sorted(rho_projection_series, key=lambda item: item[0])
    max_trials = max(np.asarray(series, dtype=np.float64).ravel().size for _, series in sorted_series)
    fig_height = 5.0 if max_trials <= 120 else 6.5
    time_axis = np.arange(max_trials)
    cmap = plt.cm.get_cmap("tab10", len(sorted_series))

    fig, ax = plt.subplots(figsize=(10, fig_height))
    for idx, (rho_value, projection) in enumerate(sorted_series):
        y_trials = np.asarray(projection, dtype=np.float64).ravel()
        trial_axis = time_axis[: y_trials.size]
        ax.plot(trial_axis, y_trials, linewidth=1.2, alpha=0.9, label=f"rho={rho_value:g}", color=cmap(idx))

    ax.set_xlabel("Trial index")
    ax.set_ylabel("Voxel-space projection")
    label_suffix = f" [{series_label}]" if series_label else ""
    rho_labels = ", ".join(f"{rho:g}" for rho, _ in sorted_series)
    ax.set_title(f"y_beta across rhos ({rho_labels}) | alpha_task={task_alpha:g}, alpha_bold={bold_alpha:g}, alpha_beta={beta_alpha:g}{label_suffix}")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path

def _reshape_projection(y_values, trial_length):
    y_values = np.asarray(y_values, dtype=np.float64).ravel()
    num_trials = y_values.size // trial_length
    return y_values, y_values.reshape(num_trials, trial_length)

def save_projection_outputs(pca_weights, bold_pca_components, trial_length, file_prefix,
                            task_alpha, bold_alpha, beta_alpha, rho_value, voxel_weights, beta_clean,
                            data=None, bold_projection=None):

    # component_weights = pca_weights.ravel()
    # y_projection_pc = component_weights @ bold_pca_components
    # np.save(f"y_projection_{file_prefix}.npy", y_projection_pc)
    # _, y_trials = _reshape_projection(y_projection_pc, trial_length)
    # plot_path = f"y_projection_trials_{file_prefix}.png"
    # plot_projection_bold(y_trials, task_alpha, bold_alpha, beta_alpha, rho_value, plot_path, series_label="PC space")

    bold_projection_signal, _ = _compute_weighted_active_bold_projection(voxel_weights, data)
    # np.save(f"y_projection_{file_prefix}.npy", bold_projection_signal)
    _, y_trials = _reshape_projection(bold_projection_signal, trial_length)
    plot_path = f"y_projection_trials_{file_prefix}.png"
    plot_projection_bold(y_trials, task_alpha, bold_alpha, beta_alpha, rho_value, plot_path, series_label="Active BOLD space")

    voxel_weights = voxel_weights.ravel()
    beta_matrix = np.nan_to_num(beta_clean, nan=0.0, posinf=0.0, neginf=0.0)
    y_projection_voxel_trials = voxel_weights @ beta_matrix
    y_projection_voxel = y_projection_voxel_trials.ravel()
    # np.save(f"y_projection_voxel_{file_prefix}.npy", y_projection_voxel)
    return y_projection_voxel

# # %%
ridge_penalty = 1e-6
solver_name = "MOSEK"

# task_penalty_sweep = [0.0, 0.5, 100.0]
# bold_penalty_sweep = [0.0, 0.25, 100.0]
# beta_penalty_sweep = [0, 50.0, 500.0]
task_penalty_sweep = [0.5]
bold_penalty_sweep = [0.25]
beta_penalty_sweep = [50]
alpha_sweep = [{"task_penalty": task_alpha, "bold_penalty": bold_alpha, "beta_penalty": beta_alpha}
    for task_alpha, bold_alpha, beta_alpha in product(task_penalty_sweep, bold_penalty_sweep, beta_penalty_sweep)]
# rho_sweep = [0.2, 0.6, 0.8]
rho_sweep = [0.6]
bootstrap_iterations = 1000
trial_downsample_metric = "median"
metric_label = _describe_trial_metric(trial_downsample_metric).replace(" ", "_")

projection_data = build_projection_dataset(bold_clean, beta_clean, behavior_matrix, nan_mask_flat, active_coords, active_flat_idx, trial_len, num_trials)

def build_custom_kfold_splits(total_trials, num_folds=5, trials_per_run=None):
    if trials_per_run is None:
        trials_per_run = total_trials // 2
    base_fold = trials_per_run // num_folds
    remainder = trials_per_run % num_folds
    fold_sizes = np.full(num_folds, base_fold, dtype=int)
    if remainder > 0:
        fold_sizes[:remainder] += 1

    folds = []
    run_start = 0
    all_trials = np.arange(total_trials, dtype=np.int64)
    for fold_idx, block_size in enumerate(fold_sizes, start=1):
        run1_start = run_start
        run1_end = run1_start + block_size
        run2_start = trials_per_run + run1_start
        run2_end = run2_start + block_size

        test_run1 = np.arange(run1_start, run1_end, dtype=np.int64)
        test_run2 = np.arange(run2_start, run2_end, dtype=np.int64)
        test_indices = np.concatenate((test_run1, test_run2))

        train_mask = np.ones(total_trials, dtype=bool)
        train_mask[test_indices] = False
        train_indices = all_trials[train_mask]

        folds.append({"fold_id": fold_idx, "train_indices": train_indices, "test_indices": test_indices, "test_run1": test_run1, "test_run2": test_run2})
        run_start = run1_end
    return folds

def run_cross_run_experiment(alpha_settings, rho_values, fold_splits, projection_data, bootstrap_samples=0):
    total_folds = len(fold_splits)
    bold_pca_components_full = projection_data["pca_model"].eigvec
    aggregate_metrics = defaultdict(lambda: {"train_corr": [], "train_r2": [], "test_corr": [], "test_r2": []})
    fold_output_tracker = defaultdict(lambda: {
        "voxel_weights": [],
        "voxel_weights_shape": None,
        "bold_corr": [],
        "bold_corr_shape": None,
        "beta_corr": [],
        "beta_corr_shape": None,
        "projection": [],
        "projection_shape": None,
    })

    def _append_if_finite(target_list, value):
        if np.isfinite(value):
            target_list.append(float(value))

    def _append_fold_array(container, field, values, combo_label):
        arr = np.asarray(values, dtype=np.float64).ravel()
        shape_key = f"{field}_shape"
        existing_shape = container.get(shape_key)
        if existing_shape is None:
            container[shape_key] = arr.shape
        elif existing_shape != arr.shape:
            print(f"Skipping accumulation of {field} for {combo_label} due to shape mismatch "
                  f"{arr.shape} vs {existing_shape}", flush=True)
            return
        container[field].append(arr)

    def _stack_nanmean(values):
        if not values:
            return None
        stacked = np.stack(values, axis=0)
        with np.errstate(invalid="ignore"):
            return np.nanmean(stacked, axis=0)

    for fold_idx, split in enumerate(fold_splits, start=1):
        print(f"\n===== Fold {fold_idx}/{total_folds} =====", flush=True)
        test_indices = split["test_indices"]
        train_indices = split["train_indices"]
        run1_desc = (int(split["test_run1"][0] + 1), int(split["test_run1"][-1] + 1))
        run2_desc = (int(split["test_run2"][0] + 1), int(split["test_run2"][-1] + 1))
        print(f"Test trials (1-based): run1 {run1_desc[0]}-{run1_desc[1]}, run2 {run2_desc[0]}-{run2_desc[1]}", flush=True)

        train_data = prepare_data_func(projection_data, train_indices, trial_len, trials_per_run)
        test_data = prepare_data_func(projection_data, test_indices, trial_len, trials_per_run)
        bold_pca_components = bold_pca_components_full

        for alpha_setting in alpha_settings:
            task_alpha = alpha_setting["task_penalty"]
            bold_alpha = alpha_setting["bold_penalty"]
            beta_alpha = alpha_setting["beta_penalty"]
            rho_projection_series = []
            alpha_prefix = f"fold{fold_idx}_sub{sub}_ses{ses}_task{task_alpha:g}_bold{bold_alpha:g}_beta{beta_alpha:g}"

            for rho_value in rho_values:
                metrics_key = (task_alpha, bold_alpha, beta_alpha, rho_value)
                combo_label = (f"task={task_alpha:g}, bold={bold_alpha:g}, "
                               f"beta={beta_alpha:g}, rho={rho_value:g}")
                sweep_suffix = f"{alpha_prefix}_rho{rho_value:g}"
                print(f"Fold {fold_idx}: optimization with task={task_alpha}, bold={bold_alpha}, beta={beta_alpha}, rho={rho_value}", flush=True)
                solution = solve_soc_problem(train_data, task_alpha, bold_alpha, beta_alpha, solver_name, rho_value)

                component_weights = np.abs(solution["weights"])
                coeff_pinv = np.asarray(train_data["coeff_pinv"])
                voxel_weights = coeff_pinv.T @ component_weights
                file_prefix = sweep_suffix
                save_active_bold_correlation_map(voxel_weights, train_data.get("active_coords"), anat_img.shape[:3], anat_img,
                                                 file_prefix, result_prefix="voxel_weights")

                projection_signal, voxel_correlations = compute_component_active_bold_correlation(voxel_weights, train_data)
                save_active_bold_correlation_map(voxel_correlations, train_data.get("active_coords"), anat_img.shape[:3], anat_img, file_prefix)

                beta_projection_signal, beta_voxel_correlations = compute_component_active_beta_correlation(
                    voxel_weights, train_data, aggregation=trial_downsample_metric)
                save_active_bold_correlation_map(beta_voxel_correlations, train_data.get("active_coords"), anat_img.shape[:3],
                                                 anat_img, file_prefix, result_prefix=f"active_beta_corr_{metric_label}")

                y_projection_voxel = save_projection_outputs(component_weights, bold_pca_components, trial_len, file_prefix,
                                                             task_alpha, bold_alpha, beta_alpha, rho_value,
                                                             voxel_weights=voxel_weights, beta_clean=train_data["beta_clean"],
                                                             data=train_data, bold_projection=projection_signal)
                rho_projection_series.append((rho_value, y_projection_voxel))
                fold_outputs = fold_output_tracker[metrics_key]
                _append_fold_array(fold_outputs, "voxel_weights", voxel_weights, combo_label)
                _append_fold_array(fold_outputs, "bold_corr", voxel_correlations, combo_label)
                _append_fold_array(fold_outputs, "beta_corr", beta_voxel_correlations, combo_label)
                _append_fold_array(fold_outputs, "projection", y_projection_voxel, combo_label)

                train_metrics = evaluate_projection(train_data, component_weights)
                test_metrics = evaluate_projection(test_data, component_weights)

                print(f"  Solution branch: {solution['branch']}", flush=True)
                print(f"  Penalty contributions: {solution['penalty_contributions']}", flush=True)
                print(f"  Total loss (train objective): {solution['total_loss']:.6f}", flush=True)
                print(f"  Train metrics -> corr: {train_metrics['pearson']:.4f}, R2: {train_metrics['r2']:.4f}", flush=True)
                print(f"  Test metrics  -> corr: {test_metrics['pearson']:.4f}, R2: {test_metrics['r2']:.4f}", flush=True)

                metrics_key = (task_alpha, bold_alpha, beta_alpha, rho_value)
                bucket = aggregate_metrics[metrics_key]
                _append_if_finite(bucket["train_corr"], train_metrics["pearson"])
                _append_if_finite(bucket["train_r2"], train_metrics["r2"])
                _append_if_finite(bucket["test_corr"], test_metrics["pearson"])
                _append_if_finite(bucket["test_r2"], test_metrics["r2"])

                if bootstrap_samples > 0:
                    print(f"  Running weight-shuffle analysis with {bootstrap_samples} samples...", flush=True)
                    shuffle_results = perform_weight_shuffle_tests(train_data, test_data, component_weights, task_alpha, bold_alpha, beta_alpha, rho_value, num_samples=bootstrap_samples)

                    # succeeded = shuffle_results["num_successful"]
                    # requested = shuffle_results["num_requested"]
                    # failed = shuffle_results["num_failed"]
                    # print(f"    Weight-shuffle successes: {succeeded}/{requested} (failed: {failed})", flush=True)

                    def format_float(value, digits):
                        return f"{value:.{digits}f}" if np.isfinite(value) else "nan"

                    for label, summary in shuffle_results["penalties"].items():
                        mean_str = format_float(summary["mean"], 6)
                        ci_lower = format_float(summary["ci_lower"], 6)
                        ci_upper = format_float(summary["ci_upper"], 6)
                        p_value = format_float(summary["p_value"], 4)
                        print(f"    Penalty {label}: mean={mean_str}, 95% CI=[{ci_lower}, {ci_upper}], p={p_value}", flush=True)

                    total_summary = shuffle_results.get("total_loss", {})
                    if total_summary:
                        mean_str = format_float(total_summary.get("mean", np.nan), 6)
                        ci_lower = format_float(total_summary.get("ci_lower", np.nan), 6)
                        ci_upper = format_float(total_summary.get("ci_upper", np.nan), 6)
                        p_value = format_float(total_summary.get("p_value", np.nan), 4)
                        print(f"    Total loss: mean={mean_str}, 95% CI=[{ci_lower}, {ci_upper}], p={p_value}", flush=True)

                    for split, summary in shuffle_results["correlation"].items():
                        mean_str = format_float(summary["mean"], 4)
                        ci_lower = format_float(summary["ci_lower"], 4)
                        ci_upper = format_float(summary["ci_upper"], 4)
                        p_value = format_float(summary["p_value"], 4)
                        print(f"    {split.capitalize()} correlation: mean={mean_str}, 95% CI=[{ci_lower}, {ci_upper}], p={p_value}", flush=True)

                    penalty_samples = shuffle_results.get("penalty_samples", {})
                    correlation_samples = shuffle_results.get("correlation_samples", {})

                    for label in ("task", "bold", "beta"):
                        samples = penalty_samples.get(label)
                        if samples is None or samples.size == 0:
                            continue
                        actual_value = float(solution["penalty_contributions"].get(label, np.nan))
                        summary = shuffle_results["penalties"].get(label, {})
                        p_value = summary.get("p_value", np.nan)

                        # penalty_plot_path = f"bootstrap_distribution_{file_prefix}_penalty_{label}.png"
                        # created_path = plot_bootstrap_distribution(samples, actual_value, p_value, f"{label.capitalize()} penalty bootstrap", "Penalty value", penalty_plot_path)

                    correlation_actuals = {"train": float(train_metrics["pearson"]) if np.isfinite(train_metrics["pearson"]) else np.nan,
                                           "test": float(test_metrics["pearson"]) if np.isfinite(test_metrics["pearson"]) else np.nan}

                    for split, samples in correlation_samples.items():
                        actual_value = correlation_actuals.get(split, np.nan)
                        summary = shuffle_results["correlation"].get(split, {})
                        p_value = summary.get("p_value", np.nan)
                        correlation_plot_path = f"bootstrap_distribution_{file_prefix}_{split}_correlation.png"
                        created_path = plot_bootstrap_distribution(samples, actual_value, p_value, f"{split.capitalize()} correlation bootstrap", "Correlation coefficient", correlation_plot_path)
                        # if created_path:
                        #     print(f"    Saved {split} correlation bootstrap plot to {created_path}", flush=True)

                    total_samples = shuffle_results.get("total_loss_samples")
                    if total_samples is not None and np.size(total_samples) > 0:
                        summary = shuffle_results.get("total_loss", {})
                        p_value = summary.get("p_value", np.nan)
                        total_plot_path = f"bootstrap_distribution_{file_prefix}_total_loss.png"
                        created_path = plot_bootstrap_distribution(total_samples, solution["total_loss"], p_value, "Total loss weight shuffle", "Total loss", total_plot_path)
                        # if created_path:
                            # print(f"    Saved total loss shuffle plot to {created_path}", flush=True)

        if rho_projection_series:
            aggregate_plot_path = f"y_projection_trials_voxel_{alpha_prefix}_all_rhos.png"
            plot_projection_beta_sweep(rho_projection_series, task_alpha, bold_alpha, beta_alpha, aggregate_plot_path, series_label="Voxel space")

    if fold_output_tracker:
        print("\n===== Saving fold-averaged spatial maps and projections =====", flush=True)
        active_coords = projection_data.get("active_coords") or ()
        volume_shape = anat_img.shape[:3]
        for metrics_key in sorted(fold_output_tracker.keys()):
            task_alpha, bold_alpha, beta_alpha, rho_value = metrics_key
            fold_outputs = fold_output_tracker[metrics_key]
            avg_prefix = f"foldavg_sub{sub}_ses{ses}_task{task_alpha:g}_bold{bold_alpha:g}_beta{beta_alpha:g}_rho{rho_value:g}"
            weights_avg = _stack_nanmean(fold_outputs["voxel_weights"])
            if weights_avg is not None:
                save_active_bold_correlation_map(weights_avg, active_coords, volume_shape, anat_img, avg_prefix,
                                                 result_prefix="voxel_weights")
            bold_corr_avg = _stack_nanmean(fold_outputs["bold_corr"])
            if bold_corr_avg is not None:
                save_active_bold_correlation_map(bold_corr_avg, active_coords, volume_shape, anat_img, avg_prefix)
            beta_corr_avg = _stack_nanmean(fold_outputs["beta_corr"])
            if beta_corr_avg is not None:
                save_active_bold_correlation_map(beta_corr_avg, active_coords, volume_shape, anat_img, avg_prefix,
                                                 result_prefix=f"active_beta_corr_{metric_label}")
            projection_avg = _stack_nanmean(fold_outputs["projection"])
            if projection_avg is not None:
                total_points = projection_avg.size
                if total_points % trial_len == 0:
                    _, avg_trials = _reshape_projection(projection_avg, trial_len)
                    avg_plot_path = f"y_projection_trials_{avg_prefix}.png"
                    plot_projection_bold(avg_trials, task_alpha, bold_alpha, beta_alpha, rho_value, avg_plot_path,
                                         series_label="Voxel space (fold avg)")
                else:
                    print(f"Skipping projection averaging for task={task_alpha:g}, bold={bold_alpha:g}, "
                          f"beta={beta_alpha:g}, rho={rho_value:g} due to incompatible length {total_points}", flush=True)

    if aggregate_metrics:
        print("\n===== Cross-fold average metrics =====", flush=True)
        def _safe_mean(values):
            return float(np.mean(values)) if values else np.nan

        for metrics_key in sorted(aggregate_metrics.keys()):
            task_alpha, bold_alpha, beta_alpha, rho_value = metrics_key
            bucket = aggregate_metrics[metrics_key]
            train_corr_mean = _safe_mean(bucket["train_corr"])
            train_r2_mean = _safe_mean(bucket["train_r2"])
            test_corr_mean = _safe_mean(bucket["test_corr"])
            test_r2_mean = _safe_mean(bucket["test_r2"])
            fold_count = len(bucket["test_corr"])
            print(f"task={task_alpha:g}, bold={bold_alpha:g}, beta={beta_alpha:g}, rho={rho_value:g} "
                f"(folds contributing={fold_count}): "
                f"train corr={train_corr_mean:.4f}, test corr={test_corr_mean:.4f}, "
                f"train R2={train_r2_mean:.4f}, test R2={test_r2_mean:.4f}", flush=True)

if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Run a subset of alpha/rho sweeps.")
#     parser.add_argument("--alpha-idx", type=int, default=None,
#                         help="0-based index into alpha_sweep; omit to iterate over all alphas.")
#     parser.add_argument("--rho-idx", type=int, default=None,
#                         help="0-based index into rho_sweep; omit to iterate over all rhos.")
#     parser.add_argument("--combo-idx", type=int, default=None,
#                         help="Single 0-based index over alpha×rho combinations (overrides --alpha-idx/--rho-idx).")
#     args = parser.parse_args()

#     combo_idx = args.combo_idx
#     if combo_idx is None:
#         env_combo = os.environ.get("SLURM_ARRAY_TASK_ID")
#         if env_combo is not None:
#             try:
#                 combo_idx = int(env_combo)
#             except ValueError as exc:
#                 raise ValueError("SLURM_ARRAY_TASK_ID must be an integer when provided.") from exc

    selected_alphas = alpha_sweep
    selected_rhos = rho_sweep

#     if combo_idx is not None:
#         total = len(alpha_sweep) * len(rho_sweep)
#         if not 0 <= combo_idx < total:
#             raise ValueError(f"combo_idx must be in [0, {total})")
#         alpha_idx, rho_idx = divmod(combo_idx, len(rho_sweep))
#         selected_alphas = [alpha_sweep[alpha_idx]]
#         selected_rhos = [rho_sweep[rho_idx]]
#         print(f"Running combo_idx={combo_idx} (alpha_idx={alpha_idx}, rho_idx={rho_idx})", flush=True)
#     else:
#         if args.alpha_idx is not None:
#             if not 0 <= args.alpha_idx < len(alpha_sweep):
#                 raise ValueError("alpha_idx out of range")
#             selected_alphas = [alpha_sweep[args.alpha_idx]]
#             print(f"Running alpha_idx={args.alpha_idx}", flush=True)
#         if args.rho_idx is not None:
#             if not 0 <= args.rho_idx < len(rho_sweep):
#                 raise ValueError("rho_idx out of range")
#             selected_rhos = [rho_sweep[args.rho_idx]]
#             print(f"Running rho_idx={args.rho_idx}", flush=True)

    fold_splits = build_custom_kfold_splits(num_trials, num_folds=5, trials_per_run=trials_per_run)
    run_cross_run_experiment(selected_alphas, selected_rhos, fold_splits, projection_data, bootstrap_samples=bootstrap_iterations)

# # %%
