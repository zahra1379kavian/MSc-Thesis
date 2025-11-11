import nibabel as nib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from os.path import join
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
from scipy.io import loadmat
import cvxpy as cp
from nilearn import plotting
matplotlib.use("Agg")
import argparse
# %% 
ses = 1
sub = "04"
run_train = 1
run_test = 2
num_trials = 90
trial_len = 9
behave_indice = 1 #1/RT

base_path = "/scratch/st-mmckeown-1/zkavian/fmri_models/MSc-Thesis/"
brain_mask_path = f"{base_path}/sub-pd0{sub}_ses-{ses}_T1w_brain_mask.nii.gz"
csf_mask_path = f"{base_path}/sub-pd0{sub}_ses-{ses}_T1w_brain_pve_0.nii.gz"
gray_mask_path = f"{base_path}/sub-pd0{sub}_ses-{ses}_T1w_brain_pve_1.nii.gz"

# %%
anat_img = nib.load(f"{base_path}/sub-pd0{sub}_ses-{ses}_T1w_brain.nii.gz")
bold_data_train = np.load(join(base_path, f'fmri_sub{sub}_ses{ses}_run{run_train}.npy'))
bold_data_test = np.load(join(base_path, f'fmri_sub{sub}_ses{ses}_run{run_test}.npy'))
brain_mask = nib.load(brain_mask_path).get_fdata().astype(np.float16)
csf_mask = nib.load(csf_mask_path).get_fdata().astype(np.float16)
gray_mask = nib.load(gray_mask_path).get_fdata().astype(np.float16)
glm_dict = np.load(f"{base_path}/TYPED_FITHRF_GLMDENOISE_RR_sub{sub}.npy", allow_pickle=True).item()
beta_glm = glm_dict["betasmd"]
beta_volume_filter_train = np.load(f"{base_path}/cleaned_beta_volume_sub{sub}_ses{ses}_run{run_train}.npy")
beta_volume_filter_test = np.load(f"{base_path}/cleaned_beta_volume_sub{sub}_ses{ses}_run{run_test}.npy")
pca_model_train = np.load(f"{base_path}/empca_model_sub{sub}_ses{ses}_run{run_train}.npy", allow_pickle=True).item()

# %%
def load_behavior_metrics(path, run_id):
    behav_data = loadmat(path)
    behav_metrics = behav_data["behav_metrics"]
    behav_block = np.stack(behav_metrics[0], axis=0)
    _, _, num_metrics = behav_block.shape
    behav_flat = behav_block.reshape(-1, num_metrics)
    if run_id == 1:
        behav_flat = behav_flat[:90, :6]
    else:
        behav_flat = behav_flat[90:180, :6]
    return behav_flat

behave_path = f"{base_path}/PSPD0{sub}_OFF_behav_metrics.mat" if ses == 1 else f"{base_path}/PSPD0{sub}_ON_behav_metrics.mat"
behavior_matrix_train = load_behavior_metrics(behave_path, run_train)
behavior_matrix_test = load_behavior_metrics(behave_path, run_test)

# %%
def calcu_matrices_func(beta_pca, bold_pca, behave_mat, behave_indice, trial_len=trial_len, num_trials=90):
    bold_pca_reshape = bold_pca.reshape(bold_pca.shape[0], num_trials, trial_len)
    behavior_selected = behave_mat[:, behave_indice]

    print("L_task...", flush=True)
    counts = np.count_nonzero(np.isfinite(beta_pca), axis=-1)
    sums = np.nansum(np.abs(beta_pca), axis=-1, dtype=np.float32)
    mean_beta = np.zeros(beta_pca.shape[0], dtype=np.float32)
    mask = counts > 0
    mean_beta[mask] = (sums[mask] / counts[mask]).astype(np.float32)

    C_task = np.zeros_like(mean_beta, dtype=np.float32)
    valid = np.abs(mean_beta) > 0
    C_task[valid] = (1.0 / mean_beta[valid]).astype(np.float32)
    C_task = np.diag(C_task)

    print("L_var...", flush=True)
    C_bold = np.zeros((bold_pca_reshape.shape[0], bold_pca_reshape.shape[0]), dtype=np.float32)
    for i in range(num_trials - 1):
        x1 = bold_pca_reshape[:, i, :]
        x2 = bold_pca_reshape[:, i + 1, :]
        C_bold += (x1 - x2) @ (x1 - x2).T
    C_bold /= (num_trials - 1)

    print("L_var...", flush=True)
    C_beta = np.zeros((beta_pca.shape[0], beta_pca.shape[0]), dtype=np.float32)
    for i in range(num_trials - 1):
        x1 = beta_pca[:, i]
        x2 = beta_pca[:, i + 1]
        diff = x1 - x2
        C_beta += np.outer(diff, diff)
    C_beta /= (num_trials - 1)

    return C_task, C_bold, C_beta, behavior_selected

# %%
def standardize_matrix(matrix):
    array = np.asarray(matrix, dtype=np.float64)

    sym_array = 0.5 * (array + array.T)
    if not np.all(np.isfinite(sym_array)):
        sym_array = np.nan_to_num(sym_array, nan=0.0, posinf=0.0, neginf=0.0)

    trace_value = np.trace(sym_array)
    if not np.isfinite(trace_value) or trace_value == 0:
        trace_value = 1.0
    
    # print(f"range before: {np.min(array)}, {np.max(array)}")
    # print(f"trace: {trace_value}")
    normalized = sym_array / trace_value
    normalized *= sym_array.shape[0]
    print(f"range after: {np.min(normalized)}, {np.max(normalized)}")
    return normalized.astype(matrix.dtype, copy=False)

def _extract_active_bold_trials(bold_timeseries, coords, num_trials, trial_length):
    if coords is None or len(coords) != 3:
        return None
    coord_arrays = tuple(np.asarray(axis, dtype=int) for axis in coords)
    if coord_arrays[0].size == 0:
        return np.zeros((0, num_trials, trial_length), dtype=np.float32)

    bold_active = bold_timeseries[coord_arrays[0], coord_arrays[1], coord_arrays[2], :]
    num_voxels, num_timepoints = bold_active.shape
    active_bold = np.full((num_voxels, num_trials, trial_length), np.nan, dtype=np.float32)

    start_idx = 0
    for trial_idx in range(num_trials):
        end_idx = start_idx + trial_length
        if end_idx > num_timepoints:
            raise ValueError("Active BOLD data does not contain enough timepoints for all trials.")
        active_bold[:, trial_idx, :] = bold_active[:, start_idx:end_idx]
        start_idx += trial_length
        if start_idx in (270, 560):
            start_idx += 20
    return active_bold

# %%
def prepare_data_func(run_id, bold_timeseries, beta_glm_volume, filtered_beta_volume,
                      brain_mask, csf_mask, gray_matter_mask_volume, 
                      behavioral_matrix, pca_model, trial_length, num_trials, 
                      reuse_nan_mask=None, reuse_active_coords=None):

    print(f"Preparing data for run {run_id}...", flush=True)
    gray_matter_mask = gray_matter_mask_volume > 0.5
    brain_mask = brain_mask > 0
    csf_mask = csf_mask > 0
    brain_without_csf = brain_mask & ~csf_mask
    gray_voxel_mask_volume = gray_matter_mask & brain_without_csf
    gray_voxel_mask = gray_voxel_mask_volume[brain_without_csf]
    gray_voxel_coords = np.where(gray_voxel_mask_volume)

    bold_gray = bold_timeseries[gray_voxel_coords]
    num_voxels, num_timepoints = bold_gray.shape
    bold_by_trial = np.full((num_voxels, num_trials, trial_length), np.nan, dtype=np.float32)
    start_idx = 0
    for trial_idx in range(num_trials):
        end_idx = start_idx + trial_length
        if end_idx > num_timepoints:
            raise ValueError("Masked BOLD data does not contain enough timepoints for all trials.")
        bold_by_trial[:, trial_idx, :] = bold_gray[:, start_idx:end_idx]
        start_idx += trial_length
        if start_idx in (270, 560):
            start_idx += 20

    print("Masking GLM betas...", flush=True)
    beta_run1, beta_run2 = beta_glm_volume[:, 0, 0, :90], beta_glm_volume[:, 0, 0, 90:]
    beta_volume = beta_run1 if run_id == 1 else beta_run2
    beta_gray = beta_volume[gray_voxel_mask]

    skipping_preprocessing = reuse_nan_mask is not None
    active_bold = None
    active_flat_indices = None

    if skipping_preprocessing:
        print("Reusing voxel mask from training run; skipping preprocessing.", flush=True)
        nan_mask_flat = np.asarray(reuse_nan_mask, dtype=bool).ravel()
        active_coords = tuple(np.array(coord) for coord in reuse_active_coords)
        active_flat_indices = np.ravel_multi_index(active_coords, filtered_beta_volume.shape[:3])
        active_bold = _extract_active_bold_trials(bold_timeseries, active_coords, num_trials, trial_length)
    else:
        nan_voxels = np.isnan(beta_gray).all(axis=1)
        if np.any(nan_voxels):
            beta_gray = beta_gray[~nan_voxels]
            bold_by_trial = bold_by_trial[~nan_voxels]
            gray_voxel_coords = tuple(coord[~nan_voxels] for coord in gray_voxel_coords)

        print("Removing outliers in beta estimates...", flush=True)
        median_beta = np.nanmedian(beta_gray, keepdims=True)
        mad_beta = np.nanmedian(np.abs(beta_gray - median_beta), keepdims=True)
        beta_scale = 1.4826 * np.maximum(mad_beta, 1e-9)

        normalized_beta = (beta_gray - median_beta) / beta_scale
        outlier_threshold = np.nanpercentile(np.abs(normalized_beta), 99.9)
        outlier_mask = np.abs(normalized_beta) > outlier_threshold

        voxel_outlier_fraction = np.nanmean(outlier_mask, axis=1)
        keep_voxel = voxel_outlier_fraction <= 0.5
        trial_outliers = outlier_mask[keep_voxel]
        gray_voxel_coords = tuple(coord[keep_voxel] for coord in gray_voxel_coords)
        # keep_trial = np.logical_not(outlier_mask[keep_voxel])

        cleaned_beta = beta_gray[keep_voxel]
        cleaned_beta[trial_outliers] = np.nan 
        retained_mask = ~np.all(np.isnan(cleaned_beta), axis=1)
        cleaned_beta = cleaned_beta[retained_mask]
        retained_indices = np.flatnonzero(keep_voxel)[retained_mask]

        bold_by_trial = bold_by_trial[keep_voxel]
        bold_by_trial = np.where(outlier_mask[keep_voxel, :, None], np.nan, bold_by_trial)
        bold_by_trial = bold_by_trial[retained_mask]
        gray_voxel_coords = tuple(coord[retained_mask] for coord in gray_voxel_coords)

        print("Filtering active voxels via t-test...", flush=True)
        pvals = ttest_1samp(cleaned_beta, popmean=0, axis=1, nan_policy="omit").pvalue
        tested_mask = np.isfinite(pvals)
        rejected = np.zeros_like(tested_mask)
        rejected[tested_mask] = multipletests(pvals[tested_mask], alpha=0.05, method="fdr_bh")[0]
        active_mask = rejected.astype(bool)

        active_beta = cleaned_beta[active_mask]
        active_indices = retained_indices[active_mask]
        active_bold = bold_by_trial[active_mask]
        active_coords = tuple(coord[active_mask] for coord in gray_voxel_coords)

        beta_nan_mask = np.all(np.isnan(filtered_beta_volume), axis=-1)
        nan_mask_flat = beta_nan_mask.ravel()
        active_flat_indices = np.ravel_multi_index(active_coords, filtered_beta_volume.shape[:3])
        keep_mask = ~nan_mask_flat[active_flat_indices]

        active_beta = active_beta[keep_mask]
        active_bold = active_bold[keep_mask]
        active_flat_indices = active_flat_indices[keep_mask]
        active_indices = active_indices[keep_mask]
        active_coords = tuple(coord[keep_mask] for coord in active_coords)

    print("Applying PCA projection...", flush=True)
    volume_flat = filtered_beta_volume.reshape(-1, filtered_beta_volume.shape[-1])
    beta_volume_clean = volume_flat[~nan_mask_flat]

    bold_pca_components = pca_model.eigvec
    coeff_pinv = np.linalg.pinv(pca_model.coeff)
    beta_pca = coeff_pinv @ np.nan_to_num(beta_volume_clean)

    (C_task, C_bold, C_beta, behavior_vector) = calcu_matrices_func(beta_pca, bold_pca_components, behavioral_matrix, 
                                                                    behave_indice, trial_len=trial_length, num_trials=num_trials)

    trial_mask = np.isfinite(behavior_vector)
    behavior_observed = behavior_vector[trial_mask]
    behavior_mean = np.mean(behavior_observed)
    behavior_centered = behavior_observed - behavior_mean
    behavior_norm = np.linalg.norm(behavior_centered)
    normalized_behaviors = behavior_centered / behavior_norm

    beta_observed = beta_pca[:, trial_mask]
    beta_centered = beta_observed - np.mean(beta_observed, axis=1, keepdims=True)

    C_task = standardize_matrix(C_task)
    C_bold = standardize_matrix(C_bold)
    C_beta = standardize_matrix(C_beta)

    return {"nan_mask_flat": nan_mask_flat, "active_coords": tuple(np.array(coord) for coord in active_coords),
        "coeff_pinv": coeff_pinv, "beta_centered": beta_centered, "behavior_centered": behavior_centered, 
        "normalized_behaviors": normalized_behaviors, "behavior_observed": behavior_observed, "beta_observed": beta_observed,
        "beta_volume_clean": beta_volume_clean, "C_task": C_task, "C_bold": C_bold, "C_beta": C_beta,
        "active_bold": active_bold, "active_flat_indices": active_flat_indices}

# %%
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

# %%
def solve_soc_problem(run_data, alpha_task, alpha_bold, alpha_beta, solver_name, phro):
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
    print(f"lambda Positive: {np.sum(lam_nonneg)}, {lam_sum1}, {lam_soc_tau}, {np.sum(lam_soc_vec)}")

    lam_nonneg   = constraints_negative[0].dual_value   # vector (same size as w) for w >= 0
    lam_sum1     = constraints_negative[1].dual_value   # scalar for sum(w) == 1
    lam_soc_tau, lam_soc_vec = constraints_negative[2].dual_value
    print(f"lambda Negative: {np.sum(lam_nonneg)}, {lam_sum1}, {lam_soc_tau}, {np.sum(lam_soc_vec)}")

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
        raise RuntimeError(
            f"SOC solver failed to find a feasible solution "
            f"(positive status={problem_positive.status}, negative status={problem_negative.status})."
        )

    contributions = {label: float(solution_weights.T @ matrix @ solution_weights) for label, matrix in penalty_matrices.items()}

    y = beta_centered.T @ solution_weights
    total_loss = float(solution_weights.T @ total_penalty @ solution_weights)

    return {"weights": solution_weights, "branch": solution_branch, "total_loss": total_loss, "Y": y,
            "objective_positive": objective_positive, "objective_negative": objective_negative, "penalty_contributions": contributions}

# %%
def evaluate_projection(run_data, weights):
    beta_centered = run_data["beta_centered"]
    behavior_centered = run_data["behavior_centered"]
    normalized_behaviors = run_data["normalized_behaviors"]

    projection = beta_centered.T @ weights
    finite_mask = np.isfinite(projection) & np.isfinite(behavior_centered)
    projection = projection[finite_mask]
    behavior_centered = behavior_centered[finite_mask]
    normalized_behaviors = normalized_behaviors[finite_mask]

    metrics = {"pearson": np.nan, "r2": np.nan, "mse": np.nan, "soc_slack": np.nan}
    metrics["pearson"] = np.corrcoef(projection, behavior_centered)[0, 1]
    residuals = behavior_centered - projection
    metrics["mse"] = np.mean(residuals**2)
    sst = np.sum(behavior_centered**2)
    metrics["r2"] = 1 - np.sum(residuals**2) / sst

    return metrics

def summarize_bootstrap(values, reference=0.0, direction="less"):
    values_array = np.asarray(values, dtype=np.float64)
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

# %%
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

    abs_weights = np.abs(weights)
    fallback_scale = np.max(abs_weights) * 1e-2
    if not np.isfinite(fallback_scale) or fallback_scale <= 0:
        fallback_scale = 1e-8
    noise_scale = np.maximum(abs_weights, fallback_scale)

    for iteration in range(num_samples):
        noise = rng.normal(loc=0.0, scale=noise_scale, size=weights.shape)
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

    num_successful = int(np.count_nonzero(np.isfinite(test_corr_array)))
    num_failed = int(num_samples - num_successful)

    return {"penalties": penalty_summary, "total_loss": total_loss_summary, "correlation": correlation_summary, 
            "num_requested": num_samples, "num_successful": num_successful, "num_failed": num_failed,
            "penalty_samples": penalty_arrays, "total_loss_samples": total_loss_array, "correlation_samples": {"train": train_corr_array, "test": test_corr_array}}

# %%
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

def _reshape_projection(y_values, trial_length):
    y_values = np.asarray(y_values, dtype=np.float64).ravel()
    if y_values.size % trial_length != 0:
        raise ValueError(f"Projection length {y_values.size} is not divisible by trial length {trial_length}.")
    num_trials = y_values.size // trial_length
    return y_values, y_values.reshape(num_trials, trial_length)

def compute_component_active_bold_correlation(voxel_weights, run_data):
    active_bold = run_data.get("active_bold")
    active_flat_indices = run_data.get("active_flat_indices")
    nan_mask_flat = run_data.get("nan_mask_flat")
    active_flat_indices = active_flat_indices.ravel()

    valid_flat_indices = np.flatnonzero(~np.asarray(nan_mask_flat, dtype=bool).ravel())
    voxel_weights = voxel_weights.ravel()
    positions = np.searchsorted(valid_flat_indices, active_flat_indices)
    if np.any(positions >= valid_flat_indices.size) or not np.array_equal(valid_flat_indices[positions], active_flat_indices):
        raise ValueError("Active voxel indices could not be mapped to voxel weight entries.")

    active_voxel_weights = voxel_weights[positions]
    bold_matrix = np.asarray(active_bold, dtype=np.float64).reshape(active_bold.shape[0], -1)
    timepoints = bold_matrix.shape[1]
    projection = np.full(timepoints, np.nan, dtype=np.float64)

    finite_mask = np.isfinite(bold_matrix)
    for t_idx in range(timepoints):
        voxel_mask = finite_mask[:, t_idx]
        if not np.any(voxel_mask):
            continue
        projection[t_idx] = np.dot(active_voxel_weights[voxel_mask], bold_matrix[voxel_mask, t_idx])

    correlations = np.full(bold_matrix.shape[0], np.nan, dtype=np.float32)
    projection_finite = np.isfinite(projection)
    for voxel_idx, voxel_series in enumerate(bold_matrix):
        voxel_finite = np.isfinite(voxel_series)
        joint_mask = projection_finite & voxel_finite
        if np.count_nonzero(joint_mask) < 2:
            continue
        correlations[voxel_idx] = np.corrcoef(projection[joint_mask], voxel_series[joint_mask])[0, 1]
    return projection, correlations

def save_active_bold_correlation_map(correlations, active_coords, volume_shape, anat_img, file_prefix):
    if correlations is None or active_coords is None or volume_shape is None or anat_img is None:
        return {}

    coord_arrays = tuple(np.asarray(axis, dtype=int) for axis in active_coords)
    corr_path = f"active_bold_corr_{file_prefix}.npy"
    np.save(corr_path, correlations.astype(np.float32))

    volume = np.full(volume_shape, np.nan, dtype=np.float32)
    if coord_arrays and coord_arrays[0].size:
        volume[coord_arrays] = correlations

    corr_img = nib.Nifti1Image(volume, anat_img.affine, anat_img.header)
    volume_path = f"active_bold_corr_{file_prefix}.nii.gz"
    nib.save(corr_img, volume_path)

    if np.any(np.isfinite(correlations)):
        display = plotting.view_img(corr_img, bg_img=anat_img, colorbar=True, symmetric_cmap=False,
            title="corr(component_weights@active_bold, active_bold)")
        display.save_as_html(f"active_bold_corr_{file_prefix}.html")

    return

def plot_projection_trials(y_trials, task_alpha, bold_alpha, beta_alpha, rho_value, output_path, max_trials=30):
    num_trials_total, trial_length = y_trials.shape
    num_trials = min(max_trials, num_trials_total)
    display_trials = y_trials[:num_trials]
    time_axis = np.arange(trial_length)
    finite_max = np.nanmax(np.abs(display_trials)) if np.any(np.isfinite(display_trials)) else 1.0
    if not np.isfinite(finite_max) or finite_max <= 0:
        finite_max = 1.0
    spacing = finite_max * 1.5 + 1e-3
    fig_height = min(14.0, max(6.0, 4.0 + num_trials * 0.05))

    fig, ax = plt.subplots(figsize=(9, fig_height))
    for idx, trial_values in enumerate(display_trials):
        offset = idx * spacing
        ax.plot(time_axis, trial_values + offset, linewidth=0.8, alpha=0.9, color="#1f77b4")

    ax.set_xlabel("Time point")
    ax.set_ylabel("Trial index (offset)")
    subtitle = ""
    if num_trials_total > num_trials:
        subtitle = f" (showing first {num_trials} / {num_trials_total})"
    ax.set_xlim(time_axis[0], time_axis[-1])
    y_ticks = np.arange(num_trials) * spacing
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(i + 1) for i in range(num_trials)])
    ax.set_title(f"y | alpha_task={task_alpha:g}, alpha_bold={bold_alpha:g}, alpha_beta={beta_alpha:g}, rho={rho_value:g}{subtitle}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path

def save_projection_outputs(pca_weights, bold_pca_components, trial_length, file_prefix,
                            task_alpha, bold_alpha, beta_alpha, rho_value):
    bold_pca_components = np.asarray(bold_pca_components, dtype=np.float64)
    y_projection = np.asarray(pca_weights, dtype=np.float64) @ bold_pca_components
    y_vector_path = f"y_projection_{file_prefix}.npy"
    np.save(y_vector_path, y_projection.astype(np.float32))

    _, y_trials = _reshape_projection(y_projection, trial_length)
    y_trials_path = f"y_projection_trials_{file_prefix}.npy"
    np.save(y_trials_path, y_trials.astype(np.float32))

    plot_path = f"y_projection_trials_{file_prefix}.png"
    plot_projection_trials(y_trials, task_alpha, bold_alpha, beta_alpha, rho_value, plot_path)
    return 
# %%
ridge_penalty = 1e-6
solver_name = "MOSEK"
soc_ratio = 0.95

alpha_sweep = [{"task_penalty": 0.5, "bold_penalty": 0.25, "beta_penalty": 50}]
rho_sweep = [0.2, soc_ratio]
bootstrap_iterations = 1000

def run_cross_run_experiment(alpha_settings, rho_values, bootstrap_samples=0):
    train_data = prepare_data_func(run_train, bold_data_train, beta_glm, beta_volume_filter_train,
                                   brain_mask, csf_mask, gray_mask, behavior_matrix_train, pca_model_train, trial_len, num_trials)

    test_data = prepare_data_func(run_test, bold_data_test, beta_glm, beta_volume_filter_test,
                                 brain_mask, csf_mask, gray_mask,  behavior_matrix_test, pca_model_train, trial_len, num_trials, 
                                 reuse_nan_mask=train_data["nan_mask_flat"], reuse_active_coords=train_data["active_coords"])

    bold_pca_components = np.asarray(pca_model_train.eigvec, dtype=np.float64)

    for alpha_setting in alpha_settings:
        task_alpha = alpha_setting["task_penalty"]
        bold_alpha = alpha_setting["bold_penalty"]
        beta_alpha = alpha_setting["beta_penalty"]

        for rho_value in rho_values:
            sweep_suffix = f"task{task_alpha:g}_bold{bold_alpha:g}_beta{beta_alpha:g}_rho{rho_value:g}"
            print(f"Cross-run optimization with task={task_alpha}, bold={bold_alpha}, beta={beta_alpha}, rho={rho_value}", flush=True)
            solution = solve_soc_problem(train_data, task_alpha, bold_alpha, beta_alpha, solver_name, rho_value)

            component_weights = np.abs(solution["weights"])
            coeff_pinv = np.asarray(train_data["coeff_pinv"])
            voxel_weights = coeff_pinv.T @ component_weights
            # print(f"voxel_weights shape: {voxel_weights.shape}, component_weights shape: {component_weights.shape}")

            file_prefix = f"sub{sub}_ses{ses}_trainrun{run_train}_testrun{run_test}_{sweep_suffix}"
            # np.save(f"behavior_weights_{file_prefix}.npy", pca_weights)
            # np.save(f"behavior_weights_pca_{file_prefix}.npy", weights)
            # view_path = save_weight_outputs(anat_img, train_data["active_coords"], pca_weights, file_prefix)

            beta_volume_clean = np.asarray(train_data["beta_volume_clean"], dtype=np.float64)
            beta_volume_clean_finite = np.nan_to_num(beta_volume_clean, nan=0.0, posinf=0.0, neginf=0.0)
            weighted_voxel_activity = voxel_weights[:, None] * beta_volume_clean_finite
            weighted_beta_path = f"beta_weighted_activity_{file_prefix}.npy"
            np.save(weighted_beta_path, weighted_voxel_activity.astype(np.float32))

            corr_outputs = {}
            try:
                projection_signal, voxel_correlations = compute_component_active_bold_correlation(voxel_weights, train_data)
            except ValueError as exc:
                projection_signal, voxel_correlations = None, None
                print(f"Skipping active BOLD correlation computation: {exc}", flush=True)

            if projection_signal is not None and np.any(np.isfinite(projection_signal)):
                projection_path = f"component_active_bold_projection_{file_prefix}.npy"
                np.save(projection_path, projection_signal.astype(np.float32))

            if voxel_correlations is not None:
                save_active_bold_correlation_map(voxel_correlations,
                    train_data.get("active_coords"), beta_volume_filter_train.shape[:3], anat_img, file_prefix)

            save_projection_outputs(component_weights, bold_pca_components, trial_len, file_prefix, 
                                    task_alpha, bold_alpha, beta_alpha, rho_value)

            train_metrics = evaluate_projection(train_data, component_weights)
            test_metrics = evaluate_projection(test_data, component_weights)

            print(f"  Solution branch: {solution['branch']}", flush=True)
            print(f"  Penalty contributions: {solution['penalty_contributions']}", flush=True)
            print(f"  Total loss (train objective): {solution['total_loss']:.6f}", flush=True)
            print(f"  Train metrics -> corr: {train_metrics['pearson']:.4f}, R2: {train_metrics['r2']:.4f}", flush=True)
            print(f"  Test metrics  -> corr: {test_metrics['pearson']:.4f}, R2: {test_metrics['r2']:.4f}", flush=True)

            if bootstrap_samples > 0:
                print(f"  Running weight-shuffle analysis with {bootstrap_samples} samples...", flush=True)
                shuffle_results = perform_weight_shuffle_tests(
                    train_data, test_data, component_weights, task_alpha, bold_alpha, beta_alpha, rho_value, num_samples=bootstrap_samples)

                succeeded = shuffle_results["num_successful"]
                requested = shuffle_results["num_requested"]
                failed = shuffle_results["num_failed"]
                print(f"    Weight-shuffle successes: {succeeded}/{requested} (failed: {failed})", flush=True)

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

                    penalty_plot_path = f"bootstrap_distribution_{file_prefix}_penalty_{label}.png"
                    created_path = plot_bootstrap_distribution(samples, actual_value, p_value, f"{label.capitalize()} penalty bootstrap", "Penalty value", penalty_plot_path)

                correlation_actuals = {"train": float(train_metrics["pearson"]) if np.isfinite(train_metrics["pearson"]) else np.nan,
                                       "test": float(test_metrics["pearson"]) if np.isfinite(test_metrics["pearson"]) else np.nan}

                for split, samples in correlation_samples.items():
                    if samples is None or samples.size == 0:
                        continue
                    actual_value = correlation_actuals.get(split, np.nan)
                    summary = shuffle_results["correlation"].get(split, {})
                    p_value = summary.get("p_value", np.nan)
                    correlation_plot_path = f"bootstrap_distribution_{file_prefix}_{split}_correlation.png"
                    created_path = plot_bootstrap_distribution(samples, actual_value, p_value, f"{split.capitalize()} correlation bootstrap", "Correlation coefficient", correlation_plot_path)
                    if created_path:
                        print(f"    Saved {split} correlation bootstrap plot to {created_path}", flush=True)

                total_samples = shuffle_results.get("total_loss_samples")
                if total_samples is not None and np.size(total_samples) > 0:
                    summary = shuffle_results.get("total_loss", {})
                    p_value = summary.get("p_value", np.nan)
                    total_plot_path = f"bootstrap_distribution_{file_prefix}_total_loss.png"
                    created_path = plot_bootstrap_distribution(total_samples, solution["total_loss"], p_value, "Total loss weight shuffle", "Total loss", total_plot_path)
                    if created_path:
                        print(f"    Saved total loss shuffle plot to {created_path}", flush=True)


if __name__ == "__main__":
    run_cross_run_experiment(alpha_sweep, rho_sweep, bootstrap_samples=bootstrap_iterations)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Run a subset of alpha/rho sweeps.")
#     parser.add_argument("--alpha-idx", type=int, default=None,
#                         help="0-based index into alpha_sweep; omit to run all.")
#     parser.add_argument("--rho-idx", type=int, default=None,
#                         help="0-based index into rho_sweep; omit to run all.")
#     parser.add_argument("--combo-idx", type=int, default=None,
#                         help="Single 0-based index over alpha×rho combinations.")
#     args = parser.parse_args()

#     selected_alphas = alpha_sweep
#     selected_rhos = rho_sweep

#     if args.combo_idx is not None:
#         total = len(alpha_sweep) * len(rho_sweep)
#         if not 0 <= args.combo_idx < total:
#             raise ValueError(f"combo_idx must be in [0, {total})")
#         alpha_idx, rho_idx = divmod(args.combo_idx, len(rho_sweep))
#         selected_alphas = [alpha_sweep[alpha_idx]]
#         selected_rhos = [rho_sweep[rho_idx]]
#     else:
#         if args.alpha_idx is not None:
#             if not 0 <= args.alpha_idx < len(alpha_sweep):
#                 raise ValueError("alpha_idx out of range")
#             selected_alphas = [alpha_sweep[args.alpha_idx]]
#         if args.rho_idx is not None:
#             if not 0 <= args.rho_idx < len(rho_sweep):
#                 raise ValueError("rho_idx out of range")
#             selected_rhos = [rho_sweep[args.rho_idx]]

#     run_cross_run_experiment(selected_alphas, selected_rhos, bootstrap_samples=bootstrap_iterations)

# %%
