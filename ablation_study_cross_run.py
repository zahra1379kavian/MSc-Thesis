# %%
import nibabel as nib
import numpy as np
import os
from os.path import join
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
from scipy.io import loadmat
import cvxpy as cp
from nilearn import plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ses = 1
sub = "04"
run_train = 1
run_test = 2
num_trials = 90
trial_len = 9

base_path = "/scratch/st-mmckeown-1/zkavian/fmri_models"
anat_img = nib.load(f"{base_path}/sub-pd0{sub}_ses-{ses}_T1w_brain.nii.gz")
bold_data_train = np.load(join(base_path, f'fmri_sub{sub}_ses{ses}_run{run_train}.npy'))
bold_data_test = np.load(join(base_path, f'fmri_sub{sub}_ses{ses}_run{run_test}.npy'))

mask_path = f"{base_path}/sub-pd0{sub}_ses-{ses}_T1w_brain_mask.nii.gz"
brain_mask = nib.load(mask_path).get_fdata().astype(np.float16)

mask_path = f"{base_path}/sub-pd0{sub}_ses-{ses}_T1w_brain_pve_0.nii.gz"
csf_mask = nib.load(mask_path).get_fdata().astype(np.float16)

mask_path = f"{base_path}/sub-pd0{sub}_ses-{ses}_T1w_brain_pve_1.nii.gz"
gray_mask = nib.load(mask_path).get_fdata().astype(np.float16)

glm_dict = np.load(f"TYPED_FITHRF_GLMDENOISE_RR_sub{sub}.npy", allow_pickle=True).item()
beta_glm = glm_dict["betasmd"]

beta_volume_filter_train = np.load(f"cleaned_beta_volume_sub{sub}_ses{ses}_run{run_train}.npy")
beta_volume_filter_test = np.load(f"cleaned_beta_volume_sub{sub}_ses{ses}_run{run_test}.npy")


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


behave_path = "PSPD004_OFF_behav_metrics.mat"
behavior_matrix_train = load_behavior_metrics(behave_path, run_train)
behavior_matrix_test = load_behavior_metrics(behave_path, run_test)

pca_model_train = np.load(f"empca_model_sub{sub}_ses{ses}_run{run_train}.npy", allow_pickle=True).item()


def matrices_func(beta_pca, bold_pca, behave_mat, trial_indices=None, trial_len=trial_len, num_trials=90):
    bold_pca_reshape = bold_pca.reshape(bold_pca.shape[0], num_trials, trial_len)
    trial_idx = np.arange(num_trials) if trial_indices is None else np.asarray(trial_indices, int).ravel()
    behavior_selected = behave_mat[trial_idx, 1]
    beta_selected = beta_pca[:, trial_idx]

    print("L_task...", flush=True)
    counts = np.count_nonzero(np.isfinite(beta_selected), axis=-1)
    sums = np.nansum(np.abs(beta_selected), axis=-1, dtype=np.float32)
    mean_beta = np.zeros(beta_selected.shape[0], dtype=np.float32)
    mask = counts > 0
    mean_beta[mask] = (sums[mask] / counts[mask]).astype(np.float32)

    C_task = np.zeros_like(mean_beta, dtype=np.float32)
    valid = np.abs(mean_beta) > 0
    C_task[valid] = (1.0 / mean_beta[valid]).astype(np.float32)

    print("L_var...", flush=True)
    C_bold = np.zeros((bold_pca_reshape.shape[0], bold_pca_reshape.shape[0]), dtype=np.float32)
    for i in range(num_trials - 1):
        x1 = bold_pca_reshape[:, i, :]
        x2 = bold_pca_reshape[:, i + 1, :]
        C_bold += (x1 - x2) @ (x1 - x2).T
    C_bold /= (num_trials - 1)

    print("L_var...", flush=True)
    C_beta = np.zeros((beta_selected.shape[0], beta_selected.shape[0]), dtype=np.float32)
    for i in range(num_trials - 1):
        x1 = beta_selected[:, i]
        x2 = beta_selected[:, i + 1]
        diff = x1 - x2
        C_beta += np.outer(diff, diff)
    C_beta /= (num_trials - 1)

    return C_task, C_bold, C_beta, behavior_selected, beta_selected


def prepare_run_data(run_id, bold_timeseries, anatomical_image, brain_mask_volume,
    csf_mask_volume, gray_matter_mask_volume, beta_glm_volume, filtered_beta_volume,
    behavioral_matrix, pca_model, trial_length, num_trials, selected_trial_indices=None,
    *, reuse_nan_mask=None, reuse_active_coords=None, save_nan_mask_path=None):

    print(f"Preparing data for run {run_id}...", flush=True)
    run_index = run_id

    brain_mask_data = brain_mask_volume > 0
    csf_mask_data = csf_mask_volume > 0
    gray_matter_mask = gray_matter_mask_volume > 0.5
    brain_without_csf = np.logical_and(brain_mask_data, ~csf_mask_data)
    masked_indices = np.where(brain_without_csf)

    gray_voxel_mask = gray_matter_mask[masked_indices]
    bold_flat = bold_timeseries[masked_indices]
    bold_gray = bold_flat[gray_voxel_mask]
    gray_voxel_coords = tuple(axis[gray_voxel_mask] for axis in masked_indices)
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
    beta_volume = beta_run1 if run_index == 1 else beta_run2
    beta_gray = beta_volume[gray_voxel_mask]

    skipping_preprocessing = reuse_nan_mask is not None

    if skipping_preprocessing:
        print("Reusing voxel mask from training run; skipping preprocessing.", flush=True)
        nan_mask_flat = np.asarray(reuse_nan_mask, dtype=bool).ravel()
        if reuse_active_coords is None:
            raise ValueError("reuse_active_coords must be provided when skipping preprocessing.")
        active_coords = tuple(np.array(coord) for coord in reuse_active_coords)
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

        cleaned_beta = beta_gray.copy()
        voxel_outlier_fraction = np.mean(outlier_mask, axis=1)
        valid_voxels = voxel_outlier_fraction <= 0.5
        cleaned_beta[~valid_voxels] = np.nan
        cleaned_beta[np.logical_and(outlier_mask, valid_voxels[:, None])] = np.nan
        retained_voxel_mask = ~np.all(np.isnan(cleaned_beta), axis=1)
        cleaned_beta = cleaned_beta[retained_voxel_mask]
        retained_indices = np.flatnonzero(retained_voxel_mask)

        bold_by_trial[~valid_voxels, :, :] = np.nan
        trial_outliers = np.logical_and(outlier_mask, valid_voxels[:, None])
        bold_by_trial = np.where(trial_outliers[:, :, None], np.nan, bold_by_trial)
        bold_by_trial = bold_by_trial[retained_voxel_mask]

        print("Filtering active voxels via t-test...", flush=True)
        _tvals, pvals = ttest_1samp(cleaned_beta, popmean=0, axis=1, nan_policy="omit")
        tested_mask = np.isfinite(pvals)
        alpha = 0.05
        rejected, q_values, _, _ = multipletests(pvals[tested_mask], alpha=alpha, method="fdr_bh")

        num_voxels_retained = cleaned_beta.shape[0]
        voxel_qvals = np.full(num_voxels_retained, np.nan)
        active_mask = np.zeros(num_voxels_retained, dtype=bool)
        active_mask[tested_mask] = rejected
        voxel_qvals[tested_mask] = q_values

        active_beta = cleaned_beta[active_mask]
        active_indices = retained_indices[active_mask]
        active_bold = bold_by_trial[active_mask]
        active_coords = tuple(coord[active_indices] for coord in gray_voxel_coords)

        if active_coords[0].size == 0:
            raise ValueError(f"No active voxels remained for run {run_index} after filtering.")

        beta_nan_mask = np.all(np.isnan(filtered_beta_volume), axis=-1)
        nan_mask_flat = beta_nan_mask.ravel()
        if save_nan_mask_path is not None:
            np.save(save_nan_mask_path, nan_mask_flat)

        active_flat_indices = np.ravel_multi_index(active_coords, beta_nan_mask.shape)
        keep_mask = ~nan_mask_flat[active_flat_indices]
        active_bold = active_bold[keep_mask, ...]
        active_beta = active_beta[keep_mask, ...]
        active_indices = active_indices[keep_mask]
        active_coords = tuple(coord[keep_mask] for coord in active_coords)

    print("Applying PCA projection...", flush=True)
    volume_flat = filtered_beta_volume.reshape(-1, filtered_beta_volume.shape[-1])
    beta_volume_clean = volume_flat[~nan_mask_flat]

    bold_pca_components = pca_model.eigvec
    coeff_pinv = np.linalg.pinv(pca_model.coeff)
    beta_pca = coeff_pinv @ np.nan_to_num(beta_volume_clean)

    (task_penalty_vector, bold_penalty_matrix, beta_penalty_matrix, behavior_vector, beta_matrix,) = matrices_func( beta_pca, bold_pca_components, behavioral_matrix, 
                                                    trial_indices=selected_trial_indices, trial_len=trial_length, num_trials=num_trials)

    trial_mask = np.isfinite(behavior_vector)
    behavior_observed = behavior_vector[trial_mask]
    behavior_mean = np.mean(behavior_observed)
    behavior_centered = behavior_observed - behavior_mean
    behavior_norm = np.linalg.norm(behavior_centered)
    if behavior_norm == 0:
        normalized_behaviors = np.zeros_like(behavior_centered)
    else:
        normalized_behaviors = behavior_centered / behavior_norm

    beta_observed = beta_matrix[:, trial_mask]
    beta_centered = beta_observed - np.mean(beta_observed, axis=1, keepdims=True)

    if np.max(task_penalty_vector) > 0:
        task_penalty_vector = task_penalty_vector / np.max(task_penalty_vector)
    if np.max(bold_penalty_matrix) > 0:
        bold_penalty_matrix = bold_penalty_matrix / np.max(bold_penalty_matrix)
    if np.max(beta_penalty_matrix) > 0:
        beta_penalty_matrix = beta_penalty_matrix / np.max(beta_penalty_matrix)

    return {
        "run_id": run_index,
        "beta_centered": beta_centered.astype(np.float64, copy=False),
        "normalized_behaviors": normalized_behaviors.astype(np.float64, copy=False),
        "behavior_centered": behavior_centered.astype(np.float64, copy=False),
        "behavior_observed": behavior_observed.astype(np.float64, copy=False),
        "behavior_vector": behavior_vector,
        "behavior_mean": behavior_mean,
        "behavior_norm": behavior_norm,
        "beta_observed": beta_observed.astype(np.float64, copy=False),
        "task_penalty_vector": task_penalty_vector.astype(np.float64, copy=False),
        "bold_penalty_matrix": bold_penalty_matrix.astype(np.float64, copy=False),
        "beta_penalty_matrix": beta_penalty_matrix.astype(np.float64, copy=False),
        "coeff_pinv": coeff_pinv,
        "active_coords": tuple(np.array(coord) for coord in active_coords),
        "trial_mask": trial_mask,
        "nan_mask_flat": nan_mask_flat}


def build_penalty_matrices(run_data, task_penalty, bold_penalty, beta_penalty, ridge_penalty):
    beta_centered = run_data["beta_centered"]
    n_components = beta_centered.shape[0]

    task_matrix = task_penalty * np.diag(run_data["task_penalty_vector"])
    bold_matrix = bold_penalty * run_data["bold_penalty_matrix"]
    beta_matrix = beta_penalty * run_data["beta_penalty_matrix"]

    base_penalty = task_matrix + bold_matrix + beta_matrix
    base_penalty = 0.5 * (base_penalty + base_penalty.T)
    total_penalty = base_penalty + ridge_penalty * np.eye(n_components)

    return {"task": task_matrix, "bold": bold_matrix, "beta": beta_matrix}, total_penalty


def solve_soc_problem(run_data, task_penalty, bold_penalty, beta_penalty, solver_name, soc_ratio):
    beta_centered = run_data["beta_centered"]
    normalized_behaviors = run_data["normalized_behaviors"]
    n_components = beta_centered.shape[0]

    penalty_matrices, total_penalty = build_penalty_matrices(run_data, task_penalty, bold_penalty, beta_penalty, ridge_penalty)

    weights_positive = cp.Variable(n_components)
    weights_negative = cp.Variable(n_components)
    projections_positive = beta_centered.T @ weights_positive
    projections_negative = beta_centered.T @ weights_negative

    constraints_positive = [weights_positive >= 0, cp.sum(weights_positive) == 1,
        cp.SOC((1.0 / soc_ratio) * (normalized_behaviors @ projections_positive), projections_positive)]
    problem_positive = cp.Problem(cp.Minimize(cp.quad_form(weights_positive, cp.psd_wrap(total_penalty))), constraints_positive)

    constraints_negative = [weights_negative >= 0, cp.sum(weights_negative) == 1,
        cp.SOC((1.0 / soc_ratio) * (-normalized_behaviors @ projections_negative), projections_negative)]
    problem_negative = cp.Problem(cp.Minimize(cp.quad_form(weights_negative, cp.psd_wrap(total_penalty))), constraints_negative)

    objective_positive = np.inf
    objective_negative = np.inf

    try:
        problem_positive.solve(solver=solver_name, warm_start=True)
    except cp.error.SolverError as exc:
        print(f"Positive branch solver failed: {exc}", flush=True)
    if weights_positive.value is not None and problem_positive.status in ("optimal", "optimal_inaccurate"):
        objective_positive = problem_positive.value

    try:
        problem_negative.solve(solver=solver_name, warm_start=True)
    except cp.error.SolverError as exc:
        print(f"Negative branch solver failed: {exc}", flush=True)
    if weights_negative.value is not None and problem_negative.status in ("optimal", "optimal_inaccurate"):
        objective_negative = problem_negative.value

    if np.isfinite(objective_positive) and (not np.isfinite(objective_negative) or objective_positive <= objective_negative):
        solution_weights = np.array(weights_positive.value).ravel()
        solution_branch = "positive"
    else:
        solution_weights = np.array(weights_negative.value).ravel()
        solution_branch = "negative"

    contributions = {label: float(solution_weights.T @ matrix @ solution_weights) for label, matrix in penalty_matrices.items()}

    projection = beta_centered.T @ solution_weights
    total_loss = float(solution_weights.T @ total_penalty @ solution_weights)

    return {
        "weights": solution_weights,
        "branch": solution_branch,
        "objective_positive": objective_positive,
        "objective_negative": objective_negative,
        "penalty_contributions": contributions,
        "total_loss": total_loss,
        "projection": projection}


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

    if projection.size > 1:
        corr = np.corrcoef(projection, behavior_centered)[0, 1]
        metrics["pearson"] = float(corr)

    if projection.size > 0:
        residuals = behavior_centered - projection
        metrics["mse"] = float(np.mean(residuals**2))
        sst = np.sum(behavior_centered**2)
        if sst > 0:
            metrics["r2"] = float(1 - np.sum(residuals**2) / sst)

    return metrics


def resample_run_data(run_data, sample_indices):
    beta_observed = np.asarray(run_data["beta_observed"])
    behavior_observed = np.asarray(run_data["behavior_observed"])

    beta_sample = beta_observed[:, sample_indices]
    behavior_sample = behavior_observed[sample_indices]

    behavior_mean = np.mean(behavior_sample)
    behavior_centered = behavior_sample - behavior_mean
    behavior_norm = np.linalg.norm(behavior_centered)
    if behavior_norm == 0:
        normalized_behaviors = np.zeros_like(behavior_centered)
    else:
        normalized_behaviors = behavior_centered / behavior_norm

    beta_centered = beta_sample - np.mean(beta_sample, axis=1, keepdims=True)

    resampled_data = run_data.copy()
    resampled_data.update({
        "beta_centered": beta_centered.astype(np.float64, copy=False),
        "normalized_behaviors": normalized_behaviors.astype(np.float64, copy=False),
        "behavior_centered": behavior_centered.astype(np.float64, copy=False),
        "behavior_observed": behavior_sample.astype(np.float64, copy=False),
        "behavior_mean": float(behavior_mean),
        "behavior_norm": float(behavior_norm)})
    return resampled_data


def summarize_bootstrap(values, *, reference=0.0, two_sided=True):
    values_array = np.asarray(values, dtype=np.float64)
    finite_mask = np.isfinite(values_array)
    values_array = values_array[finite_mask]

    summary = {"mean": np.nan, "std": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "p_value": np.nan, "num_samples": int(values_array.size)}

    summary["mean"] = float(np.mean(values_array))
    summary["std"] = float(np.std(values_array, ddof=1)) if values_array.size > 1 else 0.0
    summary["ci_lower"], summary["ci_upper"] = (float(x) for x in np.percentile(values_array, [2.5, 97.5]))

    if two_sided:
        lower_tail = (np.count_nonzero(values_array <= reference) + 1) / (values_array.size + 1)
        upper_tail = (np.count_nonzero(values_array >= reference) + 1) / (values_array.size + 1)
        p_value = min(1.0, 2.0 * min(lower_tail, upper_tail))
    else:
        p_value = (np.count_nonzero(values_array <= reference) + 1) / (values_array.size + 1)

    summary["p_value"] = float(p_value)
    return summary


def perform_weight_shuffle_tests(train_data, test_data, weights, task_alpha, bold_alpha, beta_alpha, rho_value, *, num_samples):
    
    weights = np.asarray(weights, dtype=np.float64).ravel()
    rng = np.random.default_rng()

    penalty_matrices, total_penalty = build_penalty_matrices(train_data, task_alpha, bold_alpha, beta_alpha, ridge_penalty)

    baseline_penalties = {label: float(weights.T @ matrix @ weights) for label, matrix in penalty_matrices.items()}
    baseline_loss = float(weights.T @ total_penalty @ weights)

    penalty_samples = {label: [] for label in penalty_matrices}
    total_loss_samples = []
    test_corr_samples = []
    train_corr_samples = []

    for iteration in range(num_samples):
        shuffled_weights = rng.permutation(weights)

        penalty_values = {}
        for label, matrix in penalty_matrices.items():
            penalty_value = float(shuffled_weights.T @ matrix @ shuffled_weights)
            penalty_samples[label].append(penalty_value)
            penalty_values[label] = penalty_value

        total_loss = float(shuffled_weights.T @ total_penalty @ shuffled_weights)
        total_loss_samples.append(total_loss)

        train_metrics = evaluate_projection(train_data, shuffled_weights)
        test_metrics = evaluate_projection(test_data, shuffled_weights)
        train_corr_samples.append(train_metrics["pearson"])
        test_corr_samples.append(test_metrics["pearson"])

    penalty_arrays = {label: np.asarray(values, dtype=np.float64) for label, values in penalty_samples.items()}
    total_loss_array = np.asarray(total_loss_samples, dtype=np.float64)
    train_corr_array = np.asarray(train_corr_samples, dtype=np.float64)
    test_corr_array = np.asarray(test_corr_samples, dtype=np.float64)

    penalty_summary = {label: summarize_bootstrap(values, reference=baseline_penalties.get(label, 0.0), two_sided=False) for label, values in penalty_arrays.items()}
    total_loss_summary = summarize_bootstrap(total_loss_array, reference=baseline_loss, two_sided=True)
    correlation_summary = {"train": summarize_bootstrap(train_corr_array, reference=0.0, two_sided=True),
                           "test": summarize_bootstrap(test_corr_array, reference=0.0, two_sided=True)}

    num_successful = int(np.count_nonzero(np.isfinite(test_corr_array)))
    num_failed = int(num_samples - num_successful)

    return {
        "penalties": penalty_summary,
        "total_loss": total_loss_summary,
        "correlation": correlation_summary,
        "num_requested": num_samples,
        "num_successful": num_successful,
        "num_failed": num_failed,
        "penalty_samples": penalty_arrays,
        "total_loss_samples": total_loss_array,
        "correlation_samples": {"train": train_corr_array, "test": test_corr_array}}


def plot_bootstrap_distribution(samples, actual_value, p_value, title, xlabel, output_path):
    samples = np.asarray(samples, dtype=np.float64)
    finite_samples = samples[np.isfinite(samples)]

    num_bins = max(10, min(50, int(np.sqrt(finite_samples.size))))
    sample_min = float(np.min(finite_samples))
    sample_max = float(np.max(finite_samples))
    lower_edge = float(min(sample_min, actual_value)) if np.isfinite(actual_value) else sample_min
    upper_edge = float(max(sample_max, actual_value)) if np.isfinite(actual_value) else sample_max
    span_margin = 0.05 * max(sample_max - sample_min, 1e-9)
    range_lower = max(0.0, lower_edge - span_margin)
    range_upper = upper_edge + span_margin

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(finite_samples, bins=num_bins, color="#4c72b0", alpha=0.75, edgecolor="black", range=(range_lower, range_upper))

    # ax.axvline(0.0, color="#55a868", linestyle=":", linewidth=1.5, label="Null (0)")

    annotation_parts = []

    if np.isfinite(actual_value):
        ax.axvline(actual_value, color="#c44e52", linestyle="--", linewidth=2.0, label="Actual")
        annotation_parts.append(f"actual={actual_value:.4g}")
    else:
        annotation_parts.append("actual=nan")

    if p_value is not None and np.isfinite(p_value):
        annotation_parts.append(f"p={p_value:.4g}")
    else:
        annotation_parts.append("p=nan")

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

def save_weight_outputs(anatomical_image, active_coords, pca_weights, file_prefix):
    coords = tuple(np.asarray(coord) for coord in active_coords)
    weight_volume = np.full(anatomical_image.shape, np.nan, dtype=np.float32)
    scaled_weights = np.abs(pca_weights) * 1e6
    weight_volume[coords] = scaled_weights

    finite_scaled = scaled_weights[np.isfinite(scaled_weights)]
    if finite_scaled.size > 0:
        vmax = float(np.nanpercentile(finite_scaled, 95))
    else:
        vmax = 0.0
    vmin = 0.0
    if vmax <= vmin:
        vmax = vmin + 1.0

    clipped_volume = np.clip(weight_volume, vmin, vmax)
    weight_img = nib.Nifti1Image(clipped_volume, anatomical_image.affine, anatomical_image.header)
    view = plotting.view_img(weight_img, bg_img=anatomical_image, cmap="jet", symmetric_cmap=False)
    view_path = f"W_prime_overlay_{file_prefix}.html"
    view.save_as_html(view_path)
    np.save(f"W_prime_volume_{file_prefix}_corrected.npy", clipped_volume)
    np.save(f"W_prime_volume_{file_prefix}_raw.npy", finite_scaled)
    return view_path


ridge_penalty = 1e-6
solver_name = "MOSEK"
soc_ratio = 0.95

alpha_sweep = [{"task_penalty": 1, "bold_penalty": 1, "beta_penalty": 1}]
rho_sweep = [0.2, soc_ratio]
bootstrap_iterations = 1000
bootstrap_random_seed = 13


def run_cross_run_experiment(alpha_settings, rho_values, selected_trials=None,
                             *, bootstrap_samples=0, bootstrap_random_state=None):
    shared_nan_mask_path = f"mask_all_nan_sub{sub}_ses{ses}_run{run_train}_reference.npy"
    train_data = prepare_run_data(run_train, bold_data_train, anat_img, 
                                  brain_mask, csf_mask, gray_mask, beta_glm, beta_volume_filter_train, behavior_matrix_train,
                                  pca_model_train, trial_len, num_trials, selected_trial_indices=selected_trials,
                                  save_nan_mask_path=shared_nan_mask_path)

    test_data = prepare_run_data(run_test, bold_data_test, anat_img, 
                                 brain_mask, csf_mask, gray_mask, beta_glm,
                                 beta_volume_filter_test, behavior_matrix_test, 
                                 pca_model_train, trial_len, num_trials, selected_trial_indices=selected_trials,
                                 reuse_nan_mask=train_data["nan_mask_flat"],
                                 reuse_active_coords=train_data["active_coords"])

    for alpha_setting in alpha_settings:
        task_alpha = alpha_setting["task_penalty"]
        bold_alpha = alpha_setting["bold_penalty"]
        beta_alpha = alpha_setting["beta_penalty"]

        for rho_value in rho_values:
            sweep_suffix = f"task{task_alpha:g}_bold{bold_alpha:g}_beta{beta_alpha:g}_rho{rho_value:g}"
            print(f"Cross-run optimization with task={task_alpha}, bold={bold_alpha}, beta={beta_alpha}, rho={rho_value}", flush=True)

            solution = solve_soc_problem(train_data, task_alpha, bold_alpha, beta_alpha, solver_name, rho_value)

            weights = solution["weights"]
            coeff_pinv = np.asarray(train_data["coeff_pinv"])
            pca_weights = coeff_pinv.T @ weights

            file_prefix = f"sub{sub}_ses{ses}_trainrun{run_train}_testrun{run_test}_{sweep_suffix}"
            # np.save(f"behavior_weights_{file_prefix}.npy", pca_weights)
            # np.save(f"behavior_weights_pca_{file_prefix}.npy", weights)
            # view_path = save_weight_outputs(anat_img, train_data["active_coords"], pca_weights, file_prefix)

            train_metrics = evaluate_projection(train_data, weights)
            test_metrics = evaluate_projection(test_data, weights)

            print(f"  Solution branch: {solution['branch']}", flush=True)
            print(f"  Penalty contributions: {solution['penalty_contributions']}", flush=True)
            # print(f"  Ridge contribution: {solution['ridge_contribution']:.6f}", flush=True)
            print(f"  Total loss (train objective): {solution['total_loss']:.6f}", flush=True)
            print(
                f"  Train metrics -> corr: {train_metrics['pearson']:.4f}, R2: {train_metrics['r2']:.4f}, "
                f"MSE: {train_metrics['mse']:.6f}, slack: {train_metrics['soc_slack']:.6f}", flush=True)
            print(
                f"  Test metrics  -> corr: {test_metrics['pearson']:.4f}, R2: {test_metrics['r2']:.4f}, "
                f"MSE: {test_metrics['mse']:.6f}, slack: {test_metrics['soc_slack']:.6f}", flush=True)
            
            # print(f"  Saved weights to behavior_weights_{file_prefix}.npy", flush=True)
            # print(f"  Saved PCA weights to behavior_weights_pca_{file_prefix}.npy", flush=True)
            # print(f"  HTML visualization written to {view_path}", flush=True)

            if bootstrap_samples > 0:
                print(f"  Running weight-shuffle analysis with {bootstrap_samples} samples...", flush=True)
                shuffle_results = perform_weight_shuffle_tests(train_data, test_data, weights, task_alpha, bold_alpha, beta_alpha, rho_value, num_samples=bootstrap_samples)

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
                    if created_path:
                        print(f"    Saved {label} penalty bootstrap plot to {created_path}", flush=True)

                correlation_actuals = {
                    "train": float(train_metrics["pearson"]) if np.isfinite(train_metrics["pearson"]) else np.nan,
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
    run_cross_run_experiment(
        alpha_sweep,
        rho_sweep,
        bootstrap_samples=bootstrap_iterations,
        bootstrap_random_state=bootstrap_random_seed)
