# %%
import nibabel as nib
import numpy as np
from os.path import join
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
import scipy.sparse as sp
from scipy.io import loadmat
from empca.empca.empca import empca
import cvxpy as cp
from scipy import ndimage
from nilearn import plotting

# %%
ses = 1
sub = '04'
run = 1
num_trials = 90
trial_len = 9

base_path = '/scratch/st-mmckeown-1/zkavian/fmri_models'
anat_img = nib.load(f'/scratch/st-mmckeown-1/zkavian/fmri_models/sub-pd0{sub}_ses-{ses}_T1w_brain.nii.gz')
bold_data = np.load(join(base_path, 'bold_data.npy'))

mask_path = f'/scratch/st-mmckeown-1/zkavian/fmri_models/sub-pd0{sub}_ses-{ses}_T1w_brain_mask.nii.gz'
brain_mask = nib.load(mask_path)
brain_mask = brain_mask.get_fdata()
brain_mask = brain_mask.astype(np.float16)

mask_path = f'/scratch/st-mmckeown-1/zkavian/fmri_models/sub-pd0{sub}_ses-{ses}_T1w_brain_pve_0.nii.gz'
csf_mask = nib.load(mask_path)
csf_mask = csf_mask.get_fdata()
csf_mask = csf_mask.astype(np.float16)

mask_path = f'/scratch/st-mmckeown-1/zkavian/fmri_models/sub-pd0{sub}_ses-{ses}_T1w_brain_pve_1.nii.gz'
gray_mask = nib.load(mask_path)
gray_mask = gray_mask.get_fdata()
gray_mask = gray_mask.astype(np.float16)

glm_dict = np.load(f'TYPED_FITHRF_GLMDENOISE_RR_sub{sub}.npy', allow_pickle=True).item()
beta_glm = glm_dict['betasmd']

beta_volume_filter = np.load(f'cleaned_beta_volume_sub{sub}_ses{ses}_run{run}.npy')

def load_behavior_metrics(path, run):
    behav_data = loadmat(path)
    behav_metrics = behav_data["behav_metrics"]
    behav_block = np.stack(behav_metrics[0], axis=0)
    _, _, num_metrics = behav_block.shape
    behav_flat = behav_block.reshape(-1, num_metrics)
    if run == 1:
        behav_flat = behav_flat[:90, :6]
    else:
        behav_flat = behav_flat[90:180, :6]
    return behav_flat

behave_path = 'PSPD004_OFF_behav_metrics.mat'
behavior_matrix = load_behavior_metrics(behave_path, run)

pca_model = np.load('empca_model_sub04_ses1_run1.npy', allow_pickle=True).item()

print(1, flush=True)



def matrices_func(beta_pca, bold_pca, behave_mat, trial_indices=None, trial_len=trial_len, num_trials=90):
    # reshape beta_pca, bold_pca, behave_pca if it is necessary
    bold_pca_reshape = bold_pca.reshape(bold_pca.shape[0], num_trials, trial_len)

    trial_idx = np.arange(num_trials) if trial_indices is None else np.unique(np.asarray(trial_indices, int).ravel())

    # select trials
    behavior_selected = behave_mat[trial_idx, 1]  #1/RT
    beta_selected = beta_pca[:, trial_idx]

    # ----- L_task (same idea as yours) -----
    print("L_task...", flush=True)
    counts = np.count_nonzero(np.isfinite(beta_selected), axis=-1)
    sums = np.nansum(np.abs(beta_selected), axis=-1, dtype=np.float32)
    mean_beta = np.zeros(beta_selected.shape[0], dtype=np.float32)
    m = counts > 0
    mean_beta[m] = (sums[m] / counts[m]).astype(np.float32)

    C_task = np.zeros_like(mean_beta, dtype=np.float32)
    v = np.abs(mean_beta) > 0 # avoid division by zero
    C_task[v] = (1.0 / mean_beta[v]).astype(np.float32)

    # ----- L_var_bold: variance of trial differences, as sparse diagonal -----
    print("L_var...", flush=True)
    C_bold = np.zeros((bold_pca_reshape.shape[0], bold_pca_reshape.shape[0]), dtype=np.float32)

    for i in range(num_trials-1):
        x1 = bold_pca_reshape[:, i, :]
        x2 = bold_pca_reshape[:, i+1, :]
        C_bold += (x1-x2) @ (x1-x2).T
    C_bold /= (num_trials - 1)

    # ----- L_var_beta: variance of trial differences, as sparse diagonal -----
    print("L_var...", flush=True)
    C_beta = np.zeros((beta_selected.shape[0], beta_selected.shape[0]), dtype=np.float32)
    for i in range(num_trials-1):
        x1 = beta_selected[:, i]
        x2 = beta_selected[:, i+1]
        diff = x1 - x2
        C_beta += np.outer(diff, diff)  
    C_beta /= (num_trials - 1)

    return C_task, C_bold, C_beta, behavior_selected, beta_selected


# %%
def main_func(
    bold_timeseries,
    anatomical_image,
    brain_mask_volume,
    csf_mask_volume,
    gray_matter_mask_volume,
    beta_glm_volume,
    filtered_beta_volume,
    behavioral_matrix,
    pca_model,
    bold_penalty,
    beta_penalty,
    task_penalty,
    ridge_penalty,
    solver_name,
    soc_ratio,
    trial_length,
    num_trials,
    selected_trial_indices=None,
    run_id=None,
    result_suffix="",
):
    """Solve SOC problems for the ablation study given BOLD, beta, and behavior inputs."""
    run_index = run if run_id is None else run_id

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
            start_idx += 20  # Skip discarded timepoints

    print("Masking GLM betas...", flush=True)
    beta_run1, beta_run2 = beta_glm_volume[:, 0, 0, :90], beta_glm_volume[:, 0, 0, 90:]
    beta_volume = beta_run1 if run_index == 1 else beta_run2
    beta_gray = beta_volume[gray_voxel_mask]

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

    active_volume = np.full(bold_timeseries.shape[:3] + (num_trials,), np.nan)
    active_coords = tuple(coord[active_indices] for coord in gray_voxel_coords)
    active_volume[active_coords[0], active_coords[1], active_coords[2], :] = active_beta

    print("Applying PCA projection...", flush=True)
    beta_nan_mask = np.all(np.isnan(filtered_beta_volume), axis=-1)
    nan_mask_flat = np.load(f"mask_all_nan_sub{sub}_ses{ses}_run{run_index}.npy")
    beta_volume_clean = filtered_beta_volume[~beta_nan_mask]

    active_flat_indices = np.ravel_multi_index(active_coords, beta_nan_mask.shape)
    keep_mask = ~nan_mask_flat[active_flat_indices]
    active_bold = active_bold[keep_mask, ...]
    active_beta = active_beta[keep_mask, ...]
    active_indices = active_indices[keep_mask]
    active_coords = tuple(coord[keep_mask] for coord in active_coords)

    bold_pca_components = pca_model.eigvec
    coeff_pinv = np.linalg.pinv(pca_model.coeff)
    print(f"PCA shapes -> coeff_pinv: {coeff_pinv.shape}, beta_volume_clean: {beta_volume_clean.shape}",flush=True,)
    beta_pca = coeff_pinv @ np.nan_to_num(beta_volume_clean)

    print("Building penalty matrices...", flush=True)
    (
        task_penalty_vector,
        bold_penalty_matrix,
        beta_penalty_matrix,
        behavior_vector,
        beta_matrix,
    ) = matrices_func(
        beta_pca,
        bold_pca_components,
        behavioral_matrix,
        trial_indices=selected_trial_indices,
        trial_len=trial_length,
        num_trials=num_trials,
    )
    suffix = f"_{result_suffix}" if result_suffix else ""
    np.save(f"C_task_sub{sub}_ses{ses}_run{run_index}{suffix}.npy", task_penalty_vector)
    np.save(f"C_bold_sub{sub}_ses{ses}_run{run_index}{suffix}.npy", bold_penalty_matrix)
    np.save(f"C_beta_sub{sub}_ses{ses}_run{run_index}{suffix}.npy", beta_penalty_matrix)

    trial_mask = np.isfinite(behavior_vector)
    behavior_observed = behavior_vector[trial_mask]
    behavior_mean = np.mean(behavior_observed)
    behavior_centered = behavior_observed - behavior_mean
    behavior_norm = np.linalg.norm(behavior_centered)
    normalized_behaviors = behavior_centered / behavior_norm

    beta_observed = beta_matrix[:, trial_mask]
    beta_centered = beta_observed - np.mean(beta_observed, axis=1, keepdims=True)

    task_penalty_vector = task_penalty_vector / np.max(task_penalty_vector)
    bold_penalty_matrix = bold_penalty_matrix / np.max(bold_penalty_matrix)
    beta_penalty_matrix = beta_penalty_matrix / np.max(beta_penalty_matrix)

    n_components = beta_centered.shape[0]
    total_penalty = (
        task_penalty * np.diag(task_penalty_vector)
        + bold_penalty * bold_penalty_matrix
        + beta_penalty * beta_penalty_matrix
    )
    total_penalty = 0.5 * (total_penalty + total_penalty.T)
    total_penalty += ridge_penalty * np.eye(n_components)
    print("Penalty matrix range:", total_penalty.min(), total_penalty.max(), flush=True)

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
    objective_positive = objective_negative = np.inf

    print("Solving SOC optimization problems...", flush=True)
    problem_positive.solve(solver=solver_name, warm_start=True)
    proj = beta_centered.T @ weights_negative.value
    lhs = np.linalg.norm(proj)
    rhs = (1.0 / soc_ratio) * float(normalized_behaviors @ proj)
    print("SOC slack:", rhs - lhs)

    if weights_positive.value is not None and problem_positive.status in ("optimal", "optimal_inaccurate"):
        objective_positive = problem_positive.value

    print("Solving SOC optimization problems (negative branch)...", flush=True)
    problem_negative.solve(solver=solver_name, warm_start=True)
    proj = beta_centered.T @ weights_positive.value
    lhs = np.linalg.norm(proj)
    rhs = (1.0 / soc_ratio) * float(normalized_behaviors @ proj)
    print("SOC slack:", rhs - lhs)
    if weights_negative.value is not None and problem_negative.status in ("optimal", "optimal_inaccurate"):
        objective_negative = problem_negative.value

    if not np.isfinite(objective_positive) and not np.isfinite(objective_negative):
        raise RuntimeError("Both SOC branches were infeasible or failed to solve.")

    solution_weights = (np.array(weights_positive.value).ravel()
        if objective_positive <= objective_negative else np.array(weights_negative.value).ravel())

    penalty_matrices = {
        "task": task_penalty * np.diag(task_penalty_vector),
        "bold": bold_penalty * bold_penalty_matrix,
        "beta": beta_penalty * beta_penalty_matrix}

    for label, weight_var in (("positive", weights_positive), ("negative", weights_negative)):
        weights_value = weight_var.value
        if weights_value is None:
            print(f"{label.title()} branch did not converge.", flush=True)
            continue

        print(f"{label.title()} branch contributions:", flush=True)
        for penalty_label, penalty_matrix in penalty_matrices.items():
            contribution = float(weights_value.T @ penalty_matrix @ weights_value)
            print(f"  {penalty_label}: {contribution}", flush=True)

    pca_weights = coeff_pinv.T @ solution_weights
    np.save(f"behavior_weights_sub{sub}_ses{ses}_run{run}{suffix}.npy", pca_weights)
    np.save(f"behavior_weights_pca_sub{sub}_ses{ses}_run{run}{suffix}.npy", solution_weights)

    # Reconstruct a volumetric map of W_prime for visualization on the anatomical image
    final_coords = active_coords
    weight_volume = np.full(anatomical_image.shape, np.nan, dtype=np.float32)
    weight_volume[final_coords] = np.abs(pca_weights) * 1e6
    vmin = 0
    vmax = np.nanpercentile(np.abs(pca_weights * 1e6), 95)
    print(vmin, vmax)
    weight_volume = np.clip(weight_volume, vmin, vmax)
    weight_img = nib.Nifti1Image(weight_volume, anatomical_image.affine, anatomical_image.header)

    view = plotting.view_img(
        weight_img,
        bg_img=anatomical_image,
        cmap="jet",
        symmetric_cmap=False,
    )
    view_path = f"W_prime_overlay_sub{sub}_ses{ses}_run{run}{suffix}.html"
    view.save_as_html(view_path)
    print(f"Saved visualization to {view_path}", flush=True)
    print("Optimization finished.", flush=True)






bold_penalty = 1.0
beta_penalty = 1.0
task_penalty = 1.0
ridge_penalty = 1e-6
solver_name = "MOSEK"
soc_ratio = 0.95

alpha_sweep = [
    {"task_penalty": 0, "bold_penalty": 0, "beta_penalty": 0.5},
    {"task_penalty": 0, "bold_penalty": 0.5, "beta_penalty": 0},
    {"task_penalty": 0.5, "bold_penalty": 0, "beta_penalty": 0.5},
    {"task_penalty": 0.5, "bold_penalty": 1, "beta_penalty": 10},
    {"task_penalty": 1, "bold_penalty": 0.5, "beta_penalty": 10},
    {"task_penalty": 0.5, "bold_penalty": 10, "beta_penalty": 1},
    {"task_penalty": 1, "bold_penalty": 10, "beta_penalty": 0.5},
    {"task_penalty": 10, "bold_penalty": 0.5, "beta_penalty": 1},
    {"task_penalty": 10, "bold_penalty": 1, "beta_penalty": 0.5}]

rho_sweep = [0.05, soc_ratio]


def run_parameter_sweep(alpha_settings, rho_values, selected_trials=None):
    """Run the ablation optimization across multiple alpha and rho settings."""
    for alpha_setting in alpha_settings:
        task_alpha = alpha_setting["task_penalty"]
        bold_alpha = alpha_setting["bold_penalty"]
        beta_alpha = alpha_setting["beta_penalty"]

        for rho_value in rho_values:
            sweep_suffix = (
                f"task{task_alpha:g}_bold{bold_alpha:g}_beta{beta_alpha:g}_rho{rho_value:g}")
            print(
                f"Running ablation sweep with task={task_alpha}, bold={bold_alpha}, "
                f"beta={beta_alpha}, rho={rho_value}", flush=True)
            
            main_func(
                bold_timeseries=bold_data,
                anatomical_image=anat_img,
                brain_mask_volume=brain_mask,
                csf_mask_volume=csf_mask,
                gray_matter_mask_volume=gray_mask,
                beta_glm_volume=beta_glm,
                filtered_beta_volume=beta_volume_filter,
                behavioral_matrix=behavior_matrix,
                pca_model=pca_model,
                bold_penalty=bold_alpha,
                beta_penalty=beta_alpha,
                task_penalty=task_alpha,
                ridge_penalty=ridge_penalty,
                solver_name=solver_name,
                soc_ratio=rho_value,
                trial_length=trial_len,
                num_trials=num_trials,
                selected_trial_indices=selected_trials,
                run_id=run,
                result_suffix=sweep_suffix,
            )


if __name__ == "__main__":
    run_parameter_sweep(alpha_sweep, rho_sweep)
