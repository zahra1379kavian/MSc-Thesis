# %%
import nibabel as nib
import numpy as np
from os.path import join
# import math
# import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
from nilearn import plotting
# from nilearn.image import resample_to_img
# import cvxpy as cp
# from sklearn.model_selection import KFold
# from itertools import product
# import scipy.io as sio
# import h5py
# from sklearn.decomposition import PCA
import scipy.sparse as sp
# import matplotlib.ticker as mticker
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
# from scipy.stats import zscore
from scipy.io import loadmat
from empca.empca.empca import empca
import cvxpy as cp
from scipy import ndimage
# from typing import Optional, Sequence, Tuple, Dict, Any

# %%
ses = 1
sub = '04'
run = 2
num_trials = 90
trial_len = 9

base_path = '/scratch/st-mmckeown-1/zkavian/fmri_models'
anat_img = nib.load(f'/scratch/st-mmckeown-1/zkavian/fmri_models/sub-pd0{sub}_ses-{ses}_T1w_brain.nii.gz')
bold_data = np.load(join(base_path, f'fmri_sub{sub}_ses{ses}_run{run}.npy'), allow_pickle=True)

mask_path = f'/scratch/st-mmckeown-1/zkavian/fmri_models/sub-pd0{sub}_ses-{ses}_T1w_brain_mask.nii.gz'
back_mask = nib.load(mask_path)
back_mask = back_mask.get_fdata()
back_mask = back_mask.astype(np.float16)

mask_path = f'/scratch/st-mmckeown-1/zkavian/fmri_models/sub-pd0{sub}_ses-{ses}_T1w_brain_pve_0.nii.gz'
csf_mask = nib.load(mask_path)
csf_mask = csf_mask.get_fdata()
csf_mask = csf_mask.astype(np.float16)

mask_path = f'/scratch/st-mmckeown-1/zkavian/fmri_models/sub-pd0{sub}_ses-{ses}_T1w_brain_pve_1.nii.gz'
gray_mask = nib.load(mask_path)
gray_mask = gray_mask.get_fdata()
gray_mask = gray_mask.astype(np.float16)

print(1, flush=True)
# %% [markdown]
# Apply Masks on Bold Dataset

# %%
back_mask_data = back_mask > 0
csf_mask_data = csf_mask > 0
gray_mask_data = gray_mask > 0.5
mask = np.logical_and(back_mask_data, ~csf_mask_data)
nonzero_mask = np.where(mask)

keep_voxels = gray_mask_data[nonzero_mask]

bold_flat = bold_data[nonzero_mask]
masked_bold = bold_flat[keep_voxels]
masked_coords = tuple(ax[keep_voxels] for ax in nonzero_mask)

# %% [markdown]
# Remove voxels in CSF & brain mask from the bold data

# %%

masked_bold = masked_bold.astype(np.float32)
num_voxels, num_timepoints = masked_bold.shape
bold_data_reshape = np.full((num_voxels, num_trials, trial_len), np.nan, dtype=np.float32)

start = 0
for i in range(num_trials):
    end = start + trial_len
    if end > num_timepoints:
        raise ValueError("Masked BOLD data does not contain enough timepoints for all trials")
    bold_data_reshape[:, i, :] = masked_bold[:, start:end]
    start += trial_len
    if start in (270, 560):
        start += 20  # skip discarded timepoints

print(2, flush=True)
# %% [markdown]
# Load Beta values

# %% [markdown]
# Load the filtered beta values

# %%
glm_dict = np.load(f'TYPED_FITHRF_GLMDENOISE_RR_sub{sub}.npy', allow_pickle=True).item()
beta_glm = glm_dict['betasmd']
beta_run1, beta_run2 = beta_glm[:,0,0,:90], beta_glm[:,0,0,90:]

if run == 1:
    beta = beta_run1[keep_voxels]
else:
    beta = beta_run2[keep_voxels]

nan_voxels = np.isnan(beta).all(axis=1)
if np.any(nan_voxels):
    beta = beta[~nan_voxels]
    bold_data_reshape = bold_data_reshape[~nan_voxels]
    masked_coords = tuple(coord[~nan_voxels] for coord in masked_coords)

print(3, flush=True)
# # remove last trial which have most of NaN values. 
# bold_data_reshape = bold_data_reshape[:,:-1,:]
# beta = beta[:,:-1]

# %% [markdown]
# Load the filtered beta values

# %% [markdown]
# Load the filtered beta values

# %% [markdown]
# Normalize beta value / NaN trials with extremely different beta values

# %%
med = np.nanmedian(beta, keepdims=True)
mad = np.nanmedian(np.abs(beta - med), keepdims=True)
scale = 1.4826 * np.maximum(mad, 1e-9)    
beta_norm = (beta - med) / scale      
thr = np.nanpercentile(np.abs(beta_norm), 99.9)
outlier_mask = np.abs(beta_norm) > thr  


clean_beta = beta.copy()
voxel_outlier_fraction = np.mean(outlier_mask, axis=1)
valid_voxels = voxel_outlier_fraction <= 0.5
clean_beta[~valid_voxels] = np.nan
clean_beta[np.logical_and(outlier_mask, valid_voxels[:, None])] = np.nan
keeped_mask = ~np.all(np.isnan(clean_beta), axis=1)
clean_beta = clean_beta[keeped_mask]
keeped_indices = np.flatnonzero(keeped_mask)

bold_data_reshape[~valid_voxels, :, :] = np.nan
trial_outliers = np.logical_and(outlier_mask, valid_voxels[:, None])
bold_data_reshape = np.where(trial_outliers[:, :, None], np.nan, bold_data_reshape)
bold_data_reshape = bold_data_reshape[keeped_mask]

print(4, flush=True)
# %% [markdown]
# Apply t-test and FDR, detect & remove non-active voxels

# %%
# one sample t-test against 0
tvals, pvals = ttest_1samp(clean_beta, popmean=0, axis=1, nan_policy='omit')

# FDR correction
tested = np.isfinite(pvals)
alpha=0.05
rej, q, _, _ = multipletests(pvals[tested], alpha=alpha, method='fdr_bh')

n_voxel = clean_beta.shape[0]
qvals  = np.full(n_voxel, np.nan)
reject = np.zeros(n_voxel, dtype=bool)
reject[tested] = rej
qvals[tested]  = q

# reject non-active voxels
clean_active_beta = clean_beta[reject]
clean_active_idx = keeped_indices[reject]
clean_active_bold = bold_data_reshape[reject]

# %%
# num_trials = beta.shape[-1]
clean_active_volume = np.full(bold_data.shape[:3]+(num_trials,), np.nan)
active_coords = tuple(coord[clean_active_idx] for coord in masked_coords)
clean_active_volume[active_coords[0], active_coords[1], active_coords[2], :] = clean_active_beta
print(5, flush=True)
# %% [markdown]
# apply filter

# # %%
# def hampel_filter_image(image, window_size, threshold_factor, return_stats=False):
#     footprint = np.ones((window_size,) * 3, dtype=bool)
#     insufficient_any = np.zeros(image.shape[:3], dtype=bool)
#     corrected_any = np.zeros(image.shape[:3], dtype=bool)

#     for t in range(image.shape[3]):
#         print(f"Trial Number: {t}", flush=True)
#         vol = image[..., t]
#         valid = np.isfinite(vol)

#         med = ndimage.generic_filter(vol, np.nanmedian, footprint=footprint, mode='constant', cval=np.nan).astype(np.float32)
#         mad = ndimage.generic_filter(np.abs(vol - med), np.nanmedian, footprint=footprint, mode='constant', cval=np.nan).astype(np.float32)
#         counts = ndimage.generic_filter(np.isfinite(vol).astype(np.float32), np.sum, footprint=footprint, mode='constant', cval=0).astype(np.float32)
#         neighbor_count = counts - valid.astype(np.float32)

#         scaled_mad = 1.4826 * mad
#         insufficient = valid & (neighbor_count < 3)
#         insufficient_any |= insufficient
#         image[..., t][insufficient] = np.nan

#         enough_data = (neighbor_count >= 3) & valid
#         outliers = enough_data & (np.abs(vol - med) > threshold_factor * scaled_mad)

#         corrected_any |= outliers
#         image[..., t][outliers] = med[outliers]

#     if return_stats:
#         stats = {
#             'insufficient_total': int(np.count_nonzero(insufficient_any)),
#             'corrected_total': int(np.count_nonzero(corrected_any)),
#         }
#         return image, stats

#     return image

# beta_volume_filter, hampel_stats = hampel_filter_image(clean_active_volume.astype(np.float32), window_size=5, threshold_factor=3, return_stats=True)
# # print('Insufficient neighbours per frame:', hampel_stats['insufficient_counts'], flush=True)
# print('Total voxels with <3 neighbours:', hampel_stats['insufficient_total'], flush=True)
# print('Total corrected voxels:', hampel_stats['corrected_total'], flush=True)

# # beta_volume_filter = beta_volume_filter[~np.all(np.isnan(beta_volume_filter), axis=-1)]
# # np.save(f'cleaned_beta_volume_sub{sub}_ses{ses}_run{run}.npy', beta_volume_filter)

# nan_voxels = np.all(np.isnan(beta_volume_filter), axis=-1) 
# mask_2d = nan_voxels.reshape(-1) 

# np.save(f'cleaned_beta_volume_sub{sub}_ses{ses}_run{run}.npy', beta_volume_filter)
# np.save(f'mask_all_nan_sub{sub}_ses{ses}_run{run}.npy', mask_2d)

######################################################################################################################
######################################################################################################################
######################################################################################################################

beta_volume_filter = np.load(f'cleaned_beta_volume_sub{sub}_ses{ses}_run{run}.npy')
nan_voxels = np.all(np.isnan(beta_volume_filter), axis=-1) 
mask_2d = np.load(f'mask_all_nan_sub{sub}_ses{ses}_run{run}.npy')
beta_valume_clean_2d = beta_volume_filter[~nan_voxels]      

active_flat_idx = np.ravel_multi_index(active_coords, nan_voxels.shape)
keep_mask = ~mask_2d[active_flat_idx]
clean_active_bold = clean_active_bold[keep_mask, ...]
clean_active_beta = clean_active_beta[keep_mask, ...]

# Drop voxels everywhere our Hampel mask removed them
clean_active_idx = clean_active_idx[keep_mask]
active_coords = tuple(coord[keep_mask] for coord in active_coords)


print(6, flush=True)
# %% [markdown]
# Load the filtered beta values

# %%
# mask_2d = np.load("mask_all_nan_sub04_ses1_run1.npy")


active_flat_idx = np.ravel_multi_index(active_coords, clean_active_volume.shape[:3])
active_keep_mask = ~mask_2d[active_flat_idx]
clean_active_bold = clean_active_bold[active_keep_mask]

# %% [markdown]
# Load Behaviour

# %%
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

behave_path = f'PSPD0{sub}_OFF_behav_metrics.mat'
behavior_matrix = load_behavior_metrics(behave_path, run)

######################################################################################################################
######################################################################################################################
######################################################################################################################
# %% [markdown]
# PCA

# %%
# def prepare_for_empca(data):
#     W = np.isfinite(data).astype(float)
#     Y = np.where(np.isfinite(data), data, 0.0)

#     row_weight = W.sum(axis=0, keepdims=True)
#     print(row_weight.shape)
#     mean = np.divide((Y * W).sum(axis=0, keepdims=True), row_weight, out=np.zeros_like(row_weight), where=row_weight > 0)
#     centered = Y - mean

#     var = np.divide((W * centered**2).sum(axis=0, keepdims=True), row_weight, out=np.zeros_like(row_weight), where=row_weight > 0)
#     scale = np.sqrt(var)
#     print(mean.shape, scale.shape)

#     z = np.divide(centered, np.maximum(scale, 1e-6), out=np.zeros_like(centered), where=row_weight > 0)
#     return z, W



# # %%
# trial_indices=None
# trial_len=trial_len 
# pca_components=None 
# pca_mean=None

# # num_trials = beta_valume_clean_2d.shape[-1]
# trial_idx = np.arange(num_trials) if trial_indices is None else np.unique(np.asarray(trial_indices, int).ravel())
# behavior_subset = behavior_matrix[trial_idx]
# X_bold = clean_active_bold[:, trial_idx, :]

# # ----- apply PCA -----
# print("PCA...", flush=True)
# X_reshap = X_bold.reshape(X_bold.shape[0], -1)
# X_reshap.shape

# Yc, W = prepare_for_empca(X_reshap.T)
# Yc = Yc.T
# W = W.T
# print(Yc.shape, W.shape)

# print("begin empca...", flush=True)
# m = empca(Yc.astype(np.float32, copy=False), W.astype(np.float32, copy=False), nvec=700, niter=15)
# np.save(f'empca_model_sub{sub}_ses{ses}_run{run}.npy', m)


#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

pca_model = np.load(f'empca_model_sub{sub}_ses{ses}_run{run}.npy', allow_pickle=True).item()
bold_pca = pca_model.eigvec

W_pca = np.linalg.pinv(pca_model.coeff)    
print(f"W_pca shape: {W_pca.shape}, beta_valume_clean_2d: {beta_valume_clean_2d.shape}", flush=True)
beta_pca = W_pca @ np.nan_to_num(beta_valume_clean_2d)


# pca_model.__dict__.keys()
# for i in range(pca_model.nvec):
#     print(pca_model.R2vec(i))
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
C_task, C_bold, C_beta, X_behave, X_beta = matrices_func(beta_pca, bold_pca, behavior_matrix, trial_indices=None, trial_len=trial_len, num_trials=90)
np.save(f'C_task_sub{sub}_ses{ses}_run{run}.npy', C_task)
np.save(f'C_bold_sub{sub}_ses{ses}_run{run}.npy', C_bold)
np.save(f'C_beta_sub{sub}_ses{ses}_run{run}.npy', C_beta)

alpha_bold=1.0
alpha_beta=10
alpha_task=1.0
n_comp = X_beta.shape[0]
regularization = 1e-6
solver_name = "MOSEK"
rho = 0.8

trial_select = np.isfinite(X_behave)
num_trials_selected = np.sum(trial_select)
X_behave_clean = X_behave[trial_select]
# valid_beh = np.isfinite(X_behave)
# mean_beh = np.nanmean(X_behave[valid_beh])
mean_beh = np.mean(X_behave_clean)
X_behave2 = X_behave_clean - mean_beh
behave_norm = np.linalg.norm(X_behave2)

X_beta_select = X_beta[:, trial_select]
X_beta2 = X_beta_select - np.mean(X_beta_select, axis=1, keepdims=True)

C_task = C_task / np.max(C_task)
C_bold = C_bold / np.max(C_bold)
C_beta = C_beta / np.max(C_beta)

C_total = alpha_task * np.diag(C_task) + alpha_bold * C_bold + alpha_beta * C_beta
C_total = 0.5 * (C_total + C_total.T)
C_total += regularization * np.eye(n_comp)
print('C_total range:', C_total.min(), C_total.max(), flush=True)

a = X_behave2 / behave_norm
w_plus = cp.Variable(n_comp)
w_minus = cp.Variable(n_comp)
u_plus = X_beta2.T @ w_plus
u_minus = X_beta2.T @ w_minus

constraints_plus = [w_plus >= 0, cp.sum(w_plus) == 1, cp.SOC((1.0/rho) * (a @ u_plus), u_plus)]
obj_plus = cp.Minimize(cp.quad_form(w_plus, cp.psd_wrap(C_total)))
prob_plus = cp.Problem(obj_plus, constraints_plus)

constraints_minus = [w_minus >= 0, cp.sum(w_minus) == 1, cp.SOC((1.0/rho) * (-a @ u_minus), u_minus)]
obj_minus = cp.Minimize(cp.quad_form(w_minus, cp.psd_wrap(C_total)))
prob_minus = cp.Problem(obj_minus, constraints_minus)
val_plus = val_minus = np.inf

print('Solving SOC optimization problems...', flush=True)
prob_plus.solve(solver=solver_name, warm_start=True)
if w_plus.value is not None and prob_plus.status in ("optimal", "optimal_inaccurate"):
    val_plus = prob_plus.value

print('Solving SOC optimization problems (minus)...', flush=True)
prob_minus.solve(solver=solver_name, warm_start=True)
if w_minus.value is not None and prob_minus.status in ("optimal", "optimal_inaccurate"):
    val_minus = prob_minus.value

if not np.isfinite(val_plus) and not np.isfinite(val_minus):
    raise RuntimeError("Both SOC branches were infeasible or failed to solve.")

if val_plus <= val_minus:
    weights = np.array(w_plus.value).ravel()
else:
    weights = np.array(w_minus.value).ravel()

print("Plus:", flush=True)
tmp = cp.quad_form(w_plus, cp.psd_wrap(alpha_task * np.diag(C_task)))
print(tmp.value.item())
tmp = cp.quad_form(w_plus, cp.psd_wrap(alpha_bold * C_bold))
print(tmp.value.item())
tmp = cp.quad_form(w_plus, cp.psd_wrap(alpha_beta * C_beta))
print(tmp.value.item())


print("Minus:", flush=True)
tmp = cp.quad_form(w_minus, cp.psd_wrap(alpha_task * np.diag(C_task)))
print(tmp.value.item())
tmp = cp.quad_form(w_minus, cp.psd_wrap(alpha_bold * C_bold))
print(tmp.value.item())
tmp = cp.quad_form(w_minus, cp.psd_wrap(alpha_beta * C_beta))
print(tmp.value.item())

W_prime = W_pca.T @ weights
np.save(f'behavior_weights_sub{sub}_ses{ses}_run{run}.npy', W_prime)
np.save(f'behavior_weights_pca_sub{sub}_ses{ses}_run{run}.npy', weights)

# Reconstruct a volumetric map of W_prime for visualization on the anatomical image
anatomical_image = anat_img
valid_coords = np.where(~nan_voxels)
weight_volume = np.full(anatomical_image.shape, np.nan, dtype=np.float32)
scaled_weights = np.abs(W_prime) * 1e6
weight_volume[valid_coords] = scaled_weights
vmin = 0.0
vmax = np.nanpercentile(scaled_weights, 95)
print(vmin, vmax)
weight_volume = np.clip(weight_volume, vmin, vmax)
weight_img = nib.Nifti1Image(weight_volume, anatomical_image.affine, anatomical_image.header)

view = plotting.view_img(
    weight_img,
    bg_img=anatomical_image,
    cmap="jet",
    symmetric_cmap=False,
)
suffix = f"_task{alpha_task:g}_bold{alpha_bold:g}_beta{alpha_beta:g}_rho{rho:g}"
view_path = f"W_prime_overlay_sub{sub}_ses{ses}_run{run}{suffix}.html"
view.save_as_html(view_path)
print(f"Saved visualization to {view_path}", flush=True)
print("finish!", flush=True)
