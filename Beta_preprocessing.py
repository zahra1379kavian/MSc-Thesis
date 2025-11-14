# %%
import nibabel as nib
import numpy as np
from os.path import join
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
import scipy.sparse as sp
import scipy.ndimage as ndimage


# %%
ses = 1
sub = '04'
run = 2
num_trials = 90
trial_len = 9

base_path = '/scratch/st-mmckeown-1/zkavian/fmri_models/MSc-Thesis'
anat_img = nib.load(f'/scratch/st-mmckeown-1/zkavian/fmri_models/MSc-Thesis/sub-pd0{sub}_ses-{ses}_T1w_brain.nii.gz')
bold_data = np.load(join(base_path, f'fmri_sub{sub}_ses{ses}_run{run}.npy'), allow_pickle=True)

mask_path = f'/scratch/st-mmckeown-1/zkavian/fmri_models/MSc-Thesis/sub-pd0{sub}_ses-{ses}_T1w_brain_mask.nii.gz'
back_mask = nib.load(mask_path)
back_mask = back_mask.get_fdata()
back_mask = back_mask.astype(np.float16)

mask_path = f'/scratch/st-mmckeown-1/zkavian/fmri_models/MSc-Thesis/sub-pd0{sub}_ses-{ses}_T1w_brain_pve_0.nii.gz'
csf_mask = nib.load(mask_path)
csf_mask = csf_mask.get_fdata()
csf_mask = csf_mask.astype(np.float16)

mask_path = f'/scratch/st-mmckeown-1/zkavian/fmri_models/MSc-Thesis/sub-pd0{sub}_ses-{ses}_T1w_brain_pve_1.nii.gz'
gray_mask = nib.load(mask_path)
gray_mask = gray_mask.get_fdata()
gray_mask = gray_mask.astype(np.float16)

print(1, flush=True)

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
# %%

glm_dict = np.load(f'/scratch/st-mmckeown-1/zkavian/fmri_models/MSc-Thesis/TYPED_FITHRF_GLMDENOISE_RR_sub{sub}.npy', allow_pickle=True).item()
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

# %% [markdown]
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
def hampel_filter_image(image, window_size, threshold_factor, return_stats=False):
    footprint = np.ones((window_size,) * 3, dtype=bool)
    insufficient_any = np.zeros(image.shape[:3], dtype=bool)
    corrected_any = np.zeros(image.shape[:3], dtype=bool)

    for t in range(image.shape[3]):
        print(f"Trial Number: {t}", flush=True)
        vol = image[..., t]
        valid = np.isfinite(vol)

        med = ndimage.generic_filter(vol, np.nanmedian, footprint=footprint, mode='constant', cval=np.nan).astype(np.float32)
        mad = ndimage.generic_filter(np.abs(vol - med), np.nanmedian, footprint=footprint, mode='constant', cval=np.nan).astype(np.float32)
        counts = ndimage.generic_filter(np.isfinite(vol).astype(np.float32), np.sum, footprint=footprint, mode='constant', cval=0).astype(np.float32)
        neighbor_count = counts - valid.astype(np.float32)

        scaled_mad = 1.4826 * mad
        insufficient = valid & (neighbor_count < 3)
        insufficient_any |= insufficient
        image[..., t][insufficient] = np.nan

        enough_data = (neighbor_count >= 3) & valid
        outliers = enough_data & (np.abs(vol - med) > threshold_factor * scaled_mad)

        corrected_any |= outliers
        image[..., t][outliers] = med[outliers]

    if return_stats:
        stats = {
            'insufficient_total': int(np.count_nonzero(insufficient_any)),
            'corrected_total': int(np.count_nonzero(corrected_any)),
        }
        return image, stats

    return image

beta_volume_filter, hampel_stats = hampel_filter_image(clean_active_volume.astype(np.float32), window_size=5, threshold_factor=3, return_stats=True)
# print('Insufficient neighbours per frame:', hampel_stats['insufficient_counts'], flush=True)
print('Total voxels with <3 neighbours:', hampel_stats['insufficient_total'], flush=True)
print('Total corrected voxels:', hampel_stats['corrected_total'], flush=True)

# beta_volume_filter = beta_volume_filter[~np.all(np.isnan(beta_volume_filter), axis=-1)]
# np.save(f'cleaned_beta_volume_sub{sub}_ses{ses}_run{run}.npy', beta_volume_filter)

nan_voxels = np.all(np.isnan(beta_volume_filter), axis=-1) 
mask_2d = nan_voxels.reshape(-1) 

np.save(f'cleaned_beta_volume_sub{sub}_ses{ses}_run{run}.npy', beta_volume_filter)
np.save(f'mask_all_nan_sub{sub}_ses{ses}_run{run}.npy', mask_2d)

######################################################################################################################
######################################################################################################################
######################################################################################################################

beta_volume_filter = np.load(f'cleaned_beta_volume_sub{sub}_ses{ses}_run{run}.npy')
nan_voxels = np.all(np.isnan(beta_volume_filter), axis=-1) 
mask_2d = np.load(f'mask_all_nan_sub{sub}_ses{ses}_run{run}.npy')
np.save(f"nan_mask_flat_sub{sub}_ses{ses}_run{run}.npy", mask_2d)
beta_volume_clean_2d = beta_volume_filter[~nan_voxels]     
np.save(f"beta_volume_filter_sub{sub}_ses{ses}_run{run}.npy.npy", beta_volume_clean_2d) 

active_flat_idx = np.ravel_multi_index(active_coords, nan_voxels.shape)
np.save(f"active_flat_indices__sub{sub}_ses{ses}_run{run}.npy", active_flat_idx)
keep_mask = ~mask_2d[active_flat_idx]
clean_active_bold = clean_active_bold[keep_mask, ...]
np.save(f"active_bold_sub{sub}_ses{ses}_run{run}.npy.npy", clean_active_bold)
clean_active_beta = clean_active_beta[keep_mask, ...]

# Drop voxels everywhere our Hampel mask removed them
clean_active_idx = clean_active_idx[keep_mask]
active_coords = tuple(coord[keep_mask] for coord in active_coords)
np.save(f"active_coords_sub{sub}_ses{ses}_run{run}.npy", active_coords)

# active_flat_idx = np.ravel_multi_index(active_coords, clean_active_volume.shape[:3])
# active_keep_mask = ~mask_2d[active_flat_idx]
# clean_active_bold = clean_active_bold[active_keep_mask]
