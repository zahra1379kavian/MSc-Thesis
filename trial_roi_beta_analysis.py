#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
import tarfile
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets, image, plotting


@dataclass(frozen=True)
class BetaData:
    beta_clean: np.ndarray  # (n_voxels, n_trials_total)
    beta_clean_flat_idx: np.ndarray  # (n_voxels,)
    volume_shape: tuple
    affine: np.ndarray
    header: nib.Nifti1Header


def _synchronize_beta_voxels(beta_run1, beta_run2, mask_run1, mask_run2):
    valid_run1, valid_run2 = ~mask_run1, ~mask_run2
    shared_valid = valid_run1 & valid_run2
    shared_mask_run1 = shared_valid[np.flatnonzero(valid_run1)]
    shared_mask_run2 = shared_valid[np.flatnonzero(valid_run2)]

    synced_run1, synced_run2 = beta_run1[shared_mask_run1], beta_run2[shared_mask_run2]
    shared_flat_indices = np.flatnonzero(shared_valid)
    combined_nan_mask = mask_run1 | mask_run2
    return synced_run1, synced_run2, combined_nan_mask, shared_flat_indices


def _combine_filtered_betas(beta_run1, beta_run2, nan_mask_flat, flat_indices):
    union_nan_mask = np.isnan(beta_run1) | np.isnan(beta_run2)
    if np.any(union_nan_mask):
        beta_run1 = beta_run1.copy()
        beta_run2 = beta_run2.copy()
        beta_run1[union_nan_mask] = np.nan
        beta_run2[union_nan_mask] = np.nan

    beta_combined = np.concatenate((beta_run1, beta_run2), axis=1)
    valid_voxel_mask = ~np.all(np.isnan(beta_combined), axis=1)
    kept_flat_indices = flat_indices[valid_voxel_mask]

    beta_combined = beta_combined[valid_voxel_mask]
    dropped_flat = flat_indices[~valid_voxel_mask]
    if dropped_flat.size:
        nan_mask_flat = nan_mask_flat.copy()
        nan_mask_flat[dropped_flat] = True
    return beta_combined, nan_mask_flat, kept_flat_indices


def load_subject_betas(base_path: Path, sub: str, ses: int) -> tuple[nib.Nifti1Image, BetaData]:
    anat_img = nib.load(str(base_path / f"sub-pd0{sub}_ses-{ses}_T1w_brain.nii.gz"))
    volume_shape = anat_img.shape[:3]

    nan_mask_flat_run1 = np.load(str(base_path / f"nan_mask_flat_sub{sub}_ses{ses}_run1.npy"))
    nan_mask_flat_run2 = np.load(str(base_path / f"nan_mask_flat_sub{sub}_ses{ses}_run2.npy"))

    beta_filtered_run1 = np.load(str(base_path / f"beta_volume_filter_sub{sub}_ses{ses}_run1.npy"))
    beta_filtered_run2 = np.load(str(base_path / f"beta_volume_filter_sub{sub}_ses{ses}_run2.npy"))

    beta_filtered_run1, beta_filtered_run2, shared_nan_mask_flat, shared_flat_indices = _synchronize_beta_voxels(
        beta_filtered_run1, beta_filtered_run2, nan_mask_flat_run1, nan_mask_flat_run2
    )
    beta_clean, _, beta_clean_flat_idx = _combine_filtered_betas(
        beta_filtered_run1, beta_filtered_run2, shared_nan_mask_flat.ravel(), shared_flat_indices
    )

    return anat_img, BetaData(
        beta_clean=beta_clean,
        beta_clean_flat_idx=beta_clean_flat_idx,
        volume_shape=volume_shape,
        affine=anat_img.affine,
        header=anat_img.header,
    )


def _find_local_harvardoxford_tgz() -> Optional[Path]:
    candidates = [
        Path(__file__).resolve().parent / "HarvardOxford.tgz",
        Path.cwd() / "HarvardOxford.tgz",
        Path.cwd().parent / "HarvardOxford.tgz",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _ensure_harvardoxford_extracted(tgz_path: Path, extract_root: Path) -> Path:
    extract_root.mkdir(parents=True, exist_ok=True)
    marker = extract_root / ".extracted.ok"
    if marker.is_file():
        return extract_root

    with tarfile.open(tgz_path, mode="r:gz") as tf:
        tf.extractall(path=extract_root)

    marker.write_text(f"source={tgz_path}\n", encoding="utf-8")
    return extract_root


def _load_harvardoxford_labels_from_xml(xml_path: Path) -> list[str]:
    root = ET.parse(str(xml_path)).getroot()
    labels: dict[int, str] = {}
    for el in root.iter("label"):
        idx_s = el.attrib.get("index")
        if idx_s is None:
            continue
        try:
            idx = int(idx_s)
        except ValueError:
            continue
        labels[idx] = (el.text or "").strip()

    if not labels:
        raise RuntimeError(f"No <label> entries found in {xml_path}")

    max_idx = max(labels)
    ordered = [labels.get(i, "") for i in range(max_idx + 1)]
    if any(not name for name in ordered):
        raise RuntimeError(f"Missing label names in {xml_path}")
    return ordered


def _fetch_harvard_oxford_atlas_offline(atlas_name: str, tgz_path: Path, data_dir: Path):
    extract_root = _ensure_harvardoxford_extracted(tgz_path, data_dir / "harvardoxford_fsl")
    base = extract_root / "data" / "atlases"

    parts = atlas_name.split("-")
    if len(parts) != 4:
        raise ValueError(f"Unexpected atlas_name format: {atlas_name!r}")
    atlas_family, kind, thr_part, res = parts
    if atlas_family not in {"cort", "cortl", "sub"}:
        raise ValueError(f"Unexpected atlas family: {atlas_family!r}")
    if kind not in {"maxprob", "prob"}:
        raise ValueError(f"Unexpected atlas kind: {kind!r}")
    if res not in {"1mm", "2mm"}:
        raise ValueError(f"Unexpected atlas resolution: {res!r}")

    if kind == "prob":
        nii_name = f"HarvardOxford-{atlas_family}-prob-{res}.nii.gz"
    else:
        if not thr_part.startswith("thr"):
            raise ValueError(f"Unexpected threshold segment: {thr_part!r}")
        nii_name = f"HarvardOxford-{atlas_family}-maxprob-{thr_part}-{res}.nii.gz"

    nii_path = base / "HarvardOxford" / nii_name
    if not nii_path.is_file():
        raise FileNotFoundError(f"Missing atlas file in extracted tarball: {nii_path}")

    if atlas_family == "cort":
        xml_path = base / "HarvardOxford-Cortical.xml"
    elif atlas_family == "cortl":
        xml_path = base / "HarvardOxford-Cortical-Lateralized.xml"
    else:
        xml_path = base / "HarvardOxford-Subcortical.xml"
    if not xml_path.is_file():
        raise FileNotFoundError(f"Missing labels XML in extracted tarball: {xml_path}")

    labels = ["Background"] + _load_harvardoxford_labels_from_xml(xml_path)
    return {"maps": str(nii_path), "labels": labels}


def build_harvard_oxford_roi_masks(anat_img: nib.Nifti1Image, roi_names: list[str]):
    tgz_path = _find_local_harvardoxford_tgz()
    if tgz_path is not None:
        offline_atlas = _fetch_harvard_oxford_atlas_offline(
            "cort-maxprob-thr25-2mm",
            tgz_path=tgz_path,
            data_dir=Path(__file__).resolve().parent / "nilearn_data",
        )
        atlas_maps = offline_atlas["maps"]
        labels = list(offline_atlas["labels"])
    else:
        atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
        atlas_maps = atlas.maps
        labels = list(atlas.labels)

    label_ids = {}
    for name in roi_names:
        ids = [i for i, lab in enumerate(labels) if lab == name]
        if not ids:
            raise ValueError(f"ROI label not found in HarvardOxford labels: {name!r}")
        label_ids[name] = ids[0]

    atlas_resamp = image.resample_to_img(
        atlas_maps,
        anat_img,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    atlas_data = atlas_resamp.get_fdata().astype(np.int32, copy=False)
    anat_data = anat_img.get_fdata()
    brain_mask = np.isfinite(anat_data) & (anat_data != 0)
    masks = {name: ((atlas_data == idx) & brain_mask) for name, idx in label_ids.items()}
    return masks, {"maps": atlas_maps, "labels": labels}, atlas_resamp


def save_roi_overview_png(out_path: Path, anat_img: nib.Nifti1Image, masks: dict[str, np.ndarray]):
    label_data = np.zeros(anat_img.shape[:3], dtype=np.int16)
    name_to_label = {}
    current = 1
    for name, mask in masks.items():
        label_data[np.asarray(mask, dtype=bool)] = current
        name_to_label[name] = current
        current += 1

    label_img = nib.Nifti1Image(label_data, anat_img.affine, anat_img.header)
    display = plotting.plot_roi(
        roi_img=label_img,
        bg_img=anat_img,
        title="Defined ROIs (integer labels)",
        cmap="tab20",
        alpha=0.6,
        display_mode="mosaic",
        cut_coords=10,
    )
    display.savefig(str(out_path), dpi=200)
    display.close()
    return name_to_label


def trial_beta_img(beta_data: BetaData, trial_idx: int) -> nib.Nifti1Image:
    beta = beta_data.beta_clean[:, trial_idx]
    vol = np.full(int(np.prod(beta_data.volume_shape)), 0.0, dtype=np.float32)
    vol[beta_data.beta_clean_flat_idx] = np.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )
    vol = vol.reshape(beta_data.volume_shape)
    return nib.Nifti1Image(np.abs(vol), beta_data.affine, beta_data.header)


def mean_abs_beta_img(beta_data: BetaData, trial_indices: list[int]) -> nib.Nifti1Image:
    if not trial_indices:
        raise ValueError("trial_indices must be non-empty")
    trial_indices = [int(t) for t in trial_indices]
    x = beta_data.beta_clean[:, trial_indices]
    mean_abs = np.nanmean(np.abs(x), axis=1)
    vol = np.zeros(int(np.prod(beta_data.volume_shape)), dtype=np.float32)
    vol[beta_data.beta_clean_flat_idx] = np.nan_to_num(mean_abs, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )
    vol = vol.reshape(beta_data.volume_shape)
    return nib.Nifti1Image(vol, beta_data.affine, beta_data.header)


def compute_roi_trial_stats(
    beta_data: BetaData,
    masks: dict[str, np.ndarray],
    n_trials: int,
    voxel_abs_threshold: Optional[float] = None,
) -> pd.DataFrame:
    abs_beta = np.abs(beta_data.beta_clean[:, :n_trials])
    all_mean = np.nanmean(abs_beta, axis=0)

    rows = []
    for name, mask_3d in masks.items():
        mask_flat = np.asarray(mask_3d, dtype=bool).ravel()
        roi_in_clean = np.isin(beta_data.beta_clean_flat_idx, np.flatnonzero(mask_flat))
        if roi_in_clean.sum() == 0:
            raise RuntimeError(
                f"ROI {name!r} has no overlap with beta voxels. "
                "This likely means the atlas ROI is not aligned with subject space."
            )
        roi_mean = np.nanmean(abs_beta[roi_in_clean, :], axis=0)
        roi_signed_mean = np.nanmean(beta_data.beta_clean[roi_in_clean, :n_trials], axis=0)
        row = {
            "trial": np.arange(n_trials, dtype=int),
            "roi": name,
            "roi_mean_abs_beta": roi_mean.astype(float),
            "roi_mean_signed_beta": roi_signed_mean.astype(float),
            "wholebrain_mean_abs_beta": all_mean.astype(float),
            "roi_to_wholebrain_ratio": (roi_mean / (all_mean + 1e-12)).astype(float),
            "roi_overlap_voxels": int(roi_in_clean.sum()),
        }

        if voxel_abs_threshold is not None:
            thr = float(voxel_abs_threshold)
            roi_beta = beta_data.beta_clean[roi_in_clean, :n_trials]
            roi_beta_abs = np.abs(roi_beta)
            vox_keep = roi_beta_abs >= thr
            kept_counts = vox_keep.sum(axis=0).astype(int)
            with np.errstate(invalid="ignore"):
                thr_mean_abs = np.nanmean(np.where(vox_keep, roi_beta_abs, np.nan), axis=0)
                thr_mean_signed = np.nanmean(np.where(vox_keep, roi_beta, np.nan), axis=0)
            row.update(
                {
                    "voxel_abs_threshold": thr,
                    "roi_voxels_ge_thr": kept_counts,
                    "roi_mean_abs_beta_ge_thr": thr_mean_abs.astype(float),
                    "roi_mean_signed_beta_ge_thr": thr_mean_signed.astype(float),
                }
            )

        rows.append(pd.DataFrame(row))
    return pd.concat(rows, ignore_index=True)


def select_trials_motor(
    stats: pd.DataFrame, motor_rois: list[str], method: str, value: float, metric: str
) -> list[int]:
    motor = stats[stats["roi"].isin(motor_rois)].copy()
    if motor.empty:
        raise ValueError("No rows found for motor ROIs; check ROI names.")
    if metric not in motor.columns:
        raise ValueError(f"Selection metric not found in stats: {metric!r}")
    motor = motor.groupby("trial", as_index=False)[metric].mean()

    if method == "topk":
        k = int(value)
        if k <= 0:
            return []
        return motor.sort_values(metric, ascending=False).head(k)["trial"].astype(int).tolist()
    if method == "percentile":
        p = float(value)
        if not (0.0 < p < 100.0):
            raise ValueError("--select-value for percentile must be in (0, 100)")
        thresh = float(np.nanquantile(motor[metric].to_numpy(), p / 100.0))
        return motor.loc[motor[metric] >= thresh, "trial"].astype(int).tolist()
    if method == "zscore":
        z = float(value)
        x = motor[metric].to_numpy()
        mu = float(np.nanmean(x))
        sd = float(np.nanstd(x, ddof=1))
        thresh = mu + z * sd
        return motor.loc[motor[metric] >= thresh, "trial"].astype(int).tolist()
    if method == "abs-threshold":
        thresh = float(value)
        return motor.loc[motor[metric] >= thresh, "trial"].astype(int).tolist()

    raise ValueError(f"Unknown selection method: {method!r}")


def main():
    parser = argparse.ArgumentParser(description="Compute ROI beta stats per trial and select motor trials.")
    parser.add_argument("--sub", default="09")
    parser.add_argument("--ses", type=int, default=1)
    parser.add_argument("--base-dir", default=None, help="Defaults to MSc-Thesis/sub{sub}")
    parser.add_argument("--n-trials", type=int, default=90, help="Number of trials to analyze (starting at 0).")
    parser.add_argument("--out-dir", default=None, help="Defaults to base-dir/roi_trial_analysis")
    parser.add_argument(
        "--motor-rois",
        nargs="+",
        default=["Precentral Gyrus"],
        help="HarvardOxford cortical labels to treat as motor cortex.",
    )
    parser.add_argument(
        "--extra-rois",
        nargs="*",
        default=["Postcentral Gyrus"],
        help="Additional ROIs to compute stats for and include in ROI overview image.",
    )
    parser.add_argument(
        "--voxel-abs-threshold",
        type=float,
        default=None,
        help=(
            "If set, also compute ROI means restricted to voxels with |beta| >= threshold "
            "(columns *_ge_thr in trial_roi_stats.csv)."
        ),
    )
    parser.add_argument(
        "--select-method",
        choices=["topk", "percentile", "zscore", "abs-threshold"],
        default="percentile",
        help="How to select motor trials from motor-ROI mean |beta|.",
    )
    parser.add_argument(
        "--select-value",
        type=float,
        default=95.0,
        help="Value for selection method: topk=k, percentile=p (0..100), zscore=z, abs-threshold=t.",
    )
    parser.add_argument(
        "--select-metric",
        default="roi_mean_abs_beta",
        help="Which column in trial_roi_stats.csv to use for motor trial selection.",
    )
    parser.add_argument(
        "--motor-signed-threshold",
        type=float,
        default=0.6,
        help="If set, also save trials where motor ROI mean signed beta > threshold, plus the mean over those trials.",
    )
    parser.add_argument(
        "--save-selected-mean-html",
        action="store_true",
        help="Save mean |beta| over selected trials as NIfTI + interactive HTML viewer.",
    )
    parser.add_argument(
        "--save-beta-imgs",
        action="store_true",
        help="Also save a NIfTI image of |beta| for each trial (can be slow/large).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    default_base_dir = script_dir / f"sub{args.sub}"
    base_dir = Path(args.base_dir).expanduser() if args.base_dir else default_base_dir
    base_dir = base_dir.resolve()
    out_dir = Path(args.out_dir) if args.out_dir else (base_dir / "roi_trial_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not base_dir.exists():
        raise FileNotFoundError(
            f"Base directory not found: {base_dir}\n"
            "Tip: pass --base-dir with an absolute path, or run from the repo root."
        )

    anat_img, beta_data = load_subject_betas(base_dir, args.sub, args.ses)

    roi_names = list(dict.fromkeys(args.motor_rois + (args.extra_rois or [])))
    masks, atlas, atlas_resamp = build_harvard_oxford_roi_masks(anat_img, roi_names)

    roi_label_map = save_roi_overview_png(out_dir / "roi_overview.png", anat_img, masks)

    stats = compute_roi_trial_stats(
        beta_data, masks, n_trials=args.n_trials, voxel_abs_threshold=args.voxel_abs_threshold
    )
    stats_path = out_dir / "trial_roi_stats.csv"
    stats.to_csv(stats_path, index=False)

    selected_trials = select_trials_motor(
        stats,
        motor_rois=args.motor_rois,
        method=args.select_method,
        value=args.select_value,
        metric=args.select_metric,
    )
    selected_trials = sorted(set(int(t) for t in selected_trials))

    motor = stats[stats["roi"].isin(args.motor_rois)].groupby("trial", as_index=False)[
        ["roi_mean_abs_beta", "roi_mean_signed_beta", "roi_to_wholebrain_ratio"]
    ].mean()
    motor = motor.sort_values("roi_mean_abs_beta", ascending=False)
    motor.to_csv(out_dir / "motor_trial_ranking.csv", index=False)

    if args.save_selected_mean_html and selected_trials:
        mean_img = mean_abs_beta_img(beta_data, selected_trials)
        mean_nii_path = out_dir / "mean_abs_beta_selected_trials.nii.gz"
        nib.save(mean_img, str(mean_nii_path))
        mean_data = mean_img.get_fdata()
        nonzero = mean_data[np.isfinite(mean_data) & (mean_data > 0)]
        vmax = float(np.nanpercentile(nonzero, 99)) if nonzero.size else None
        view = plotting.view_img(
            mean_img,
            bg_img=anat_img,
            threshold=0.0,
            symmetric_cmap=False,
            vmax=vmax,
            cmap="magma",
        )
        view.save_as_html(str(out_dir / "mean_abs_beta_selected_trials.html"))

    summary = {
        "sub": args.sub,
        "ses": args.ses,
        "n_trials": args.n_trials,
        "motor_rois": args.motor_rois,
        "extra_rois": args.extra_rois,
        "voxel_abs_threshold": args.voxel_abs_threshold,
        "roi_label_map": roi_label_map,
        "selection": {
            "method": args.select_method,
            "value": args.select_value,
            "metric": args.select_metric,
            "trials": selected_trials,
        },
        "selected_trials_mean_motor_abs_beta": float(
            motor[motor["trial"].isin(selected_trials)]["roi_mean_abs_beta"].mean()
        )
        if selected_trials
        else None,
        "selected_trials_mean_motor_signed_beta": float(
            motor[motor["trial"].isin(selected_trials)]["roi_mean_signed_beta"].mean()
        )
        if selected_trials
        else None,
    }

    if args.motor_signed_threshold is not None:
        thr = float(args.motor_signed_threshold)
        above = motor[motor["roi_mean_signed_beta"] > thr].copy()
        above = above.sort_values("roi_mean_signed_beta", ascending=False)
        above_path = out_dir / f"motor_trials_signed_beta_gt_{thr:g}.csv"
        above.to_csv(above_path, index=False)
        summary["motor_signed_threshold"] = {
            "threshold": thr,
            "n_trials": int(above.shape[0]),
            "trials": above["trial"].astype(int).tolist(),
            "mean_motor_abs_beta": float(above["roi_mean_abs_beta"].mean()) if not above.empty else None,
            "mean_motor_signed_beta": float(above["roi_mean_signed_beta"].mean()) if not above.empty else None,
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    if args.save_beta_imgs:
        beta_dir = out_dir / "beta_imgs"
        beta_dir.mkdir(parents=True, exist_ok=True)
        for t in range(args.n_trials):
            img = trial_beta_img(beta_data, t)
            nib.save(img, str(beta_dir / f"trial_{t:03d}_abs_beta.nii.gz"))

    print(f"Wrote ROI overview: {out_dir / 'roi_overview.png'}")
    print(f"Wrote per-trial ROI stats: {stats_path}")
    print(f"Wrote motor trial ranking: {out_dir / 'motor_trial_ranking.csv'}")
    print(f"Wrote selection summary: {out_dir / 'summary.json'}")
    print(f"Selected trials ({len(selected_trials)}/{args.n_trials}): {selected_trials}")
    if args.save_selected_mean_html and selected_trials:
        print(f"Wrote selected-trials mean |beta| NIfTI: {out_dir / 'mean_abs_beta_selected_trials.nii.gz'}")
        print(f"Wrote selected-trials mean |beta| HTML: {out_dir / 'mean_abs_beta_selected_trials.html'}")
    if selected_trials:
        print(
            f"Mean motor ROI mean|beta| over selected trials: {summary['selected_trials_mean_motor_abs_beta']}"
        )
        print(
            f"Mean motor ROI mean beta (signed) over selected trials: {summary['selected_trials_mean_motor_signed_beta']}"
        )


if __name__ == "__main__":
    main()
