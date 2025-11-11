#!/usr/bin/env python3
import argparse
import numpy as np
import main_bootstrap as mb


def _default_alpha():
    sweep = getattr(mb, "alpha_sweep", None)
    if sweep:
        return sweep[0]
    return {"task_penalty": 2000.0, "bold_penalty": 1500.0, "beta_penalty": 2500.0}


def _default_rho():
    rho_values = getattr(mb, "rho_sweep", None)
    if rho_values:
        return rho_values[0]
    return getattr(mb, "soc_ratio", 0.95)


def parse_args():
    alpha_defaults = _default_alpha()
    parser = argparse.ArgumentParser(
        description="Summarize key statistics for matrices, losses, and weights used in main_bootstrap."
    )
    parser.add_argument("--task-alpha", type=float, default=alpha_defaults["task_penalty"], help="Task penalty weight.")
    parser.add_argument("--bold-alpha", type=float, default=alpha_defaults["bold_penalty"], help="Bold penalty weight.")
    parser.add_argument("--beta-alpha", type=float, default=alpha_defaults["beta_penalty"], help="Beta penalty weight.")
    parser.add_argument("--rho", type=float, default=_default_rho(), help="SOC ratio / rho value.")
    parser.add_argument(
        "--solver", type=str, default=getattr(mb, "solver_name", "MOSEK"), help="Convex solver to use for SOC problem."
    )
    return parser.parse_args()


def collect_stats(array):
    arr = np.asarray(array)
    stats = {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "size": arr.size,
        "finite_count": 0,
        "finite_fraction": 0.0,
        "min": np.nan,
        "max": np.nan,
        "range": np.nan,
        "mean": np.nan,
        "std": np.nan,
        "median": np.nan,
        "abs_max": np.nan,
        "l1_sum": np.nan,
        "l2_norm": np.nan,
    }
    if arr.size == 0:
        return stats

    finite_mask = np.isfinite(arr)
    finite_count = int(np.count_nonzero(finite_mask))
    stats["finite_count"] = finite_count
    stats["finite_fraction"] = finite_count / arr.size

    if finite_count:
        finite_vals = arr[finite_mask].astype(np.float64, copy=False)
        stats["min"] = float(np.min(finite_vals))
        stats["max"] = float(np.max(finite_vals))
        stats["range"] = stats["max"] - stats["min"]
        stats["mean"] = float(np.mean(finite_vals))
        stats["std"] = float(np.std(finite_vals))
        stats["median"] = float(np.median(finite_vals))
        stats["abs_max"] = float(np.max(np.abs(finite_vals)))
        stats["l1_sum"] = float(np.sum(np.abs(finite_vals)))
        stats["l2_norm"] = float(np.linalg.norm(finite_vals))

    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        diag = np.diag(arr)
        stats["trace"] = float(np.nansum(diag)) if diag.size else np.nan
        diag_mask = np.isfinite(diag)
        if np.any(diag_mask):
            diag_vals = diag[diag_mask].astype(np.float64, copy=False)
            stats["diag_mean"] = float(np.mean(diag_vals))
            stats["diag_std"] = float(np.std(diag_vals)) if diag_vals.size > 1 else 0.0
        else:
            stats["diag_mean"] = np.nan
            stats["diag_std"] = np.nan
    return stats


def fmt(value, digits=5):
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}g}"


def print_stats(name, stats):
    finite = stats.get("finite_count", 0)
    size = stats.get("size", 0)
    fraction = stats.get("finite_fraction", 0.0)
    fraction_str = f"{fraction:.2%}" if size else "n/a"
    print(
        f"- {name}: shape={stats.get('shape')}, dtype={stats.get('dtype')}, "
        f"finite={finite}/{size} ({fraction_str})"
    )
    if finite == 0:
        return
    print(f"  range=[{fmt(stats['min'])}, {fmt(stats['max'])}] span={fmt(stats['range'])}")
    print(f"  mean={fmt(stats['mean'])}, std={fmt(stats['std'])}, median={fmt(stats['median'])}")
    print(f"  abs_max={fmt(stats['abs_max'])}, l1_sum={fmt(stats['l1_sum'])}, l2_norm={fmt(stats['l2_norm'])}")
    if "trace" in stats:
        print(f"  trace={fmt(stats['trace'])}, diag_mean={fmt(stats.get('diag_mean'))}, diag_std={fmt(stats.get('diag_std'))}")


def print_section(title):
    print("\n" + title)
    print("-" * len(title))


def describe_run(label, run_data):
    print_section(f"{label} run data")
    nan_mask = run_data.get("nan_mask_flat")
    if nan_mask is not None:
        nan_mask = np.asarray(nan_mask)
        if nan_mask.size:
            invalid = int(np.count_nonzero(nan_mask))
            total = nan_mask.size
            print(f"Voxels without usable betas: {invalid}/{total} ({invalid / total:.2%})")
    coords = run_data.get("active_coords")
    if coords:
        num_active = coords[0].size
        print(f"Active voxel count: {num_active}")

    for key, pretty in (
        ("beta_centered", "Beta (centered)"),
        ("beta_observed", "Beta (observed)"),
        ("behavior_centered", "Behavior (centered)"),
        ("behavior_observed", "Behavior (observed)"),
        ("normalized_behaviors", "Behavior (normalized)"),
        ("coeff_pinv", "PCA coefficient pseudoinverse"),
    ):
        array = run_data.get(key)
        if array is None:
            continue
        stats = collect_stats(array)
        print_stats(pretty, stats)


def describe_matrices(title, matrices):
    print_section(title)
    for label, matrix in matrices.items():
        print_stats(label, collect_stats(matrix))


def report_solution(solution, train_data, test_data):
    weights = solution["weights"]
    print_section("Weight summary")
    print_stats("Behavior weights", collect_stats(weights))
    train_metrics = mb.evaluate_projection(train_data, weights)
    test_metrics = mb.evaluate_projection(test_data, weights)
    print(f"SOC branch: {solution['branch']}")
    print(f"Total loss: {fmt(solution['total_loss'])}")
    print("Penalty contributions:")
    for label, value in solution["penalty_contributions"].items():
        print(f"  {label}: {fmt(value)}")
    print_section("Prediction metrics")
    for split, metrics in (("train", train_metrics), ("test", test_metrics)):
        print(
            f"{split.capitalize()}: corr={fmt(metrics.get('pearson'))}, "
            f"R2={fmt(metrics.get('r2'))}, MSE={fmt(metrics.get('mse'))}"
        )


def prepare_runs():
    train_data = mb.prepare_data_func(
        mb.run_train,
        mb.bold_data_train,
        mb.beta_glm,
        mb.beta_volume_filter_train,
        mb.brain_mask,
        mb.csf_mask,
        mb.gray_mask,
        mb.behavior_matrix_train,
        mb.pca_model_train,
        mb.trial_len,
        mb.num_trials,
    )
    test_data = mb.prepare_data_func(
        mb.run_test,
        mb.bold_data_test,
        mb.beta_glm,
        mb.beta_volume_filter_test,
        mb.brain_mask,
        mb.csf_mask,
        mb.gray_mask,
        mb.behavior_matrix_test,
        mb.pca_model_train,
        mb.trial_len,
        mb.num_trials,
        reuse_nan_mask=train_data["nan_mask_flat"],
        reuse_active_coords=train_data["active_coords"],
    )
    return train_data, test_data


def main():
    args = parse_args()
    print_section("Configuration")
    print(
        f"Subject=sub-pd0{mb.sub}, session={mb.ses}, runs(train/test)=({mb.run_train}, {mb.run_test}), "
        f"trials={mb.num_trials}, trial_len={mb.trial_len}"
    )
    print(
        f"Penalties: task={args.task_alpha}, bold={args.bold_alpha}, beta={args.beta_alpha}, "
        f"rho={args.rho}, solver={args.solver}"
    )

    train_data, test_data = prepare_runs()
    describe_run("Train", train_data)
    describe_run("Test", test_data)

    base_mats = {
        "C_task": train_data["C_task"],
        "C_bold": train_data["C_bold"],
        "C_beta": train_data["C_beta"],
    }
    describe_matrices("Unscaled regularizers (train)", base_mats)

    penalty_mats, total_penalty = mb.calcu_penalty_terms(
        train_data, args.task_alpha, args.bold_alpha, args.beta_alpha
    )
    penalty_with_total = dict(penalty_mats)
    penalty_with_total["total_penalty"] = total_penalty
    describe_matrices("Scaled penalty matrices", penalty_with_total)

    solution = mb.solve_soc_problem(
        train_data, args.task_alpha, args.bold_alpha, args.beta_alpha, args.solver, args.rho
    )
    report_solution(solution, train_data, test_data)


if __name__ == "__main__":
    main()
