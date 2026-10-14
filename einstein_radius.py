"""einstein_radius.py - Einstein radii of the real Euclid lenses estimated with UNet-Radius-mini, compared with the
radii measured by lens modelling

UNet-Starlight.

UNet-Radius-mini is first trained on simulated images (train_simulations.py). This script applies it to the real lenses
of the EUCLID-SAGE benchmarks, and fine-tunes it on their measured Einstein radii.

Measured radii
    The lens models of the Euclid Q1 Strong Lensing Discovery Engine, fitted with PyAutoLens (Zenodo record 15025832,
    file modeling_lens_mass.csv). The measured radius is the median of the posterior of the effective Einstein radius
    (einstein_radius_effective_median_pdf, arcseconds). The models are joined to the benchmark lenses by the lens
    identifier (id_str in the Zenodo file, lens_id in the benchmark tables). Lenses without a model are not used, nor
    lenses with a radius below --min-radius (0.05"): the effective Einstein radius is 0 when the lens model has no
    tangential critical curve, so such a value does not measure the size of the lens. The report gives the
    distribution of the measured radii and the median of their 1-sigma uncertainties.

Networks (--networks NAME=PATH ...)
    One or more UNet-Radius-mini networks trained on simulations. In the paper: the network trained on the simulated
    benchmark (radius-mini), and the network trained on the second simulated set, made with simulate.py radius-set
    (radius-mini-10k).

1. Networks applied without training on real stamps
    Each network is applied to the real stamps with the input transform of its simulated training images. The
    simulations have pixels of 0.101", the real stamps pixels of 0.1"; a structure that spans the same number of pixels
    is therefore 0.1 / 0.101 times smaller on the sky in a real stamp, and the estimated radius is multiplied by this
    factor. For comparison, the radius is also estimated from the segmentation output alone: the median distance from
    the stamp centre of the pixels with an arc probability of at least 0.5 (the threshold saved with the network).
    Lenses without such pixels have no estimate of this kind.

2. Networks fine-tuned on the measured radii
    The encoder of UNet-Radius-mini and its Einstein radius layer (unet_models.RadiusRegressor) are trained on the
    measured radii, with the absolute error as the loss; the decoder is not used, because the real stamps have no arc
    masks. The regressor starts either from the weights of a network trained on simulations (initialisation
    pretrained-NAME) or from random weights (random; with the architecture and input transform of the first network
    of --networks). The procedure follows that of the classifiers (train_benchmarks.py):
    - five folds (seed 42), so that every lens is tested once; within each fold, 80% of the training lenses for
      training and 20% for validation (seed 42 + fold number). The radii are continuous, so the folds are not
      stratified;
    - for each learning rate of the grid (1e-4, 3e-4, 1e-3, 3e-3): 60 epochs with AdamW, weight decay 1e-4, batches of
      16 stamps, a learning rate that decreases to zero along a cosine, and the weights of the epoch with the lowest
      validation error;
    - the learning rate with the lowest validation error (rounded to four decimals; for equal values, the lower rate)
      is chosen, and its network is tested once on the test lenses of the fold;
    - input: asinh(x / s) with the scale s saved with the network, standardised with the mean and standard deviation
      of the training stamps of the fold; training stamps are randomly rotated and reflected (data.py);
    - the random generators are seeded with 1000 s + k (training seed s, fold k); every training seed repeats the
      search of the learning rate.

Metrics (over the lenses with a measured radius; error e = estimated - measured radius)
    Mean absolute error with a 95% bootstrap interval (2,000 resamples of the lenses), median absolute error, root mean
    square error, normalised median absolute deviation NMAD = 1.4826 median |e - median(e)| (a robust estimate of the
    standard deviation, equal to it for a normal distribution), mean error (bias), median relative error |e| / measured
    radius, Spearman rank correlation of estimated and measured radii, and the mean absolute error of grade A and grade
    B lenses. With several training seeds: mean and standard deviation of the metrics over the seeds, and the metrics of
    the estimate averaged over the seeds.
    Paired comparisons of two estimates for the same lenses use the difference of their absolute errors: its mean with
    a bootstrap interval, and the Wilcoxon signed-rank test, which does not assume normally distributed differences.
    The comparisons are: each pretrained initialisation against random weights; the pretrained initialisations against
    each other; each fine-tuned network against the same network applied without fine-tuning; and the applied networks
    against each other.

Outputs (results/<--out-name>/; <--out-name>_quick/ for a quick run)
    applied_<name>.csv                              per lens: identifiers, grade, measured radius, estimated radius
                                                    and arc-pixel radius
    finetuned_<initialisation>_seed<s>.csv          per lens: estimate of the network of the fold in which the lens
                                                    was a test lens, fold, chosen learning rate
    folds/<initialisation>_seed<s>_fold<k>.json     search of the learning rate in every fold (finished folds are
                                                    skipped when the script is run again)
    folds/<initialisation>_seed<s>_fold<k>_network.pt
                                                    the tested network of every fold: weights of
                                                    unet_models.RadiusRegressor and the input transform of the fold
    measured_radii.json, summary.csv, seeds.csv, comparisons.csv, results.md

Usage
    python einstein_radius.py --measured modeling_lens_mass.csv
    python einstein_radius.py --measured modeling_lens_mass.csv --seeds 1 2 3 --out-name einstein_radius_real_10k \
        --networks radius-mini=results/simulations/radius-mini/best_unet.pt \
                   radius-mini-10k=results/simulations/radius-mini-10k/best_unet.pt
    python einstein_radius.py --measured modeling_lens_mass.csv --quick       # 2 learning rates, 3 folds, 3 epochs
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr, wilcoxon
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Dataset

from data import (RESULTS_FOLDER, asinh_transform, load_stamps_in_simulation_units, read_benchmark_table,
                  set_random_seeds)
from model_metrics import arc_pixel_radius
from unet_models import UNet, build_regressor, choose_device, load_saved_network

SIMULATION_PIXEL_SCALE = 0.101          # arcseconds per pixel of the simulated images
STAMP_PIXEL_SCALE = 0.1                 # arcseconds per pixel of the real stamps
BENCHMARK = "EUCLID-SAGE-1k-R_VIS"      # all benchmarks contain the same 487 lenses
RESAMPLES = 2000                        # bootstrap resamples for the 95% intervals
BOOTSTRAP_SEED = 20261008               # one generator with this seed is shared by all bootstrap intervals
NMAD_FACTOR = 1.4826                    # makes the NMAD equal to the standard deviation for a normal distribution


# ===================================================================================================== 1. data

def uncertainty_columns(radius_column):
    """Names of the columns with the 1-sigma bounds of the radius column (naming of PyAutoLens), or None."""
    for suffix in ("_median_pdf", "_max_lh"):
        if radius_column.endswith(suffix):
            base = radius_column[: -len(suffix)]
            return base + "_lower_1_sigma", base + "_upper_1_sigma"
    return None


def lenses_with_measured_radius(measured_path, id_column, radius_column, min_radius):
    """The lenses of the benchmarks that have a valid measured Einstein radius, and statistics of the measured radii."""
    # The lenses (label 1) of one benchmark table; the non-lenses have no Einstein radius.
    lenses = read_benchmark_table(BENCHMARK)
    lenses = lenses[lenses.label == 1][["stamp_file", "object_id", "lens_id", "grade"]]
    measured = pd.read_csv(measured_path)
    if id_column not in measured or radius_column not in measured:
        raise SystemExit(f"{measured_path} needs the columns {id_column} and {radius_column}; it has "
                         f"{', '.join(measured.columns)}")
    # Keep the radius column and, if the file has them, its 1-sigma bounds; one model per lens.
    bounds = uncertainty_columns(radius_column)
    has_bounds = bounds is not None and all(bound in measured for bound in bounds)
    columns = [id_column, radius_column] + (list(bounds) if has_bounds else [])
    measured = measured[columns].dropna(subset=[radius_column]).drop_duplicates(id_column)
    measured = measured.rename(columns={id_column: "lens_id", radius_column: "measured_radius"})
    # Only lenses with a model remain after the join; then the radii below min_radius are left out.
    joined = lenses.merge(measured, on="lens_id", how="inner").reset_index(drop=True)
    too_small = joined.measured_radius < min_radius
    valid = joined[~too_small].reset_index(drop=True)

    # Numbers of lenses, quantiles of the measured radii, and the fractions of small (< 0.6") and large (> 2") radii.
    statistics = {"radii_in_file": int(len(measured)), "lenses_matched": int(len(joined)),
                  "lenses_below_min_radius": int(too_small.sum()), "min_radius": min_radius,
                  "lenses_used": int(len(valid)), "grade_A": int((valid.grade == "A").sum()),
                  "grade_B": int((valid.grade == "B").sum()),
                  "quantiles": {f"{quantile:g}": round(float(valid.measured_radius.quantile(quantile)), 3)
                                for quantile in (0, 0.05, 0.25, 0.5, 0.75, 0.95, 1)},
                  "fraction_below_0.6": round(float((valid.measured_radius < 0.6).mean()), 3),
                  "fraction_above_2.0": round(float((valid.measured_radius > 2.0).mean()), 3)}
    if len(columns) == 4:
        # The bounds are either the radii at -1 and +1 sigma, or the errors below and above the radius. If more
        # than 90% of the radii lie between the two bounds, the bounds are taken as the radii at -1 and +1 sigma.
        lower, upper = valid[columns[2]].to_numpy(float), valid[columns[3]].to_numpy(float)
        radius = valid.measured_radius.to_numpy(float)
        bounds_are_values = np.mean((lower <= radius) & (radius <= upper)) > 0.9
        half_width = (upper - lower) / 2 if bounds_are_values else (np.abs(lower) + np.abs(upper)) / 2
        statistics["one_sigma_half_width_median"] = round(float(np.nanmedian(half_width)), 4)
        statistics["one_sigma_relative_median"] = round(float(np.nanmedian(half_width / radius)), 4)
        statistics["one_sigma_bounds"] = "values" if bounds_are_values else "errors"
    print(f"{len(joined)} of {len(lenses)} lenses have a measured Einstein radius ({len(measured)} radii in "
          f"{measured_path.name}); {int(too_small.sum())} below {min_radius}\" left out; {len(valid)} used "
          f"({statistics['grade_A']} grade A, {statistics['grade_B']} grade B)", flush=True)
    if len(valid) < 20:                 # usually a sign that the identifiers of the two tables do not match
        raise SystemExit("Too few lenses with a measured radius; check the identifier column.")
    return valid, statistics


class RadiusDataset(Dataset):
    """asinh-transformed stamps, standardised with the given mean and standard deviation, with their measured radii.
    With apply_symmetries=True, every stamp receives a random rotation and reflection."""

    def __init__(self, transformed_stamps, radii, mean, standard_deviation, apply_symmetries):
        self.transformed_stamps, self.radii = transformed_stamps, radii.astype(np.float32)
        # The lower limit of the standard deviation only prevents a division by zero.
        self.mean, self.standard_deviation = float(mean), max(float(standard_deviation), 1e-8)
        self.apply_symmetries = apply_symmetries

    def __len__(self):
        return len(self.radii)

    def __getitem__(self, index):
        image = self.transformed_stamps[index]
        if self.apply_symmetries:
            image = np.rot90(image, np.random.randint(4))  # rotation by 0, 90, 180 or 270 degrees
            if np.random.rand() < 0.5:                     # reflection with probability 1/2
                image = np.fliplr(image)
        image = (np.ascontiguousarray(image) - self.mean) / self.standard_deviation
        # image[None] adds the channel axis: the network expects images of shape (1, 101, 101).
        return torch.from_numpy(image[None].astype(np.float32)), torch.tensor(self.radii[index])


# ================================================================================= 2. networks applied unchanged

@torch.no_grad()
def apply_network(saved_network, transformed_stamps, device):
    """Estimated radii (converted to the pixel scale of the real stamps) and arc probabilities of a saved network."""
    network = UNet(**saved_network["architecture"])
    network.load_state_dict(saved_network["model_state_dict"])
    network.to(device).eval()
    # Standardisation with the mean and standard deviation of the simulated training images of the network, as in
    # its training. The stamps are processed in batches of 32.
    preprocessing = saved_network["preprocessing"]
    radii, probabilities = [], []
    for start in range(0, len(transformed_stamps), 32):
        batch = (transformed_stamps[start:start + 32] - preprocessing["mean"]) / preprocessing["std"]
        logits, radius = network(torch.from_numpy(batch[:, None].astype(np.float32)).to(device),
                                 return_einstein_radius=True)
        radii.append(radius.cpu().numpy())
        probabilities.append(torch.sigmoid(logits)[:, 0].cpu().numpy())     # arc probability of every pixel
    # The network gives the radius on the pixel scale of the simulations; see the description at the top.
    return np.concatenate(radii) * STAMP_PIXEL_SCALE / SIMULATION_PIXEL_SCALE, np.concatenate(probabilities)


# ======================================================================================== 3. fine-tuning

@torch.no_grad()
def mean_absolute_error(network, loader, device):
    """Mean absolute error of the estimated radii over a loader, and the estimates."""
    network.eval()                      # no dropout during evaluation
    errors, estimates = [], []
    for images, radii in loader:
        estimate = network(images.to(device)).cpu()
        errors.append((estimate - radii).abs())
        estimates.append(estimate)
    return float(torch.cat(errors).mean()), torch.cat(estimates).numpy()


def train_one_rate(network, training_loader, validation_loader, learning_rate, settings, device):
    """Train for settings.epochs epochs and keep the weights of the epoch with the lowest validation error."""
    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=settings.weight_decay)
    # The learning rate decreases from its initial value to zero along half a cosine period over all epochs.
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(settings.epochs, 1))
    loss_function = nn.L1Loss()                     # mean absolute error of the radius over a batch
    best = {"validation_error": float("inf"), "best_epoch": 0}
    best_weights = copy.deepcopy(network.state_dict())
    for epoch in range(1, settings.epochs + 1):
        network.train()
        for images, radii in training_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(network(images.to(device)), radii.to(device))
            loss.backward()
            optimizer.step()
        schedule.step()
        # After every epoch: the error on the validation lenses; the weights of the best epoch are kept.
        validation_error, _ = mean_absolute_error(network, validation_loader, device)
        if validation_error < best["validation_error"]:
            best = {"validation_error": validation_error, "best_epoch": epoch}
            best_weights = copy.deepcopy(network.state_dict())
    network.load_state_dict(best_weights)
    return best


def run_fold(fold, training_rows, test_rows, transformed, radii, saved_network, initialisation, seed, settings, device,
             network_path):
    """Choice of the learning rate on the validation lenses of one fold, then one test on its test lenses. The tested
    network is saved to network_path."""
    pretrained = initialisation != "random"
    # 80% of the training lenses of the fold for training, 20% for validation. The mean and standard deviation for
    # the standardisation come from the training lenses only, so that the validation and test lenses do not
    # influence the training.
    fit_rows, validation_rows = train_test_split(training_rows, test_size=0.2, random_state=settings.split_seed + fold)
    mean, standard_deviation = transformed[fit_rows].mean(), transformed[fit_rows].std()

    def loader(rows, augment, shuffle):
        """Batches of the given lenses; augment: random rotations and reflections; shuffle: random order."""
        return DataLoader(RadiusDataset(transformed[rows], radii[rows], mean, standard_deviation, augment),
                          batch_size=settings.batch_size, shuffle=shuffle)

    # One network per learning rate. The seed is set again before each network, so that all learning rates start
    # from the same random weights and see the same random order of the training lenses.
    candidates = []
    for learning_rate in settings.learning_rates:
        set_random_seeds(1000 * seed + fold)
        network = build_regressor(saved_network, pretrained).to(device)
        best = train_one_rate(network, loader(fit_rows, True, True), loader(validation_rows, False, False),
                              learning_rate, settings, device)
        candidates.append({"learning_rate": learning_rate, **best, "network": network})
        print(f"      fold {fold}, learning rate {learning_rate:g}: best epoch {best['best_epoch']}, validation "
              f"error {best['validation_error']:.4f}\"", flush=True)
    # The learning rate with the lowest validation error; for errors equal to four decimals, the lower rate.
    # Only the chosen network is applied to the test lenses.
    chosen = min(candidates, key=lambda candidate: (round(candidate["validation_error"], 4),
                                                    candidate["learning_rate"]))
    test_error, estimates = mean_absolute_error(chosen["network"], loader(test_rows, False, False), device)
    # The tested network, with the input transform of the fold, so that it can be applied again.
    torch.save({"model_state_dict": chosen["network"].cpu().state_dict(),
                "architecture": saved_network["architecture"], "initialisation": initialisation, "seed": seed,
                "fold": fold, "learning_rate": chosen["learning_rate"], "best_epoch": chosen["best_epoch"],
                "preprocessing": {"asinh_scale": float(saved_network["preprocessing"]["asinh_scale"]),
                                  "mean": float(mean), "std": float(standard_deviation)}},
               network_path)
    # Record of the fold, saved as JSON by fine_tune.
    return {"fold": fold, "seed": seed, "chosen_learning_rate": chosen["learning_rate"],
            "chosen_best_epoch": chosen["best_epoch"], "chosen_validation_error": chosen["validation_error"],
            "candidates": [{key: value for key, value in candidate.items() if key != "network"}
                           for candidate in candidates],
            "test_rows": [int(row) for row in test_rows], "test_estimates": [float(estimate) for estimate in estimates],
            "test_error": test_error}


def fine_tune(lenses, transformed, saved_network, initialisation, seed, settings, device, output_folder):
    """Estimates for all lenses, each from the network of the fold in which the lens was a test lens, for one
    initialisation ("random" or "pretrained-<name>") and one training seed."""
    radii = lenses.measured_radius.to_numpy(dtype=float)
    fold_folder = output_folder / "folds"
    fold_folder.mkdir(parents=True, exist_ok=True)
    estimates = np.full(len(lenses), np.nan)       # estimate of every lens, from the fold in which it is tested
    chosen = np.full(len(lenses), np.nan)          # learning rate chosen in that fold
    folds = np.zeros(len(lenses), dtype=int)       # number of that fold
    # The same folds for all initialisations and training seeds (split seed 42), so that the estimates of different
    # initialisations can be compared lens by lens.
    splitter = KFold(n_splits=settings.folds, shuffle=True, random_state=settings.split_seed)
    for fold, (training_rows, test_rows) in enumerate(splitter.split(radii), start=1):
        path = fold_folder / f"{initialisation}_seed{seed}_fold{fold}.json"
        network_path = fold_folder / f"{initialisation}_seed{seed}_fold{fold}_network.pt"
        if path.exists():                            # a fold finished in an earlier, interrupted run
            record = json.loads(path.read_text())
            note = "" if network_path.exists() else " (no saved network)"
            print(f"    fold {fold}: finished earlier{note}", flush=True)
        else:
            start = time.time()
            record = run_fold(fold, training_rows, test_rows, transformed, radii, saved_network, initialisation, seed,
                              settings, device, network_path)
            record["init"] = initialisation          # key name kept, so that earlier fold records can be resumed
            record["runtime_minutes"] = round((time.time() - start) / 60, 1)
            path.write_text(json.dumps(record, indent=1))
            print(f"    fold {fold}: learning rate {record['chosen_learning_rate']:g}, best epoch "
                  f"{record['chosen_best_epoch']}, test error {record['test_error']:.3f}\", "
                  f"{record['runtime_minutes']:.1f} min", flush=True)
        rows = np.array(record["test_rows"])
        estimates[rows], chosen[rows], folds[rows] = record["test_estimates"], record["chosen_learning_rate"], fold
    table = lenses[["lens_id", "object_id", "grade", "measured_radius"]].copy()
    table["estimated_radius"], table["fold"], table["learning_rate"] = estimates.round(4), folds, chosen
    table.to_csv(output_folder / f"finetuned_{initialisation}_seed{seed}.csv", index=False)
    return estimates


# ============================================================================================== 4. metrics

def error_summary(name, estimates, measured, grades, generator):
    """The metrics of one set of estimated radii (see the description at the top)."""
    estimates = np.asarray(estimates, dtype=float)
    error = estimates - measured                     # positive: the radius is overestimated
    absolute = np.abs(error)
    valid = np.isfinite(absolute)                    # lenses without an estimate (NaN) are left out
    # Bootstrap: the lenses are drawn with replacement, and the mean absolute error is computed for every resample.
    resampled = [absolute[valid][generator.integers(0, valid.sum(), valid.sum())].mean() for _ in range(RESAMPLES)]
    row = {"estimate": name, "lenses": int(valid.sum()), "mean_absolute_error": float(absolute[valid].mean()),
           "mae_ci_low": float(np.percentile(resampled, 2.5)), "mae_ci_high": float(np.percentile(resampled, 97.5)),
           "median_absolute_error": float(np.median(absolute[valid])),
           "rmse": float(np.sqrt(np.mean(error[valid] ** 2))),
           "nmad": float(NMAD_FACTOR * np.median(np.abs(error[valid] - np.median(error[valid])))),
           "mean_error": float(error[valid].mean()),
           "median_relative_error": float(np.median(absolute[valid] / measured[valid])),
           "spearman": float(spearmanr(estimates[valid], measured[valid])[0])}
    for grade in ("A", "B"):
        selected = valid & (grades == grade)
        row[f"mae_grade_{grade}"] = float(absolute[selected].mean()) if selected.any() else float("nan")
    return {key: (round(value, 4) if isinstance(value, float) else value) for key, value in row.items()}


SEED_METRICS = ["mean_absolute_error", "median_absolute_error", "rmse", "nmad", "mean_error", "median_relative_error",
                "spearman"]


def seed_spread(name, rows):
    """Mean and standard deviation of the metrics over the training seeds."""
    table = pd.DataFrame(rows)
    result = {"estimate": name, "seeds": len(rows)}
    for metric in SEED_METRICS:
        result[metric + "_mean"] = round(float(table[metric].mean()), 4)
        result[metric + "_std"] = round(float(table[metric].std(ddof=1)), 4) if len(rows) > 1 else 0.0
    return result


def paired_comparison(name, estimates_1, estimates_2, measured, generator):
    """Absolute error of estimate 1 minus that of estimate 2, per lens (negative: estimate 1 is closer)."""
    difference = np.abs(np.asarray(estimates_1) - measured) - np.abs(np.asarray(estimates_2) - measured)
    difference = difference[np.isfinite(difference)]
    # Bootstrap interval of the mean difference: the lenses are drawn with replacement, keeping the pairs together.
    resampled = [difference[generator.integers(0, len(difference), len(difference))].mean() for _ in range(RESAMPLES)]
    return {"comparison": name, "lenses": len(difference), "mean_difference": round(float(difference.mean()), 4),
            "ci_low": round(float(np.percentile(resampled, 2.5)), 4),
            "ci_high": round(float(np.percentile(resampled, 97.5)), 4),
            # The Wilcoxon test is not defined if all differences are zero (identical estimates).
            "wilcoxon_p": float(wilcoxon(difference).pvalue) if np.any(difference != 0) else 1.0}


def write_report(statistics, radius_column, summary, spreads, comparisons, output_folder):
    """results.md: the measured radii, the metrics of all estimates and the paired comparisons."""
    quantiles = statistics["quantiles"]
    lines = ["# Einstein radii of the real lenses", "",
             f"Measured radii: PyAutoLens lens models ({radius_column}). Brackets: 95% bootstrap interval.", "",
             "## Measured radii", "",
             f"{statistics['lenses_matched']} lenses matched ({statistics['radii_in_file']} radii in the file); "
             f"{statistics['lenses_below_min_radius']} below {statistics['min_radius']}\" left out; "
             f"{statistics['lenses_used']} used ({statistics['grade_A']} grade A, {statistics['grade_B']} grade B).",
             "",
             f"Quantiles (0, 5, 25, 50, 75, 95, 100%): {quantiles['0']}\", {quantiles['0.05']}\", "
             f"{quantiles['0.25']}\", {quantiles['0.5']}\", {quantiles['0.75']}\", {quantiles['0.95']}\", "
             f"{quantiles['1']}\". Below 0.6\": {100 * statistics['fraction_below_0.6']:.1f}%; above 2.0\": "
             f"{100 * statistics['fraction_above_2.0']:.1f}%."]
    if "one_sigma_half_width_median" in statistics:
        lines += ["", f"Median 1-sigma half-width of the measured radii: {statistics['one_sigma_half_width_median']}\" "
                      f"({100 * statistics['one_sigma_relative_median']:.1f}% of the radius)."]
    lines += ["", "## Errors", "",
              "| Estimate | Lenses | Mean absolute error | Median absolute error | RMSE | NMAD | Mean error | "
              "Median relative error | Spearman | Grade A | Grade B |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['estimate']} | {row['lenses']} | {row['mean_absolute_error']:.3f}\" "
                     f"[{row['mae_ci_low']:.3f}, {row['mae_ci_high']:.3f}] | {row['median_absolute_error']:.3f}\" | "
                     f"{row['rmse']:.3f}\" | {row['nmad']:.3f}\" | {row['mean_error']:+.3f}\" | "
                     f"{100 * row['median_relative_error']:.0f}% | {row['spearman']:.2f} | "
                     f"{row['mae_grade_A']:.3f}\" | {row['mae_grade_B']:.3f}\" |")
    if spreads:
        lines += ["", "## Mean and standard deviation over training seeds", "",
                  "| Initialisation | Seeds | Mean absolute error | Median absolute error | RMSE | NMAD | "
                  "Median relative error | Spearman |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in spreads:
            lines.append(
                f"| {row['estimate']} | {row['seeds']} | {row['mean_absolute_error_mean']:.3f} ± "
                f"{row['mean_absolute_error_std']:.3f}\" | {row['median_absolute_error_mean']:.3f} ± "
                f"{row['median_absolute_error_std']:.3f}\" | {row['rmse_mean']:.3f} ± {row['rmse_std']:.3f}\" | "
                f"{row['nmad_mean']:.3f} ± {row['nmad_std']:.3f}\" | {100 * row['median_relative_error_mean']:.0f} ± "
                f"{100 * row['median_relative_error_std']:.0f}% | {row['spearman_mean']:.2f} ± "
                f"{row['spearman_std']:.2f} |")
    lines += ["", "## Paired comparisons", "", "| Comparison (absolute error of 1 - 2) | Lenses | Mean difference | "
              "Wilcoxon p |", "|---|---:|---:|---:|"]
    for comparison in comparisons:
        lines.append(f"| {comparison['comparison']} | {comparison['lenses']} | {comparison['mean_difference']:+.3f}\" "
                     f"[{comparison['ci_low']:+.3f}, {comparison['ci_high']:+.3f}] | {comparison['wilcoxon_p']:.2g} |")
    (output_folder / "results.md").write_text("\n".join(lines) + "\n")


# ========================================================================================= 5. command line

def parse_networks(entries):
    """NAME=PATH entries; without NAME=, the name of the folder of the network file is used."""
    networks = {}
    for entry in entries:
        name, _, path = entry.partition("=") if "=" in entry else (Path(entry).parent.name, "", entry)
        networks[name] = Path(path)
    return networks


def main():
    argument_parser = argparse.ArgumentParser(description="Einstein radii of the real lenses with UNet-Radius-mini")
    argument_parser.add_argument("--measured", type=Path, required=True,
                                 help="modeling_lens_mass.csv, the lens models of the Q1 Strong Lensing "
                                      "Discovery Engine (Zenodo)")
    argument_parser.add_argument("--networks", nargs="+", default=None,
                                 help="NAME=PATH of UNet-Radius-mini networks trained on simulations "
                                      "(default radius-mini=results/simulations/radius-mini/best_unet.pt)")
    argument_parser.add_argument("--id-column", default="id_str", help="column of the lens identifier in --measured")
    argument_parser.add_argument("--radius-column", default="einstein_radius_effective_median_pdf",
                                 help="column of the measured Einstein radius (arcseconds) in --measured")
    argument_parser.add_argument("--min-radius", type=float, default=0.05,
                                 help="smallest measured radius used (default 0.05\")")
    argument_parser.add_argument("--learning-rates", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3, 3e-3],
                                 help="grid of initial learning rates searched in every fold")
    argument_parser.add_argument("--epochs", type=int, default=60, help="training epochs per learning rate")
    argument_parser.add_argument("--batch-size", type=int, default=16, help="stamps per batch")
    argument_parser.add_argument("--weight-decay", type=float, default=1e-4, help="weight decay of AdamW")
    argument_parser.add_argument("--folds", type=int, default=5, help="number of cross-validation folds")
    argument_parser.add_argument("--split-seed", type=int, default=42,
                                 help="seed of the folds and of the training-validation splits")
    argument_parser.add_argument("--seeds", nargs="+", type=int, default=[1], help="training seeds")
    argument_parser.add_argument("--no-random", action="store_true", help="leave out the random initialisation")
    argument_parser.add_argument("--out-name", default="einstein_radius_real", help="name of the results folder")
    argument_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                                 help="auto: a GPU if one is available, otherwise the CPU")
    argument_parser.add_argument("--quick", action="store_true", help="2 learning rates, 3 folds and 3 epochs")
    settings = argument_parser.parse_args()
    if settings.quick:
        settings.learning_rates, settings.folds, settings.epochs = [3e-4, 1e-3], 3, 3
    # The networks trained on simulations; only UNet-Radius-mini has the Einstein radius output.
    networks = parse_networks(settings.networks or
                              [f"radius-mini={RESULTS_FOLDER / 'simulations' / 'radius-mini' / 'best_unet.pt'}"])
    saved_networks = {}
    for name, path in networks.items():
        saved_networks[name] = load_saved_network(path)
        if not saved_networks[name]["architecture"].get("einstein_radius", False):
            raise SystemExit(f"{path} has no Einstein radius output (UNet-Radius-mini expected)")
    first_network = next(iter(networks))    # its architecture and input transform are used for random weights
    device = choose_device(settings.device)
    output_folder = RESULTS_FOLDER / (settings.out_name + ("_quick" if settings.quick else ""))
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"Networks: {', '.join(f'{name} ({path})' for name, path in networks.items())}; device {device}; results "
          f"in {output_folder}", flush=True)

    lenses, statistics = lenses_with_measured_radius(settings.measured, settings.id_column, settings.radius_column,
                                                     settings.min_radius)
    (output_folder / "measured_radii.json").write_text(json.dumps(statistics, indent=1))
    print(f"  measured radii: {statistics['quantiles']}", flush=True)
    # The real stamps in the units of the simulations (data.py), then the asinh transform of each network.
    stamps = load_stamps_in_simulation_units(lenses)
    transformed = {name: asinh_transform(stamps, float(saved["preprocessing"]["asinh_scale"]))
                   for name, saved in saved_networks.items()}
    measured = lenses.measured_radius.to_numpy(dtype=float)
    grades = lenses.grade.to_numpy()
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    # summary: metrics of every estimate; spreads: spread over training seeds; comparisons: paired comparisons;
    # applied: estimates of the networks applied without fine-tuning.
    summary, spreads, comparisons, applied = [], [], [], {}

    print("\n1. Networks applied without training on real stamps", flush=True)
    for name, saved in saved_networks.items():
        applied[name], probabilities = apply_network(saved, transformed[name], device)
        arc_radius = arc_pixel_radius(probabilities, float(saved.get("threshold", 0.5)), STAMP_PIXEL_SCALE)
        table = lenses[["lens_id", "object_id", "grade", "measured_radius"]].copy()
        table["estimated_radius"], table["median_arc_radius"] = applied[name].round(4), arc_radius.round(4)
        table.to_csv(output_folder / f"applied_{name}.csv", index=False)
        summary.append(error_summary(f"applied {name}", applied[name], measured, grades, generator))
        summary.append(error_summary(f"median arc radius, applied {name}", arc_radius, measured, grades, generator))
        print(f"  {name}: mean absolute error {summary[-2]['mean_absolute_error']:.3f}\", mean error "
              f"{summary[-2]['mean_error']:+.3f}\" (median arc radius: {summary[-1]['mean_absolute_error']:.3f}\")",
              flush=True)

    print("\n2. Networks fine-tuned on the measured radii", flush=True)
    # One pretrained initialisation per network, and one with random weights.
    initialisations = [f"pretrained-{name}" for name in networks] + ([] if settings.no_random else ["random"])
    finetuned = {}                                  # estimates for every (initialisation, training seed)
    for seed in settings.seeds:
        for initialisation in initialisations:
            name = initialisation[len("pretrained-"):] if initialisation != "random" else first_network
            print(f"  {initialisation}, seed {seed}", flush=True)
            finetuned[(initialisation, seed)] = fine_tune(lenses, transformed[name], saved_networks[name],
                                                          initialisation, seed, settings, device, output_folder)
            summary.append(error_summary(f"fine-tuned {initialisation}, seed {seed}", finetuned[(initialisation, seed)],
                                         measured, grades, generator))
            print(f"  {initialisation}, seed {seed}: mean absolute error {summary[-1]['mean_absolute_error']:.3f}\", "
                  f"median {summary[-1]['median_absolute_error']:.3f}\"", flush=True)
    # With several training seeds: the spread of the metrics over the seeds, and the metrics of the mean estimate.
    ensembles = {}
    if len(settings.seeds) > 1:
        for initialisation in initialisations:
            rows = [row for row in summary if row["estimate"].startswith(f"fine-tuned {initialisation}, seed ")]
            spreads.append(seed_spread(f"fine-tuned {initialisation}", rows))
            ensembles[initialisation] = np.mean([finetuned[(initialisation, seed)] for seed in settings.seeds], axis=0)
            summary.append(error_summary(f"fine-tuned {initialisation}, mean of seeds "
                                         f"{', '.join(map(str, settings.seeds))}",
                                         ensembles[initialisation], measured, grades, generator))

    # Paired comparisons (see the description at the top), for every training seed and for the mean over seeds.
    estimate_sets = [(f"seed {seed}", {initialisation: finetuned[(initialisation, seed)]
                                       for initialisation in initialisations}) for seed in settings.seeds]
    if ensembles:
        estimate_sets.append(("mean of seeds", ensembles))
    pretrained_initialisations = [initialisation for initialisation in initialisations if initialisation != "random"]
    for label, estimates in estimate_sets:
        for initialisation in pretrained_initialisations:
            if "random" in estimates:
                comparisons.append(paired_comparison(f"fine-tuned {initialisation} - fine-tuned random, {label}",
                                                     estimates[initialisation], estimates["random"], measured,
                                                     generator))
        for index, later in enumerate(pretrained_initialisations):
            for earlier in pretrained_initialisations[:index]:
                comparisons.append(paired_comparison(f"fine-tuned {later} - fine-tuned {earlier}, {label}",
                                                     estimates[later], estimates[earlier], measured, generator))
        for initialisation in pretrained_initialisations:
            name = initialisation[len("pretrained-"):]
            comparisons.append(paired_comparison(f"fine-tuned {initialisation}, {label} - applied {name}",
                                                 estimates[initialisation], applied[name], measured, generator))
    # The networks applied without fine-tuning, against each other.
    names = list(networks)
    for index, later in enumerate(names):
        for earlier in names[:index]:
            comparisons.append(paired_comparison(f"applied {later} - applied {earlier}", applied[later],
                                                 applied[earlier], measured, generator))

    pd.DataFrame(summary).to_csv(output_folder / "summary.csv", index=False)
    if spreads:
        pd.DataFrame(spreads).to_csv(output_folder / "seeds.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output_folder / "comparisons.csv", index=False)
    write_report(statistics, settings.radius_column, summary, spreads, comparisons, output_folder)
    print(f"\n{(output_folder / 'results.md').read_text()}\nResults saved in {output_folder}", flush=True)


if __name__ == "__main__":
    main()
