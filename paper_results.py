"""paper_results.py - the tables of the Results section of the paper, computed from the saved test predictions

UNet-Starlight.

The script reads the outputs of train_simulations.py, train_benchmarks.py and einstein_radius.py and writes the
tables of the paper. No network is trained or run, and no weights are needed.

Check of the recorded metrics
    Every metric of the simulated benchmark and of the 1k benchmarks is computed again from the saved test
    predictions, with the functions used during training (model_metrics.py), and compared with the value written
    during training. A difference larger than 1e-6 stops the script. Each test stamp was predicted once, by the
    network chosen without it.

Statistics (post_training_statistics.py)
    - 95% intervals of the AUROC and of the completeness at a 5% false-alarm rate: stratified bootstrap over the 1,000
      test stamps of a benchmark, 2,000 resamples. All models of a benchmark use the same resamples, so that their
      differences can be resampled as pairs.
    - Paired comparisons of two networks tested on the same stamps (pretrained against random initialisation; the
      smaller against the larger model): difference with a bootstrap interval and bootstrap p-value, and the DeLong
      test of the AUROC difference. The DeLong p-values are adjusted with the method of Holm within each of the two
      sets of comparisons.
    - Completeness at a 5% false-alarm rate (C5%): the fraction of lenses scored above the threshold that at most 25
      of the 513 non-lenses exceed. At 1% the threshold would rest on 5 non-lenses, and the estimate would be too
      unstable to compare models.
    - Simulated benchmark: interval of the mean lens Dice from a bootstrap over the 100 test lenses.
    - Training seeds: every seed is reported, and the mean and standard deviation over seeds give the variation
      between training runs, which the bootstrap intervals do not include.

Tables and the parts of the Results section
    simulations.csv            Simulated benchmark; Einstein radius with UNet-Radius-mini (simulated test lenses)
    seeds.csv                  Pretraining on simulations: AUROC and C5%, mean and standard deviation over seeds
    differences.csv            Pretraining on simulations: pretrained - random, per model, benchmark and seed
    size_differences.csv       Model size: smaller - larger model, per benchmark, initialisation and seed
    seed_differences.csv       Pretraining and model size: the differences, mean and standard deviation over seeds
    einstein_radius_real.csv   Einstein radius with UNet-Radius-mini (real lenses), from the outputs of
                               einstein_radius.py
    benchmarks.csv             all metrics of every model, benchmark, initialisation and seed, behind the means of
                               seeds.csv
    false_alarms_10k.csv       a test, not used in the paper (see below)
    results.md                 all tables in Markdown

False alarms on the 10k benchmarks (a test, not used in the paper)
    The paper uses only the 1k benchmarks; the 10k benchmarks are left for a later paper, in which the networks will
    also be trained on them. If results/predictions_10k/ exists (written by a separate test script, not part of this
    repository), the false-alarm rates of the saved networks on the 9,000 further non-lenses of the 10k benchmarks
    are collected, with the precision expected at the survey lens fractions.

Inputs (results/)
    simulations/<model>/test_predictions.npz, summary.json                  train_simulations.py
    benchmarks/<model>/<benchmark>/<initialisation>/seed<s>/                train_benchmarks.py
        predictions_per_stamp.csv (lens probability, decision and fold of every stamp), summary.json
    <--radius-results>/summary.csv, seeds.csv, comparisons.csv, measured_radii.json         einstein_radius.py
    Models, benchmarks and folders without results are skipped.

Outputs: results/tables/ (results/tables_quick/ with --quick)

Usage
    python paper_results.py
    python paper_results.py --quick          # the results of the --quick runs of the other scripts
"""

import argparse
import json

import numpy as np
import pandas as pd

import post_training_statistics as statistics
from data import BENCHMARKS, RESULTS_FOLDER
from model_metrics import (LENS_FRACTIONS_FOR_PRECISION, area_under_roc, benchmark_metrics, classification_metrics,
                           einstein_radius_metrics, segmentation_metrics)
from unet_models import EARLIER_KEYS, FILTERS, NAMES, model_key

INITIALISATIONS = ["pretrained", "random", "frozen"]     # frozen: initial experiments only, not in the paper
TOLERANCE = 1e-6                                         # largest allowed difference to a recorded metric
FALSE_ALARM_RATE = 0.05                                  # operating point of the completeness C5%
SIZE_PAIRS = [("starlight-mini", "starlight"), ("starlight-tiny", "starlight"),
              ("starlight-tiny", "starlight-mini")]                                        # smaller - larger
# Descriptions of the UNet-Radius-mini networks trained on simulations, as used in the paper (with their earlier names)
RADIUS_NETWORK_DESCRIPTIONS = {"radius-mini": "2,000 simulated images", "radius-mini-10k": "second simulated set",
                               "sparkle-er": "2,000 simulated images", "sparkle-er-10k": "second simulated set"}


def results_folder(parent, model, suffix):
    """The results folder of a model: parent/<key><suffix>, or the folder with the earlier key of the model if only
    that one exists."""
    folder = parent / (model + suffix)
    earlier_folders = [parent / (earlier + suffix) for earlier, key in EARLIER_KEYS.items() if key == model]
    if not folder.exists():
        for earlier_folder in earlier_folders:
            if earlier_folder.exists():
                return earlier_folder
    return folder


def completeness_at_operating_point(labels, scores):
    """Completeness at the false-alarm rate FALSE_ALARM_RATE (C5%)."""
    return statistics.completeness_at_false_alarm_rate(labels, scores, FALSE_ALARM_RATE)


# ============================================================================== 1. check of the recorded metrics

def agrees(recomputed, recorded):
    """True if a recomputed value agrees with the value recorded during training (lists must be equal, numbers
    within TOLERANCE)."""
    if isinstance(recomputed, (list, tuple)):
        return list(recomputed) == list(recorded)
    if recomputed is None or recorded is None:
        return recomputed is None and recorded is None
    return bool(np.isclose(recomputed, recorded, rtol=0, atol=TOLERANCE, equal_nan=True))


def check_recorded_metrics(recomputed, recorded, source):
    """Stop if a recomputed metric differs from the recorded one."""
    differing = [key for key in recomputed if key in recorded and not agrees(recomputed[key], recorded[key])]
    if differing:
        raise SystemExit(f"{source}: recomputed metrics differ from the recorded ones: {', '.join(differing)}")


# ======================================================================================= 2. simulated benchmark

def lens_dice_per_image(probabilities, true_masks, labels, threshold):
    """Dice coefficient of the predicted arc mask of every test lens (the formula of model_metrics.py)."""
    is_lens = labels == 1
    predicted, true = probabilities[is_lens] >= threshold, true_masks[is_lens].astype(bool)
    intersection = np.logical_and(predicted, true).sum(axis=(1, 2))
    return (2 * intersection + 1e-8) / (predicted.sum(axis=(1, 2)) + true.sum(axis=(1, 2)) + 1e-8)


def simulation_row(model, folder):
    """Test results of one U-Net on the test images of the simulated benchmark."""
    summary = json.loads((folder / "summary.json").read_text())
    recorded = summary["test_at_requested_threshold"]
    predictions = np.load(folder / "test_predictions.npz")
    metrics = segmentation_metrics(predictions["probability"], predictions["truth_mask"], predictions["label"],
                                   recorded["threshold"], recorded["min_pixels_for_lens"])
    check_recorded_metrics(metrics, recorded, folder / "summary.json")
    # Confusion matrix of the lens decision; rows: true non-lens, true lens; columns: decided non-lens, decided lens.
    (true_nonlenses, false_alarms), (missed, found) = metrics["mask_based_lens_confusion_0_1"]

    # Bootstrap interval of the mean lens Dice: the 100 test lenses are drawn with replacement.
    dice = lens_dice_per_image(predictions["probability"], predictions["truth_mask"], predictions["label"],
                               recorded["threshold"])
    if not agrees(float(dice.mean()), metrics["lens_dice_mean"]):
        raise SystemExit(f"{folder}: the Dice coefficients per image do not reproduce the mean lens Dice")
    generator = np.random.default_rng(statistics.BOOTSTRAP_SEED)
    resampled_means = generator.choice(dice, size=(statistics.RESAMPLES, len(dice)), replace=True).mean(axis=1)
    dice_low, dice_high = statistics.percentile_interval(resampled_means)

    # Einstein radius of the test lenses (UNet-Radius-mini only), and the arc-pixel estimate for comparison.
    radius = {"einstein_radius_mean_absolute_error": None, "einstein_radius_median_absolute_error": None,
              "einstein_radius_mean_error": None, "arc_pixel_radius_mean_absolute_error": None}
    if "estimated_theta_E" in predictions.files:
        radius_metrics = einstein_radius_metrics(predictions["theta_E"], predictions["estimated_theta_E"],
                                                 predictions["probability"], predictions["label"],
                                                 recorded["threshold"])
        check_recorded_metrics(radius_metrics, summary["einstein_radius"], folder / "summary.json")
        radius = {"einstein_radius_mean_absolute_error": round(radius_metrics["mean_absolute_error"], 3),
                  "einstein_radius_median_absolute_error": round(radius_metrics["median_absolute_error"], 3),
                  "einstein_radius_mean_error": round(radius_metrics["mean_error"], 3),
                  "arc_pixel_radius_mean_absolute_error":
                      round(radius_metrics["arc_pixel_estimate_mean_absolute_error"], 3)}
    return {"model": NAMES[model], "parameters": summary["model_parameters"],
            "best_epoch": summary["training"]["best_epoch"],
            "dice": round(metrics["lens_dice_mean"], 3), "dice_ci_low": round(dice_low, 3),
            "dice_ci_high": round(dice_high, 3), "dice_median": round(metrics["lens_dice_median"], 3),
            "iou": round(metrics["lens_iou_mean"], 3),
            "balanced_accuracy": round(metrics["mask_based_lens_balanced_accuracy"], 3),
            "lenses_found": found, "lenses": found + missed,
            "false_alarms": false_alarms, "nonlenses": true_nonlenses + false_alarms,
            "nonlenses_with_empty_mask": round(metrics["nonlens_zero_mask_fraction"], 3),
            "arc_pixels_per_nonlens": round(metrics["nonlens_mean_predicted_pixels"], 2), **radius}


# ============================================================================================ 3. 1k benchmarks

def fold_aurocs(predictions):
    """AUROC of each of the five test folds, in fold order."""
    return [area_under_roc(group.label, group.lens_probability) for _, group in predictions.groupby("fold")]


def benchmark_row(model, benchmark, initialisation, seed_folder, resample_rows):
    """Test results of one classifier (model, benchmark, initialisation, training seed), and its predictions."""
    predictions = pd.read_csv(seed_folder / "predictions_per_stamp.csv")
    if (predictions.fold < 1).any():                # fold -1 marks stamps of a fold that is not finished
        raise SystemExit(f"{seed_folder}: not all folds are finished")
    metrics = {**classification_metrics(predictions.label.to_numpy(), predictions.predicted_lens.to_numpy(),
                                        predictions.lens_probability.to_numpy()),
               **benchmark_metrics(predictions)}
    check_recorded_metrics(metrics, json.loads((seed_folder / "summary.json").read_text())["out_of_fold_results"],
                           seed_folder / "summary.json")
    per_fold = fold_aurocs(predictions)
    labels, scores = predictions.label.to_numpy(), predictions.lens_probability.to_numpy()
    # The AUROC of all 1,000 test predictions together, and C5%, each with its bootstrap interval.
    auroc_low, auroc_high = statistics.percentile_interval(
        statistics.bootstrap_values(statistics.auroc_from_ranks, labels, scores, resample_rows))
    operating_point = completeness_at_operating_point(labels, scores)
    operating_low, operating_high = statistics.percentile_interval(
        statistics.bootstrap_values(completeness_at_operating_point, labels, scores, resample_rows))
    return {"model": NAMES[model], "benchmark": benchmark, "initialisation": initialisation,
            "seed": int(seed_folder.name[len("seed"):]),
            "auroc": round(metrics["auroc"], 4), "auroc_ci_low": round(auroc_low, 4),
            "auroc_ci_high": round(auroc_high, 4),
            "completeness_at_5pct_false_alarms": round(operating_point, 4),
            "completeness_at_5pct_ci_low": round(operating_low, 4),
            "completeness_at_5pct_ci_high": round(operating_high, 4),
            "fold_auroc_mean": round(float(np.mean(per_fold)), 4),
            "fold_auroc_std": round(float(np.std(per_fold, ddof=1)), 4),
            # decision p > 0.5: completeness, false alarms and precision (model_metrics.py)
            **{key: metrics[key] for key in ("completeness", "lenses_found", "lenses", "false_alarm_rate",
                                             "false_alarms", "nonlenses", "completeness_grade_A",
                                             "completeness_grade_B")},
            **{key: value for key, value in metrics.items() if key.startswith("precision_")},
            "balanced_accuracy": round(metrics["balanced_accuracy"], 4)}, predictions


def paired_comparison(predictions_1, predictions_2, resample_rows, description):
    """Difference (1 - 2) of the AUROC and of C5% for two networks tested on the same stamps: values, paired bootstrap
    intervals and p-values, and the DeLong p-value of the AUROC difference."""
    if not predictions_1.stamp_file.equals(predictions_2.stamp_file):
        raise SystemExit(f"{description}: the two networks were not tested on the same stamps")
    labels = predictions_1.label.to_numpy()
    scores_1, scores_2 = predictions_1.lens_probability.to_numpy(), predictions_2.lens_probability.to_numpy()
    result = {}
    for name, metric in (("auroc", statistics.auroc_from_ranks), ("completeness_at_5pct",
                                                                  completeness_at_operating_point)):
        # The same resamples for both networks, so that the difference is computed on every resample.
        differences = (statistics.bootstrap_values(metric, labels, scores_1, resample_rows)
                       - statistics.bootstrap_values(metric, labels, scores_2, resample_rows))
        low, high = statistics.percentile_interval(differences)
        result.update({f"{name}_difference": round(metric(labels, scores_1) - metric(labels, scores_2), 4),
                       f"{name}_difference_ci_low": round(low, 4), f"{name}_difference_ci_high": round(high, 4),
                       f"{name}_bootstrap_p": round(statistics.bootstrap_p_value(differences), 4)})
    _, standard_error, delong_p = statistics.delong_test(labels, scores_1, scores_2)
    result.update({"auroc_difference_standard_error": round(standard_error, 4), "auroc_delong_p": delong_p})
    return result


def pretraining_row(model, benchmark, seed, predictions_pretrained, predictions_random, resample_rows):
    """Pretrained - random initialisation for one model, benchmark and seed, on the same stamps and folds."""
    if not predictions_pretrained.fold.equals(predictions_random.fold):
        raise SystemExit(f"{NAMES[model]}, {benchmark}: pretrained and random networks were not tested on the same "
                         f"folds")
    differences = np.array(fold_aurocs(predictions_pretrained)) - np.array(fold_aurocs(predictions_random))
    auroc_pretrained = area_under_roc(predictions_pretrained.label, predictions_pretrained.lens_probability)
    auroc_random = area_under_roc(predictions_random.label, predictions_random.lens_probability)
    return {"model": NAMES[model], "benchmark": benchmark, "seed": seed,
            "auroc_pretrained": round(auroc_pretrained, 4), "auroc_random": round(auroc_random, 4),
            **paired_comparison(predictions_pretrained, predictions_random, resample_rows,
                                f"{NAMES[model]}, {benchmark}"),
            # The AUROC difference in each of the five folds: its mean and spread, and in how many folds the
            # pretrained network is better.
            "fold_difference_mean": round(float(differences.mean()), 4),
            "fold_difference_std": round(float(differences.std(ddof=1)), 4),
            "folds_pretrained_better": int((differences > 0).sum()), "folds": len(differences)}


def add_holm_adjusted_p(rows, p_column="auroc_delong_p"):
    """Holm-adjusted DeLong p-values within one set of comparisons."""
    if rows:
        for row, adjusted in zip(rows, statistics.holm_adjusted([row[p_column] for row in rows])):
            row["auroc_p_holm"] = float(adjusted)


SEED_COLUMNS = ["auroc", "completeness_at_5pct_false_alarms", "completeness", "false_alarms", "false_alarm_rate",
                "balanced_accuracy"]


def mean_and_spread(table, keys, columns):
    """Mean, standard deviation (NaN for a single seed) and number of seeds of the columns, per group of keys."""
    rows = []
    for group_keys, group in table.groupby(keys, sort=False):
        row = dict(zip(keys, group_keys))
        row["seeds"] = len(group)
        for column in columns:
            row[f"{column}_mean"] = round(float(group[column].mean()), 4)
            row[f"{column}_std"] = round(float(group[column].std(ddof=1)), 4) if len(group) > 1 else float("nan")
        rows.append(row)
    return rows


# ================================================================================ 4. Einstein radius, real lenses

def real_lens_radius_rows(folder):
    """The errors of the estimated Einstein radii of the real lenses (outputs of einstein_radius.py): the networks
    applied without fine-tuning, and the fine-tuned networks (mean and standard deviation over training seeds).
    Also returns the paired comparisons and the number of lenses."""
    summary = pd.read_csv(folder / "summary.csv")
    lenses = json.loads((folder / "measured_radii.json").read_text())["lenses_used"]

    def description(name):
        return RADIUS_NETWORK_DESCRIPTIONS.get(name, name)

    rows = []
    for _, row in summary[summary.estimate.str.startswith("applied ")].iterrows():
        rows.append({"estimate": f"Applied, {description(row.estimate[len('applied '):])}", "lenses": row.lenses,
                     "seeds": None, "mean_absolute_error": row.mean_absolute_error, "mean_absolute_error_std": None,
                     "median_absolute_error": row.median_absolute_error, "median_absolute_error_std": None,
                     "median_relative_error": row.median_relative_error, "spearman": row.spearman})
    # With several training seeds, einstein_radius.py writes seeds.csv; with one seed, the rows of that seed are used.
    if (folder / "seeds.csv").exists():
        for _, row in pd.read_csv(folder / "seeds.csv").iterrows():
            initialisation = row.estimate[len("fine-tuned "):]
            start = ("random weights" if initialisation == "random"
                     else description(initialisation[len("pretrained-"):]))
            rows.append({"estimate": f"Fine-tuned, {start}", "lenses": lenses, "seeds": row.seeds,
                         "mean_absolute_error": row.mean_absolute_error_mean,
                         "mean_absolute_error_std": row.mean_absolute_error_std,
                         "median_absolute_error": row.median_absolute_error_mean,
                         "median_absolute_error_std": row.median_absolute_error_std,
                         "median_relative_error": row.median_relative_error_mean, "spearman": row.spearman_mean})
    else:
        for _, row in summary[summary.estimate.str.startswith("fine-tuned ")].iterrows():
            rows.append({"estimate": row.estimate, "lenses": row.lenses, "seeds": 1,
                         "mean_absolute_error": row.mean_absolute_error, "mean_absolute_error_std": None,
                         "median_absolute_error": row.median_absolute_error, "median_absolute_error_std": None,
                         "median_relative_error": row.median_relative_error, "spearman": row.spearman})
    comparisons = pd.read_csv(folder / "comparisons.csv").to_dict("records")
    return rows, comparisons, lenses


# ================================================================== 5. false alarms on the 10k benchmarks (a test)

def false_alarm_10k_rows(suffix, benchmark_rows):
    """A test, not used in the paper: false-alarm rates of the saved networks on the further non-lenses of the 10k
    benchmarks, with the precision expected at the survey lens fractions for the completeness of the same seeds on
    the 1k benchmark."""
    rows = []
    completeness = pd.DataFrame(benchmark_rows)
    files_read = set()
    for path in sorted((RESULTS_FOLDER / "predictions_10k").glob(f"*{suffix}/*/false_alarms.csv")):
        if path.resolve() in files_read:            # a folder reached through a link under an earlier model key
            continue
        files_read.add(path.resolve())
        table = pd.read_csv(path)
        model = model_key(path.parent.parent.name[:len(path.parent.parent.name) - len(suffix)])
        if suffix == "" and model.endswith("_quick"):
            continue
        # init: column name of the input files
        for (trained_on, initialisation), group in table.groupby(["trained_on", "init"], sort=False):
            per_network = group[group.network != "averaged"]      # each of the five fold networks of a seed
            averaged = group[group.network == "averaged"]         # mean lens probability of the five networks
            seeds = sorted(averaged.seed.unique())
            matching = completeness[(completeness.model == NAMES[model]) & (completeness.benchmark == trained_on)
                                    & (completeness.initialisation == initialisation)
                                    & (completeness.seed.isin(seeds))]
            row = {"model": NAMES[model], "benchmark": trained_on, "scored": group.scored.iloc[0],
                   "initialisation": initialisation, "seeds": ", ".join(str(seed) for seed in seeds),
                   "stamps": int(group.stamps.iloc[0]), "networks": len(per_network),
                   "false_alarm_rate_per_network_mean": float(per_network.false_alarm_rate.mean()),
                   "false_alarm_rate_per_network_std": float(per_network.false_alarm_rate.std(ddof=1)),
                   "false_alarm_rate_averaged_mean": float(averaged.false_alarm_rate.mean()),
                   "false_alarm_rate_averaged_std": (float(averaged.false_alarm_rate.std(ddof=1))
                                                     if len(averaged) > 1 else float("nan")),
                   "completeness_1k_mean": float(matching.completeness.mean()) if len(matching) else float("nan")}
            # precision(f) = C f / (C f + F (1 - f)), see model_metrics.py
            for name, fraction in LENS_FRACTIONS_FOR_PRECISION.items():
                for kind in ("per_network", "averaged"):
                    rate, found = row[f"false_alarm_rate_{kind}_mean"], row["completeness_1k_mean"]
                    denominator = found * fraction + rate * (1 - fraction)
                    precision = round(found * fraction / denominator, 4) if denominator > 0 else float("nan")
                    row[f"precision_{kind}_at_lens_fraction_{name}"] = precision
            rows.append(row)
    return rows


# ======================================================================================== 6. Markdown tables

def spread_text(mean, standard_deviation, number_format=".3f", sign=False):
    """mean ± standard deviation; only the mean for a single seed (standard deviation NaN or missing)."""
    text = f"{mean:{'+' if sign else ''}{number_format}}"
    if standard_deviation is None or np.isnan(standard_deviation):
        return text
    return f"{text} ± {standard_deviation:{number_format}}"


def interval_text(value, low, high, digits=3):
    """value [low, high]"""
    return f"{value:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def difference_text(value, low, high, digits=3):
    """Signed difference with its interval."""
    return f"{value:+.{digits}f} [{low:+.{digits}f}, {high:+.{digits}f}]"


def p_text(p_value):
    return "<0.001" if p_value < 0.001 else f"{p_value:.3f}"


def markdown_table(header, rows, text_columns=1):
    """Markdown table; the first text_columns columns are aligned left, the others (numbers) right."""
    alignment = "|".join("---" if column < text_columns else "---:" for column in range(len(header)))
    lines = ["| " + " | ".join(header) + " |", "|" + alignment + "|"]
    lines += ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join(lines)


def write_markdown(output_folder, simulation_rows, benchmark_rows, pretraining_rows, size_rows, seed_rows,
                   seed_difference_rows, radius_rows, radius_comparisons, radius_lenses, false_alarm_rows):
    """results.md: all tables; sections without results are left out."""
    short_name = {benchmark: benchmark.replace("EUCLID-SAGE-", "").replace("_VIS", "") for benchmark in BENCHMARKS}
    sections = ["# UNet-Starlight: results", "",
                "Brackets: 95% intervals from a stratified bootstrap over the test stamps. "
                "p: DeLong test of the AUROC difference, Holm-adjusted within each table.\n"]
    if simulation_rows:
        sections += ["## Simulated benchmark: arc segmentation and lens decision on the test images", "",
                     markdown_table(
                         ["Model", "Parameters", "Best epoch", "Lens Dice", "IoU", "Balanced accuracy",
                          "Lenses found", "False alarms"],
                         [[row["model"], f"{row['parameters']:,}", row["best_epoch"],
                           interval_text(row["dice"], row["dice_ci_low"], row["dice_ci_high"]), f"{row['iou']:.3f}",
                           f"{row['balanced_accuracy']:.3f}", f"{row['lenses_found']} / {row['lenses']}",
                           f"{row['false_alarms']} / {row['nonlenses']}"] for row in simulation_rows]), ""]
        radius_models = [row for row in simulation_rows if row["einstein_radius_mean_absolute_error"] is not None]
        if radius_models:
            sections += ["Einstein radius of the 100 simulated test lenses (UNet-Radius-mini), against the arc-pixel "
                         "estimate: the median distance of the predicted arc pixels from the image centre.", "",
                         markdown_table(
                             ["Model", "Lens Dice (median)", "Non-lenses with an empty mask", "Arc pixels per "
                              "non-lens", "Mean absolute error", "Median absolute error", "Mean error",
                              "Arc-pixel estimate, mean absolute error"],
                             [[row["model"], f"{row['dice_median']:.3f}",
                               f"{100 * row['nonlenses_with_empty_mask']:.0f}%", f"{row['arc_pixels_per_nonlens']:.2f}",
                               f"{row['einstein_radius_mean_absolute_error']:.3f}\"",
                               f"{row['einstein_radius_median_absolute_error']:.3f}\"",
                               f"{row['einstein_radius_mean_error']:+.3f}\"",
                               f"{row['arc_pixel_radius_mean_absolute_error']:.3f}\""] for row in radius_models]), ""]
    if benchmark_rows:
        sections += ["## Lens classification on the 1k benchmarks, every training seed", "", markdown_table(
            ["Model", "Benchmark", "Initialisation", "Seed", "AUROC", "Completeness at 5% false alarms",
             "Fold AUROC", "Lenses found (p > 0.5)", "False alarms (p > 0.5)"],
            [[row["model"], short_name[row["benchmark"]], row["initialisation"], row["seed"],
              interval_text(row["auroc"], row["auroc_ci_low"], row["auroc_ci_high"]),
              interval_text(row["completeness_at_5pct_false_alarms"], row["completeness_at_5pct_ci_low"],
                            row["completeness_at_5pct_ci_high"]),
              f"{row['fold_auroc_mean']:.3f} ± {row['fold_auroc_std']:.3f}",
              f"{row['lenses_found']} / {row['lenses']}", f"{row['false_alarms']} / {row['nonlenses']}"]
             for row in benchmark_rows], text_columns=3), ""]
    if seed_rows:
        sections += ["## Pretraining on simulations: mean ± standard deviation over training seeds", "",
                     markdown_table(
                         ["Model", "Benchmark", "Initialisation", "Seeds", "AUROC", "Completeness at 5% false alarms",
                          "Completeness (p > 0.5)", "False alarms (p > 0.5)", "Balanced accuracy"],
                         [[row["model"], short_name[row["benchmark"]], row["initialisation"], row["seeds"],
                           spread_text(row["auroc_mean"], row["auroc_std"]),
                           spread_text(row["completeness_at_5pct_false_alarms_mean"],
                                       row["completeness_at_5pct_false_alarms_std"]),
                           spread_text(row["completeness_mean"], row["completeness_std"]),
                           spread_text(row["false_alarms_mean"], row["false_alarms_std"], ".1f"),
                           spread_text(row["balanced_accuracy_mean"], row["balanced_accuracy_std"])]
                          for row in seed_rows], text_columns=3), ""]
    if pretraining_rows:
        sections += ["## Pretraining on simulations: pretrained - random initialisation", "", markdown_table(
            ["Model", "Benchmark", "Seed", "AUROC pretrained", "AUROC random", "AUROC difference", "p",
             "Difference in completeness at 5% false alarms", "Folds with pretrained better"],
            [[row["model"], short_name[row["benchmark"]], row["seed"], f"{row['auroc_pretrained']:.3f}",
              f"{row['auroc_random']:.3f}",
              difference_text(row["auroc_difference"], row["auroc_difference_ci_low"],
                              row["auroc_difference_ci_high"]),
              p_text(row["auroc_p_holm"]),
              difference_text(row["completeness_at_5pct_difference"], row["completeness_at_5pct_difference_ci_low"],
                              row["completeness_at_5pct_difference_ci_high"]),
              f"{row['folds_pretrained_better']} of {row['folds']}"] for row in pretraining_rows], text_columns=2), ""]
    if size_rows:
        sections += ["## Model size: smaller - larger model", "", markdown_table(
            ["Comparison", "Benchmark", "Initialisation", "Seed", "AUROC difference", "p",
             "Difference in completeness at 5% false alarms"],
            [[row["comparison"], short_name[row["benchmark"]], row["initialisation"], row["seed"],
              difference_text(row["auroc_difference"], row["auroc_difference_ci_low"],
                              row["auroc_difference_ci_high"]),
              p_text(row["auroc_p_holm"]),
              difference_text(row["completeness_at_5pct_difference"], row["completeness_at_5pct_difference_ci_low"],
                              row["completeness_at_5pct_difference_ci_high"])] for row in size_rows],
            text_columns=3), ""]
    if seed_difference_rows:
        sections += ["## Pretraining and model size: differences, mean ± standard deviation over training seeds", "",
                     markdown_table(
                         ["Comparison", "Benchmark", "Initialisation", "Seeds", "AUROC difference",
                          "Difference in completeness at 5% false alarms"],
                         [[row["comparison"], short_name[row["benchmark"]], row.get("initialisation", ""),
                           row["seeds"], spread_text(row["auroc_difference_mean"], row["auroc_difference_std"],
                                                     sign=True),
                           spread_text(row["completeness_at_5pct_difference_mean"],
                                       row["completeness_at_5pct_difference_std"], sign=True)]
                          for row in seed_difference_rows], text_columns=3), ""]
    if radius_rows:
        sections += [f"## Einstein radius on the {radius_lenses} real lenses with a measured radius", "",
                     ("Applied: network trained on simulations, without further training. Fine-tuned: starting "
                      "point of the fine-tuning; mean ± standard deviation over training seeds."), "",
                     markdown_table(
                         ["Estimate", "Mean absolute error (\")", "Median absolute error (\")",
                          "Median relative error", "Spearman"],
                         [[row["estimate"], spread_text(row["mean_absolute_error"], row["mean_absolute_error_std"]),
                           spread_text(row["median_absolute_error"], row["median_absolute_error_std"]),
                           f"{100 * row['median_relative_error']:.0f}%", f"{row['spearman']:.2f}"]
                          for row in radius_rows]), "",
                     "Paired comparisons: absolute error of estimate 1 minus that of estimate 2 (negative: estimate 1 "
                     "is closer to the measured radius), with a bootstrap interval and the Wilcoxon signed-rank test.",
                     "", markdown_table(
                         ["Comparison (1 - 2)", "Mean difference (\")", "Wilcoxon p"],
                         [[comparison["comparison"],
                           difference_text(comparison["mean_difference"], comparison["ci_low"], comparison["ci_high"]),
                           f"{comparison['wilcoxon_p']:.2g}"] for comparison in radius_comparisons]), ""]
    if false_alarm_rows:
        sections += ["## Test, not used in the paper: false alarms on the 10k benchmarks", "",
                     ("Networks trained on the 1k benchmarks, applied to the non-lenses of the 10k benchmarks that "
                      "are not in the 1k benchmarks. Per network: mean ± standard deviation over all fold networks and "
                      "seeds. Averaged: mean probability of the 5 fold networks of a seed, mean ± standard deviation "
                      "over seeds. Precision: expected at the lens fraction of the population, with the completeness "
                      "of the same seeds on the 1k benchmark."), "", markdown_table(
                         ["Model", "Benchmark", "Initialisation", "Seeds", "Stamps", "False-alarm rate per network",
                          "False-alarm rate, averaged networks", "Completeness (1k)", "Precision at 2e-4",
                          "Precision at 2e-5"],
                         [[row["model"], short_name.get(row["benchmark"], row["benchmark"]), row["initialisation"],
                           row["seeds"], f"{row['stamps']:,}",
                           spread_text(row["false_alarm_rate_per_network_mean"],
                                       row["false_alarm_rate_per_network_std"], ".5f"),
                           spread_text(row["false_alarm_rate_averaged_mean"], row["false_alarm_rate_averaged_std"],
                                       ".5f"),
                           f"{row['completeness_1k_mean']:.3f}", row["precision_per_network_at_lens_fraction_2e-4"],
                           row["precision_per_network_at_lens_fraction_2e-5"]] for row in false_alarm_rows],
                         text_columns=3), ""]
    report = "\n".join(sections)
    (output_folder / "results.md").write_text(report)
    return report


# ============================================================================================ 7. command line

def main():
    argument_parser = argparse.ArgumentParser(description="The tables of the Results section of the paper")
    argument_parser.add_argument("--quick", action="store_true", help="read the results of the --quick runs")
    argument_parser.add_argument("--radius-results", default="einstein_radius_real_10k",
                                 help="results folder of einstein_radius.py (its --out-name)")
    settings = argument_parser.parse_args()
    suffix = "_quick" if settings.quick else ""          # folders of the --quick runs end in _quick
    output_folder = RESULTS_FOLDER / ("tables" + suffix)

    # Simulated benchmark: one row per model.
    simulation_rows = []
    for model in FILTERS:
        simulation_folder = results_folder(RESULTS_FOLDER / "simulations", model, suffix)
        if (simulation_folder / "test_predictions.npz").exists():
            simulation_rows.append(simulation_row(model, simulation_folder))

    # 1k benchmarks: one row per model, initialisation and seed, then the paired comparisons within each benchmark.
    benchmark_rows, pretraining_rows, size_rows = [], [], []
    for benchmark in BENCHMARKS:
        resample_rows, predictions = None, {}          # predictions[(model, initialisation, seed)]
        for model in FILTERS:
            benchmark_folder = results_folder(RESULTS_FOLDER / "benchmarks", model, suffix) / benchmark
            for initialisation in INITIALISATIONS:
                for seed_folder in sorted((benchmark_folder / initialisation).glob("seed*")):
                    if not (seed_folder / "predictions_per_stamp.csv").exists():
                        continue
                    if resample_rows is None:          # the same resamples for every network of this benchmark
                        labels = pd.read_csv(seed_folder / "predictions_per_stamp.csv").label.to_numpy()
                        resample_rows = statistics.stratified_resamples(labels)
                    row, table = benchmark_row(model, benchmark, initialisation, seed_folder, resample_rows)
                    benchmark_rows.append(row)
                    predictions[(model, initialisation, row["seed"])] = table
        for (model, initialisation, seed), table in predictions.items():
            if initialisation == "pretrained" and (model, "random", seed) in predictions:
                pretraining_rows.append(pretraining_row(model, benchmark, seed, table,
                                                        predictions[(model, "random", seed)], resample_rows))
        for smaller, larger in SIZE_PAIRS:
            for (model, initialisation, seed), table in predictions.items():
                if model == smaller and (larger, initialisation, seed) in predictions:
                    size_rows.append({"comparison": f"{NAMES[smaller]} - {NAMES[larger]}", "benchmark": benchmark,
                                      "initialisation": initialisation, "seed": seed,
                                      **paired_comparison(table, predictions[(larger, initialisation, seed)],
                                                          resample_rows,
                                                          f"{NAMES[smaller]} and {NAMES[larger]}, {benchmark}")})
    if not (simulation_rows or benchmark_rows):
        raise SystemExit(f"No results found in {RESULTS_FOLDER}")
    # Holm adjustment within each of the two sets of comparisons.
    add_holm_adjusted_p(pretraining_rows)
    add_holm_adjusted_p(size_rows)

    # Mean and standard deviation over training seeds, of the metrics and of the differences.
    seed_rows = (mean_and_spread(pd.DataFrame(benchmark_rows), ["model", "benchmark", "initialisation"],
                                 SEED_COLUMNS) if benchmark_rows else [])
    seed_difference_rows = []
    if pretraining_rows:
        for row in mean_and_spread(pd.DataFrame(pretraining_rows), ["model", "benchmark"],
                                   ["auroc_difference", "completeness_at_5pct_difference"]):
            seed_difference_rows.append({"comparison": f"{row.pop('model')}: pretrained - random", **row})
    if size_rows:
        seed_difference_rows.extend(mean_and_spread(pd.DataFrame(size_rows),
                                                    ["comparison", "benchmark", "initialisation"],
                                                    ["auroc_difference", "completeness_at_5pct_difference"]))

    # Einstein radius of the real lenses, from the outputs of einstein_radius.py.
    radius_folder = RESULTS_FOLDER / (settings.radius_results + suffix)
    radius_rows, radius_comparisons, radius_lenses = [], [], 0
    if (radius_folder / "summary.csv").exists():
        radius_rows, radius_comparisons, radius_lenses = real_lens_radius_rows(radius_folder)

    false_alarm_rows = false_alarm_10k_rows(suffix, benchmark_rows)      # a test, not used in the paper

    output_folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(simulation_rows).to_csv(output_folder / "simulations.csv", index=False)
    pd.DataFrame(benchmark_rows).to_csv(output_folder / "benchmarks.csv", index=False)
    pd.DataFrame(pretraining_rows).to_csv(output_folder / "differences.csv", index=False)
    pd.DataFrame(size_rows).to_csv(output_folder / "size_differences.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(output_folder / "seeds.csv", index=False)
    pd.DataFrame(seed_difference_rows).to_csv(output_folder / "seed_differences.csv", index=False)
    pd.DataFrame(radius_rows).to_csv(output_folder / "einstein_radius_real.csv", index=False)
    pd.DataFrame(false_alarm_rows).to_csv(output_folder / "false_alarms_10k.csv", index=False)
    report = write_markdown(output_folder, simulation_rows, benchmark_rows, pretraining_rows, size_rows, seed_rows,
                            seed_difference_rows, radius_rows, radius_comparisons, radius_lenses, false_alarm_rows)
    print(report)
    print(f"Tables saved in {output_folder}")


if __name__ == "__main__":
    main()
