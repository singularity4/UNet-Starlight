"""model_metrics.py - the metrics of the U-Nets and of the classifiers on test images

UNet-Starlight.

Arc segmentation (simulated images)
    A pixel belongs to the predicted arc mask if its arc probability is at least the threshold (0.5). For every lens
    the overlap of the predicted mask P and the true mask T is measured by the Dice coefficient,
    2 |P ∩ T| / (|P| + |T|), and by the intersection over union, |P ∩ T| / |P ∪ T|; both are summarised over the
    lenses by their mean and median. A constant of 1e-8 added to numerator and denominator makes both measures equal
    to 1 when the two masks are empty. For non-lenses, whose true mask is empty, the number of predicted arc pixels is
    reported.
    Lens decision: a stamp is classified as a lens if at least 40 pixels have an arc probability of at least 0.5. The
    decisions are summarised by the balanced accuracy and the confusion matrix.

Einstein radius (UNet-Radius-mini, simulated images)
    The error of the estimated Einstein radius over the test lenses, in arcseconds: mean and median absolute error,
    and mean signed error. For comparison, an estimate that needs no training: the median distance of the predicted
    arc pixels from the image centre, times the pixel scale of the simulations (0.101"). Arcs form near the Einstein
    radius, so this distance approximates it.

Lens classification (real benchmarks)
    AUROC of the lens probability: the probability that a randomly chosen lens receives a higher score than a randomly
    chosen non-lens. For the decision of the classifier (lens probability above 0.5, that is the larger of its two
    outputs): completeness, the fraction of lenses classified as lenses, also separately for grade A and grade B
    lenses; false-alarm rate, the fraction of non-lenses classified as lenses; balanced accuracy, accuracy, macro F1
    and the confusion matrix.
    Precision is the fraction of the objects classified as lenses that are lenses; in a benchmark it depends on the
    lens fraction of the benchmark (48.7%). The precision expected at a lens fraction f is
        precision(f) = C f / (C f + F (1 - f)),
    with completeness C and false-alarm rate F. f = 2e-5 for the population of R (497 candidates among about 27
    million classified objects) and f = 2e-4 for the populations of S and D (497 among about 2.2 million galaxies
    brighter than VIS = 23). With few false alarms these precisions are uncertain, so the counts are reported with
    them.
"""

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

LENS_FRACTIONS_FOR_PRECISION = {"2e-4": 2e-4, "2e-5": 2e-5}      # populations of S and D, population of R
SIMULATION_PIXEL_SCALE = 0.101          # arcseconds per pixel in the Euclid VIS settings of lenstronomy


# ======================================================================================== 1. arc segmentation

def segmentation_metrics(probabilities, true_masks, labels, threshold, min_pixels=40):
    """Overlap of predicted and true arc masks for the lenses, predicted arc pixels for the non-lenses, and the lens
    decisions of the rule 'at least min_pixels pixels with arc probability >= threshold'."""
    predicted_masks = probabilities >= threshold
    true_masks = true_masks.astype(bool)
    labels = labels.astype(int)
    is_lens = labels == 1
    is_nonlens = ~is_lens

    intersection = np.logical_and(predicted_masks, true_masks).sum(axis=(1, 2))
    union = np.logical_or(predicted_masks, true_masks).sum(axis=(1, 2))
    predicted_pixels = predicted_masks.sum(axis=(1, 2))
    true_pixels = true_masks.sum(axis=(1, 2))
    dice = (2 * intersection + 1e-8) / (predicted_pixels + true_pixels + 1e-8)
    intersection_over_union = (intersection + 1e-8) / (union + 1e-8)
    predicted_lens = (predicted_pixels >= int(min_pixels)).astype(int)

    def summary(function, values, selection):
        return float(function(values[selection])) if np.any(selection) else float("nan")

    return {"threshold": float(threshold),
            "min_pixels_for_lens": int(min_pixels),
            "lens_dice_mean": summary(np.mean, dice, is_lens),
            "lens_dice_median": summary(np.median, dice, is_lens),
            "lens_iou_mean": summary(np.mean, intersection_over_union, is_lens),
            "lens_iou_median": summary(np.median, intersection_over_union, is_lens),
            "nonlens_mean_predicted_pixels": summary(np.mean, predicted_pixels, is_nonlens),
            "nonlens_zero_mask_fraction": summary(np.mean, predicted_pixels == 0, is_nonlens),
            "mask_based_lens_balanced_accuracy": float(balanced_accuracy_score(labels, predicted_lens)),
            "mask_based_lens_confusion_0_1": confusion_matrix(labels, predicted_lens, labels=[0, 1]).tolist()}


def mean_lens_dice(probabilities, true_masks, labels, threshold=0.5):
    """Mean Dice coefficient of the predicted arc masks over the lenses: the criterion by which the epoch is chosen
    in the training on simulations."""
    predicted_masks = probabilities >= threshold
    true_masks = true_masks.astype(bool)
    is_lens = labels == 1
    if not np.any(is_lens):
        return float("nan")
    intersection = np.logical_and(predicted_masks[is_lens], true_masks[is_lens]).sum(axis=(1, 2))
    predicted_pixels = predicted_masks[is_lens].sum(axis=(1, 2))
    true_pixels = true_masks[is_lens].sum(axis=(1, 2))
    dice = (2 * intersection + 1e-8) / (predicted_pixels + true_pixels + 1e-8)
    return float(dice.mean())


# ========================================================================================= 2. Einstein radius

def arc_pixel_radius(probabilities, threshold, pixel_scale=SIMULATION_PIXEL_SCALE):
    """For every image, the median distance (arcseconds) of the pixels with arc probability >= threshold from the
    image centre; NaN if no pixel reaches the threshold."""
    height, width = probabilities.shape[-2:]
    rows, columns = np.mgrid[0:height, 0:width]
    distance = np.hypot(rows - (height - 1) / 2, columns - (width - 1) / 2) * pixel_scale
    radii = []
    for probability_map in probabilities:
        arc_pixels = probability_map >= threshold
        radii.append(float(np.median(distance[arc_pixels])) if arc_pixels.any() else float("nan"))
    return np.array(radii)


def einstein_radius_metrics(true_radii, estimated_radii, probabilities, labels, threshold):
    """Errors of the estimated Einstein radius over the lenses, and of the arc-pixel estimate for comparison."""
    is_lens = np.asarray(labels) == 1
    true_radii = np.asarray(true_radii, dtype=float)[is_lens]
    error = np.asarray(estimated_radii, dtype=float)[is_lens] - true_radii
    arc_pixel_error = arc_pixel_radius(np.asarray(probabilities)[is_lens], threshold) - true_radii
    found = np.isfinite(arc_pixel_error)
    return {"lenses": int(is_lens.sum()),
            "mean_absolute_error": float(np.mean(np.abs(error))),
            "median_absolute_error": float(np.median(np.abs(error))),
            "mean_error": float(np.mean(error)),
            "arc_pixel_estimate_mean_absolute_error": float(np.mean(np.abs(arc_pixel_error[found]))),
            "arc_pixel_estimate_median_absolute_error": float(np.median(np.abs(arc_pixel_error[found]))),
            "arc_pixel_estimate_lenses": int(found.sum())}


# ===================================================================================== 3. lens classification

def area_under_roc(labels, probabilities):
    """AUROC of the lens probabilities; NaN if only one class is present."""
    return float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else float("nan")


def classification_metrics(labels, decisions, lens_probabilities):
    """Balanced accuracy, accuracy, macro F1, AUROC and confusion matrix."""
    return {"balanced_accuracy": float(balanced_accuracy_score(labels, decisions)),
            "accuracy": float(accuracy_score(labels, decisions)),
            "macro_f1": float(f1_score(labels, decisions, average="macro")),
            "auroc": area_under_roc(labels, lens_probabilities),
            "confusion_matrix": confusion_matrix(labels, decisions, labels=[0, 1]).tolist()}


def benchmark_metrics(predictions):
    """Completeness, false alarms and precision, from a table with the columns label, grade and predicted_lens."""
    lenses, nonlenses = predictions[predictions.label == 1], predictions[predictions.label == 0]
    completeness = lenses.predicted_lens.mean()
    false_alarm_rate = nonlenses.predicted_lens.mean()
    metrics = {"lenses_found": int(lenses.predicted_lens.sum()), "lenses": len(lenses),
               "completeness": round(float(completeness), 4),
               "false_alarms": int(nonlenses.predicted_lens.sum()), "nonlenses": len(nonlenses),
               "false_alarm_rate": round(float(false_alarm_rate), 5)}
    for grade in ("A", "B"):
        graded = lenses[lenses.grade == grade]
        metrics[f"completeness_grade_{grade}"] = round(float(graded.predicted_lens.mean()), 4) if len(graded) else None
    lens_fractions = {"in_benchmark": len(lenses) / len(predictions),
                      **{f"at_lens_fraction_{name}": value for name, value in LENS_FRACTIONS_FOR_PRECISION.items()}}
    for fraction_name, lens_fraction in lens_fractions.items():
        denominator = completeness * lens_fraction + false_alarm_rate * (1 - lens_fraction)
        metrics[f"precision_{fraction_name}"] = (round(float(completeness * lens_fraction / denominator), 4)
                                                 if denominator else None)
    return metrics
