"""post_training_statistics.py - confidence intervals and paired tests for the test results

UNet-Starlight.

The statistics are computed after training from the saved test predictions, that is the true label and the lens
probability of every test stamp; no network is trained or run. They apply in the same way to the cross-validation
predictions of a 1k benchmark, in which every stamp is tested once, and to the predictions of saved networks on further
stamps, such as the additional non-lenses of the 10k benchmarks.

Each 1k benchmark has 487 lenses and 513 non-lenses. The uncertainty of a metric comes mainly from this finite set of
test stamps and is estimated by resampling them.

Stratified bootstrap
    Each resample draws 487 lenses from the lenses and 513 non-lenses from the non-lenses, with replacement; there are
    2,000 resamples with a fixed seed. The 95% interval of a metric is formed by the 2.5th and 97.5th percentiles of
    its values over the resamples.

Paired comparison
    Two networks tested on the same stamps (pretrained against random weights, or two model sizes) are compared on the
    same resamples, which gives an interval for the difference itself. This interval is narrower than the two separate
    intervals, because both networks tend to err on the same stamps. The difference of the AUROC is also tested with
    the method of DeLong et al. (1988) for correlated ROC curves, in the fast form of Sun and Xu (2014).

Several comparisons
    The p-values of each of the two sets of comparisons (pretrained against random weights; pairs of model sizes) are
    adjusted with the method of Holm (1979), which keeps the probability of at least one false positive within a set
    below the significance level.

Completeness at a fixed false-alarm rate
    The fraction of lenses scored above the threshold that at most a given fraction of the non-lenses exceeds (5% in
    the paper, that is 25 of 513 non-lenses). It is a point on the ROC curve, not a threshold chosen for a network.

The intervals do not include the variation between training runs; that variation is reported separately as the spread
over training seeds.
"""

import numpy as np
from scipy.stats import norm, rankdata

RESAMPLES = 2000
BOOTSTRAP_SEED = 20261008
LEVEL = 0.95


def auroc_from_ranks(labels, scores):
    """AUROC computed from the ranks of the scores (Mann-Whitney U statistic), with ties counted one half. It equals
    the AUROC of scikit-learn and is faster, which matters for the 2,000 bootstrap resamples."""
    labels = np.asarray(labels).astype(bool)
    ranks = rankdata(scores)
    lenses, nonlenses = int(labels.sum()), int((~labels).sum())
    return float((ranks[labels].sum() - lenses * (lenses + 1) / 2) / (lenses * nonlenses))


def completeness_at_false_alarm_rate(labels, scores, rate=0.01):
    """The fraction of lenses scored above the threshold that at most the fraction `rate` of the non-lenses
    exceeds."""
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=float)
    nonlens_scores = np.sort(scores[~labels])[::-1]
    allowed = int(np.floor(rate * len(nonlens_scores)))
    threshold = nonlens_scores[allowed]               # the (allowed + 1)-th highest score of a non-lens
    return float(np.mean(scores[labels] > threshold))


def stratified_resamples(labels, resamples=RESAMPLES, seed=BOOTSTRAP_SEED):
    """Row indices of the bootstrap resamples, with shape (resamples, number of stamps). Lenses and non-lenses are
    drawn separately, so that every resample has the same numbers of lenses and non-lenses as the benchmark."""
    labels = np.asarray(labels).astype(bool)
    lens_rows, nonlens_rows = np.flatnonzero(labels), np.flatnonzero(~labels)
    generator = np.random.default_rng(seed)
    return np.hstack([generator.choice(lens_rows, size=(resamples, len(lens_rows)), replace=True),
                      generator.choice(nonlens_rows, size=(resamples, len(nonlens_rows)), replace=True)])


def bootstrap_values(metric, labels, scores, resample_rows):
    """The value of metric(labels, scores) on every resample."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    return np.array([metric(labels[rows], scores[rows]) for rows in resample_rows])


def percentile_interval(values, level=LEVEL):
    """The interval between the (1 - level)/2 and (1 + level)/2 percentiles of the bootstrap values."""
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    return float(low), float(high)


def bootstrap_p_value(differences):
    """Two-sided bootstrap p-value of a difference: twice the smaller of the fractions of resamples with a difference
    of at most 0 and of at least 0, limited to 1."""
    differences = np.asarray(differences)
    return float(min(1.0, 2 * min(np.mean(differences <= 0), np.mean(differences >= 0))))


def delong_test(labels, scores_a, scores_b):
    """DeLong test of AUROC(a) - AUROC(b) for two sets of scores of the same stamps.
    Returns (difference, standard error, two-sided p-value)."""
    labels = np.asarray(labels).astype(bool)
    lens_count, nonlens_count = int(labels.sum()), int((~labels).sum())
    placements_lenses, placements_nonlenses, aurocs = [], [], []
    for scores in (np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)):
        lens_scores, nonlens_scores = scores[labels], scores[~labels]
        combined_ranks = rankdata(np.concatenate([lens_scores, nonlens_scores]))
        lens_ranks, nonlens_ranks = rankdata(lens_scores), rankdata(nonlens_scores)
        aurocs.append((combined_ranks[:lens_count].sum() - lens_count * (lens_count + 1) / 2)
                      / (lens_count * nonlens_count))
        # placement values: for every lens the fraction of non-lenses below it, for every non-lens the fraction of
        # lenses above it
        placements_lenses.append((combined_ranks[:lens_count] - lens_ranks) / nonlens_count)
        placements_nonlenses.append(1 - (combined_ranks[lens_count:] - nonlens_ranks) / lens_count)
    covariance = (np.cov(np.vstack(placements_lenses)) / lens_count
                  + np.cov(np.vstack(placements_nonlenses)) / nonlens_count)
    variance = covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1]
    difference = aurocs[0] - aurocs[1]
    standard_error = float(np.sqrt(max(variance, 0.0)))
    p_value = float(2 * norm.sf(abs(difference) / standard_error)) if standard_error > 0 else float("nan")
    return float(difference), standard_error, p_value


def holm_adjusted(p_values):
    """Holm-adjusted p-values, in the order of the input: the k-th smallest of n p-values is multiplied by n - k + 1,
    the results are made non-decreasing in this order and limited to 1."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running_maximum = 0.0
    for position, index in enumerate(order):
        running_maximum = max(running_maximum, (len(p_values) - position) * p_values[index])
        adjusted[index] = min(1.0, running_maximum)
    return adjusted
