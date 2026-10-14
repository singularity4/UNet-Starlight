"""train_benchmarks.py - lens classification on the EUCLID-SAGE benchmarks, with the encoder pretrained on simulations
or initialised randomly

UNet-Starlight.

Classifier
    The encoder of a U-Net (blocks inc and d1-d4), global average pooling of the bottleneck feature maps and a linear
    layer with two outputs (unet_models.py). The architecture, and for the pretrained encoder its weights, are taken
    from the U-Net trained on simulations (train_simulations.py).

Initialisation of the encoder (--init)
    pretrained   weights of the U-Net trained on simulations; all weights are trained
    random       random weights; all weights are trained
    frozen       weights of the U-Net trained on simulations, kept fixed; only the linear layer is trained
                 (initial experiments, not in the results of the paper)
    The linear layer starts from random weights in all three cases. The comparison of pretrained and random
    initialisation, with everything else equal, measures the effect of pretraining on simulations.

Data
    The 1k benchmarks EUCLID-SAGE-1k-R, -S and -D (487 lenses, 513 non-lenses each). The stamps are converted to the
    units of the simulations and transformed with asinh(x / s), with the scale s of the U-Net trained on simulations;
    they are standardised with the mean and standard deviation of the training stamps of each fold, and training
    stamps are randomly rotated and reflected (data.py).

Cross-validation and choice of the learning rate
    Every benchmark is divided into five stratified folds (seed 42), so that every stamp is tested exactly once by a
    network that has not seen it. Within each fold:
    1. The 800 training stamps are divided into 640 for training and 160 for validation, with the lens fraction kept;
       the division is the same for all initialisations and seeds. The 200 test stamps are used only for the final
       test, not to choose the epoch or the learning rate.
    2. For each learning rate of the grid (1e-4, 3e-4, 1e-3, 3e-3) the network is trained on the 640 stamps for 60
       epochs: AdamW with weight decay 1e-4, batches of 16 stamps, a learning rate that decreases to zero along a
       cosine over the 60 epochs, and the cross-entropy with class weights N / (2 N_c) (N training stamps, N_c of them
       in class c), which give lenses and non-lenses equal total weight. After every epoch the validation loss and
       AUROC are computed; the weights of the epoch with the lowest validation loss are kept.
    3. The learning rate whose kept network has the highest validation AUROC (rounded to four decimals; for equal
       values, the lower validation loss) is chosen, and its network is tested once on the 200 test stamps.
    Before every training the random generators are seeded with 1000 s + k (training seed s, fold k). Within a fold,
    all learning rates therefore start from the same weights, and the pretrained and random networks start from the
    same weights of the linear layer and see the training stamps in the same order and with the same augmentation.

Training seeds
    The learning rate is searched with one training seed (--search-seed, default 1). Further seeds (--seeds 2 3)
    retrain each fold with the learning rate chosen by the search seed; the spread over seeds measures the variation
    between training runs.

Decision and metrics
    A stamp is classified as a lens if its lens probability, the softmax of the two outputs, is above 0.5. The metrics
    (model_metrics.py) are computed from the pooled test predictions of the five folds.

Checks (printed at the end and written to checks.txt)
    A chosen learning rate at the edge of the grid indicates that the grid may be too narrow; a best epoch among the
    last 10 indicates that more epochs may be needed.

Resuming
    Every fold is saved when it is finished. A second run with the same settings and the same U-Net trained on
    simulations (compared by the SHA-256 checksum of its file) skips the finished folds. Results on a GPU can differ
    slightly from those on a CPU, because the order of floating-point operations differs.

Outputs (results/benchmarks/<model>/; <model>_quick/ for a quick run)
    settings.json
    <benchmark>/<init>/seed<s>/fold<k>.json                 choices, validation results and test predictions of fold k
    <benchmark>/<init>/seed<s>/fold<k>_lr<rate>_curve.csv   training and validation loss and AUROC of every epoch
    <benchmark>/<init>/seed<s>/predictions_per_stamp.csv    lens probability (6 decimals) and decision of every stamp
    <benchmark>/<init>/seed<s>/fold<k>_network.pt           the tested network of fold k (not with --no-save-weights)
    <benchmark>/<init>/seed<s>/summary.json
    summary.csv, learning_rate_choices.csv, checks.txt
    paper_results.py makes the tables of the paper from these files.

Usage
    python train_benchmarks.py --model starlight-mini --save-weights              # learning rate search, seed 1
    python train_benchmarks.py --model starlight-mini --seeds 2 3 --save-weights  # further training seeds
    python train_benchmarks.py --model starlight-tiny --quick --benchmarks EUCLID-SAGE-1k-R_VIS    # test of the setup
"""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import (BENCHMARKS, RESULTS_FOLDER, BenchmarkStampDataset, asinh_transform, cross_validation_folds,
                  inner_validation_split, load_stamps_in_simulation_units, read_benchmark_table, set_random_seeds)
from model_metrics import area_under_roc, benchmark_metrics, classification_metrics
from unet_models import (FILTERS, NAMES, build_classifier, choose_device, load_saved_network, model_key,
                         model_of_saved_network)

INITIALISATIONS = {"pretrained": "encoder pretrained on simulations",
                   "random": "random initialisation",
                   "frozen": "pretrained encoder kept fixed (initial experiments)"}
LATE_EPOCHS = 10                    # a best epoch among the last 10 is reported
SETTINGS_COMPARED_ON_RESUME = ("model", "checkpoint_sha256", "learning_rates", "epochs", "batch_size",
                               "weight_decay", "folds", "split_seed", "validation_fraction", "search_seed")


# =========================================================================================== 1. training and validation

def class_weights_for(labels):
    """Class weights N / (2 N_c), inversely proportional to the class frequencies."""
    counts = np.bincount(labels, minlength=2).astype(float)
    return torch.tensor(len(labels) / (2.0 * np.maximum(counts, 1)), dtype=torch.float32)


@torch.no_grad()
def validate(network, loader, loss_function):
    """Mean validation loss, lens probabilities and labels."""
    network.eval()
    device = next(network.parameters()).device
    total_loss, number_seen, probabilities, labels = 0.0, 0, [], []
    for images, batch_labels, _ in loader:
        outputs = network(images.to(device))
        total_loss += loss_function(outputs, batch_labels.to(device)).item() * len(batch_labels)
        number_seen += len(batch_labels)
        probabilities.extend(torch.softmax(outputs, 1)[:, 1].cpu().numpy())
        labels.extend(batch_labels.numpy())
    return total_loss / number_seen, np.array(probabilities), np.array(labels)


def train_and_validate(network, training_loader, validation_loader, initialisation, learning_rate, settings,
                       class_weights):
    """Train for settings.epochs epochs and keep the weights of the epoch with the lowest validation loss.
    Returns the curve of all epochs and the validation results of the kept epoch."""
    device = next(network.parameters()).device
    loss_function = nn.CrossEntropyLoss(weight=class_weights.to(device))
    if initialisation == "frozen":
        for parameter in network.encoder.parameters():
            parameter.requires_grad = False
        trained_parameters = network.classifier.parameters()
    else:
        trained_parameters = network.parameters()
    optimizer = torch.optim.AdamW(trained_parameters, lr=learning_rate, weight_decay=settings.weight_decay)
    learning_rate_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(settings.epochs, 1))
    curve, best = [], {"validation_loss": float("inf")}
    best_weights = copy.deepcopy(network.state_dict())
    for epoch in range(1, settings.epochs + 1):
        network.train()
        if initialisation == "frozen":
            network.encoder.eval()              # a fixed encoder: no dropout
        total_loss, number_seen = 0.0, 0
        for images, labels, _ in training_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(network(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            number_seen += len(labels)
        learning_rate_schedule.step()
        validation_loss, probabilities, labels = validate(network, validation_loader, loss_function)
        validation_auroc = area_under_roc(labels, probabilities)
        curve.append({"epoch": epoch, "training_loss": total_loss / number_seen,
                      "validation_loss": validation_loss, "validation_auroc": validation_auroc})
        if validation_loss < best["validation_loss"]:
            best = {"best_epoch": epoch, "validation_loss": validation_loss, "validation_auroc": validation_auroc}
            best_weights = copy.deepcopy(network.state_dict())
    network.load_state_dict(best_weights)
    return curve, best


@torch.no_grad()
def predict(network, loader):
    """Lens probabilities, decisions (the larger of the two outputs) and positions of the test stamps."""
    network.eval()
    device = next(network.parameters()).device
    probabilities, decisions, positions = [], [], []
    for images, _, indices in loader:
        outputs = network(images.to(device))
        probabilities.extend(torch.softmax(outputs, 1)[:, 1].cpu().numpy())
        decisions.extend(outputs.argmax(1).cpu().numpy())
        positions.extend(indices.numpy())
    return np.array(probabilities), np.array(decisions), np.array(positions)


# =============================================================================================== 2. one fold

def run_fold(fold, training_rows, test_rows, labels, transformed_stamps, initialisation, seed, learning_rates,
             context):
    """Choice of the learning rate on the validation stamps of one fold, then one test on its test stamps."""
    settings, pretrained_network, folder = context["settings"], context["pretrained_network"], context["folder"]
    fit_rows, validation_rows = inner_validation_split(training_rows, labels, settings.validation_fraction,
                                                       settings.split_seed + fold)
    assert not set(test_rows) & (set(fit_rows) | set(validation_rows)), "test stamps used in training or validation"
    mean, standard_deviation = transformed_stamps[fit_rows].mean(), transformed_stamps[fit_rows].std()

    def loader(rows, augment, shuffle):
        return DataLoader(BenchmarkStampDataset(transformed_stamps[rows], labels[rows], mean, standard_deviation,
                                                apply_symmetries=augment),
                          batch_size=settings.batch_size, shuffle=shuffle)

    candidates = []
    for learning_rate in learning_rates:
        set_random_seeds(1000 * seed + fold)            # the same initial weights and order for every learning rate
        network = build_classifier(pretrained_network, initialisation).to(context["device"])
        curve, best = train_and_validate(network, loader(fit_rows, True, True), loader(validation_rows, False, False),
                                         initialisation, learning_rate, settings, class_weights_for(labels[fit_rows]))
        pd.DataFrame(curve).to_csv(folder / f"fold{fold}_lr{learning_rate:g}_curve.csv", index=False)
        candidates.append({"learning_rate": learning_rate, **best, "network": network})
        print(f"      fold {fold}, learning rate {learning_rate:g}: best epoch {best['best_epoch']}, validation "
              f"loss {best['validation_loss']:.4f}, AUROC {best['validation_auroc']:.4f}", flush=True)

    chosen = min(candidates, key=lambda candidate: (-round(candidate["validation_auroc"], 4),
                                                    candidate["validation_loss"]))
    probabilities, decisions, positions = predict(chosen["network"], loader(test_rows, False, False))
    if settings.save_weights:
        torch.save({"model_state_dict": chosen["network"].cpu().state_dict(),
                    "architecture": pretrained_network["architecture"], "model": settings.model,
                    "init": initialisation, "benchmark": context["benchmark_name"], "seed": seed, "fold": fold,
                    "learning_rate": chosen["learning_rate"], "best_epoch": chosen["best_epoch"],
                    "preprocessing": {"asinh_scale": float(pretrained_network["preprocessing"]["asinh_scale"]),
                                      "mean": float(mean), "std": float(standard_deviation)},
                    "pretraining_checkpoint_sha256": settings.checkpoint_sha256},
                   folder / f"fold{fold}_network.pt")
    rows_in_benchmark = test_rows[positions]
    return {"fold": fold, "seed": seed, "model": settings.model, "init": initialisation,
            "training_stamps": len(fit_rows), "validation_stamps": len(validation_rows), "test_stamps": len(test_rows),
            "chosen_learning_rate": chosen["learning_rate"], "chosen_best_epoch": chosen["best_epoch"],
            "chosen_validation_auroc": chosen["validation_auroc"],
            "candidates": [{key: value for key, value in candidate.items() if key != "network"}
                           for candidate in candidates],
            "test_rows": rows_in_benchmark.tolist(), "test_lens_probability": probabilities.round(6).tolist(),
            "test_decision": decisions.tolist(),
            "test_auroc": area_under_roc(labels[rows_in_benchmark], probabilities)}


# ======================================================================= 3. one benchmark, initialisation and seed

def run_initialisation(benchmark_name, benchmark_table, transformed_stamps, initialisation, seed, context):
    """All folds of one benchmark, initialisation and seed; finished folds are read from their files.
    Returns the summary row and the learning rate choices."""
    settings, output_folder = context["settings"], context["output_folder"]
    folder = output_folder / benchmark_name / initialisation / f"seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    context = {**context, "folder": folder, "benchmark_name": benchmark_name}
    labels = benchmark_table.label.to_numpy().astype(int)
    search_seed = settings.search_seed
    fold_records = []
    for fold, training_rows, test_rows in cross_validation_folds(labels, settings.folds, settings.split_seed):
        record_path = folder / f"fold{fold}.json"
        if record_path.exists():
            fold_records.append(json.loads(record_path.read_text()))
            missing_network = settings.save_weights and not (folder / f"fold{fold}_network.pt").exists()
            note = " (finished without --save-weights: no saved network)" if missing_network else ""
            print(f"    fold {fold}: finished earlier{note}", flush=True)
            continue
        if seed == search_seed:
            learning_rates = settings.learning_rates
        else:                                           # further seeds: the learning rate chosen by the search seed
            search_record = output_folder / benchmark_name / initialisation / f"seed{search_seed}" / f"fold{fold}.json"
            if not search_record.exists():
                raise SystemExit(f"Run seed {search_seed} first: {search_record} is missing.")
            learning_rates = [json.loads(search_record.read_text())["chosen_learning_rate"]]
        start = time.time()
        record = run_fold(fold, training_rows, test_rows, labels, transformed_stamps, initialisation, seed,
                          learning_rates, context)
        record["runtime_minutes"] = round((time.time() - start) / 60, 1)
        record_path.write_text(json.dumps(record, indent=1))
        fold_records.append(record)
        print(f"    fold {fold}: learning rate {record['chosen_learning_rate']:g} (best epoch "
              f"{record['chosen_best_epoch']}), test AUROC {record['test_auroc']:.4f}, "
              f"{record['runtime_minutes']:.0f} min", flush=True)

    predictions = benchmark_table[["stamp_file", "label", "grade", "object_id"]].copy()
    predictions["fold"], predictions["lens_probability"], predictions["predicted_lens"] = -1, np.nan, -1
    for record in fold_records:
        rows = np.array(record["test_rows"])
        predictions.loc[rows, "fold"] = record["fold"]
        predictions.loc[rows, "lens_probability"] = record["test_lens_probability"]
        predictions.loc[rows, "predicted_lens"] = record["test_decision"]
    predictions.to_csv(folder / "predictions_per_stamp.csv", index=False)
    overall = {**classification_metrics(labels, predictions.predicted_lens.to_numpy(),
                                        predictions.lens_probability.to_numpy()),
               **benchmark_metrics(predictions)}
    summary = {"benchmark": benchmark_name, "model": settings.model, "init": initialisation, "seed": seed,
               "out_of_fold_results": overall,
               "runtime_minutes": round(sum(record.get("runtime_minutes", 0) for record in fold_records), 1)}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    choices = [{"benchmark": benchmark_name, "model": settings.model, "init": initialisation, "seed": seed,
                "fold": record["fold"], "chosen_learning_rate": record["chosen_learning_rate"],
                "best_epoch": record["chosen_best_epoch"], "validation_auroc": record["chosen_validation_auroc"],
                "test_auroc": record["test_auroc"]} for record in fold_records]
    fold_aurocs = [record["test_auroc"] for record in fold_records]
    row = {"benchmark": benchmark_name, "model": settings.model, "init": initialisation, "seed": seed,
           **{key: value for key, value in overall.items() if key != "confusion_matrix"},
           "fold_auroc_mean": float(np.mean(fold_aurocs)), "fold_auroc_std": float(np.std(fold_aurocs, ddof=1)),
           "confusion_matrix": json.dumps(overall["confusion_matrix"])}
    return row, choices


# ============================================================================================ 4. tables and checks

def replace_rows(path, new_rows, keys):
    """Write new_rows to a CSV table, replacing earlier rows with the same keys and keeping the others."""
    new_table = pd.DataFrame(new_rows)
    if path.exists():
        earlier = pd.read_csv(path)
        replaced = earlier.set_index(keys).index.isin(new_table.set_index(keys).index)
        new_table = pd.concat([earlier[~replaced], new_table], ignore_index=True)
    new_table.to_csv(path, index=False)
    return new_table


def write_checks(choices_table, settings, output_folder):
    """Report chosen learning rates at the edge of the grid and best epochs among the last LATE_EPOCHS epochs."""
    searched = choices_table[choices_table.seed == settings.search_seed]
    edges = (min(settings.learning_rates), max(settings.learning_rates))
    lines = [f"Checks of the learning rate search, {NAMES[settings.model]}", ""]
    for (benchmark, initialisation), group in searched.groupby(["benchmark", "init"]):
        rates = ", ".join(f"{rate:g}" for rate in group.chosen_learning_rate)
        epochs = ", ".join(str(int(epoch)) for epoch in group.best_epoch)
        lines.append(f"{benchmark}, {initialisation} ({INITIALISATIONS[initialisation]}): learning rates {rates}; "
                     f"best epochs {epochs}")
        at_edge = group.chosen_learning_rate.isin(edges).sum() if len(settings.learning_rates) > 1 else 0
        late = (group.best_epoch > settings.epochs - LATE_EPOCHS).sum() if settings.epochs > LATE_EPOCHS else 0
        if at_edge:
            lines.append(f"  in {at_edge} of {len(group)} folds the chosen learning rate is at the edge of the grid "
                         f"({edges[0]:g} or {edges[1]:g}); the grid may be too narrow")
        if late:
            lines.append(f"  in {late} of {len(group)} folds the best epoch is among the last {LATE_EPOCHS} of "
                         f"{settings.epochs}; more epochs may be needed")
    text = "\n".join(lines) + "\n"
    (output_folder / "checks.txt").write_text(text)
    print("\n" + text, flush=True)


# ===================================================================================================== 5. main

def parse_arguments():
    argument_parser = argparse.ArgumentParser(description="Lens classification on the EUCLID-SAGE benchmarks")
    argument_parser.add_argument("--model", type=model_key, choices=list(FILTERS), default="starlight",
                                 help="starlight, starlight-mini or starlight-tiny (or radius-mini)")
    argument_parser.add_argument("--checkpoint", type=Path, default=None,
                                 help="the U-Net trained on simulations (default "
                                      "results/simulations/<model>/best_unet.pt)")
    argument_parser.add_argument("--benchmarks", nargs="+", default=BENCHMARKS)
    argument_parser.add_argument("--init", nargs="+", default=["pretrained", "random"], choices=list(INITIALISATIONS),
                                 help="initialisations of the encoder")
    argument_parser.add_argument("--learning-rates", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3, 3e-3])
    argument_parser.add_argument("--epochs", type=int, default=60)
    argument_parser.add_argument("--batch-size", type=int, default=16)
    argument_parser.add_argument("--weight-decay", type=float, default=1e-4)
    argument_parser.add_argument("--folds", type=int, default=5)
    argument_parser.add_argument("--split-seed", type=int, default=42, help="seed of the folds and validation splits")
    argument_parser.add_argument("--validation-fraction", type=float, default=0.2)
    argument_parser.add_argument("--seeds", nargs="+", type=int, default=[1], help="training seeds")
    argument_parser.add_argument("--search-seed", type=int, default=1,
                                 help="training seed with the learning rate search; other seeds use its choices")
    argument_parser.add_argument("--save-weights", action=argparse.BooleanOptionalAction, default=True,
                                 help="save the tested network of every fold (default; --no-save-weights to skip)")
    argument_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                                 help="auto: a GPU if one is available, otherwise the CPU")
    argument_parser.add_argument("--quick", action="store_true",
                                 help="2 learning rates, 3 folds and 3 epochs, to test the setup")
    settings = argument_parser.parse_args()
    if settings.quick:
        settings.learning_rates, settings.folds, settings.epochs = [3e-4, 1e-3], 3, 3
    return settings


def main():
    settings = parse_arguments()
    device = choose_device(settings.device)
    settings.device_used = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    run_name = settings.model + ("_quick" if settings.quick else "")
    if settings.checkpoint is None:
        settings.checkpoint = RESULTS_FOLDER / "simulations" / run_name / "best_unet.pt"
    settings.checkpoint = settings.checkpoint.expanduser().resolve()

    pretrained_network = load_saved_network(settings.checkpoint)
    settings.checkpoint_sha256 = hashlib.sha256(settings.checkpoint.read_bytes()).hexdigest()
    if settings.search_seed in settings.seeds:
        settings.seeds = [settings.search_seed] + [seed for seed in settings.seeds if seed != settings.search_seed]
    if model_of_saved_network(pretrained_network) != settings.model:
        raise SystemExit(f"{settings.checkpoint} is a {NAMES[model_of_saved_network(pretrained_network)]} network, "
                         f"not {NAMES[settings.model]}.")
    output_folder = RESULTS_FOLDER / "benchmarks" / run_name
    output_folder.mkdir(parents=True, exist_ok=True)

    recorded = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(settings).items()}
    settings_path = output_folder / "settings.json"
    if settings_path.exists():
        earlier = json.loads(settings_path.read_text())
        earlier["model"] = model_key(earlier.get("model"))      # results made with an earlier model key
        changed = [key for key in SETTINGS_COMPARED_ON_RESUME if earlier.get(key) != recorded.get(key)]
        if changed:
            raise SystemExit(f"{output_folder} contains results with other settings ({', '.join(changed)}). "
                             "Move that folder, or run again with the earlier settings.")
    settings_path.write_text(json.dumps(recorded, indent=2))

    print(f"{NAMES[settings.model]}; U-Net trained on simulations: {settings.checkpoint}; results in {output_folder}; "
          f"device {settings.device_used}\n"
          f"learning rates {', '.join(f'{rate:g}' for rate in settings.learning_rates)}; {settings.epochs} epochs; "
          f"{settings.folds} folds; training seeds {settings.seeds}", flush=True)
    context = {"settings": settings, "pretrained_network": pretrained_network, "output_folder": output_folder,
               "device": device}
    stamps_per_benchmark = {}
    for seed in settings.seeds:
        for benchmark_name in settings.benchmarks:
            benchmark_table = read_benchmark_table(benchmark_name)
            if benchmark_name not in stamps_per_benchmark:
                stamps = load_stamps_in_simulation_units(benchmark_table)
                asinh_scale = float(pretrained_network["preprocessing"]["asinh_scale"])
                stamps_per_benchmark[benchmark_name] = asinh_transform(stamps, asinh_scale)
            print(f"\nSeed {seed}, {benchmark_name}: {len(benchmark_table):,} stamps "
                  f"({int(benchmark_table.label.sum())} lenses)", flush=True)
            for initialisation in settings.init:
                print(f"  {initialisation} ({INITIALISATIONS[initialisation]})", flush=True)
                row, choices = run_initialisation(benchmark_name, benchmark_table,
                                                  stamps_per_benchmark[benchmark_name], initialisation, seed, context)
                replace_rows(output_folder / "summary.csv", [row], ["benchmark", "init", "seed"])
                choices_table = replace_rows(output_folder / "learning_rate_choices.csv", choices,
                                             ["benchmark", "init", "seed", "fold"])
                print(f"  {initialisation}: AUROC {row['auroc']:.4f}; lenses found {row['lenses_found']} of "
                      f"{row['lenses']}; false alarms {row['false_alarms']} of {row['nonlenses']}", flush=True)
    write_checks(choices_table, settings, output_folder)
    print(f"Results saved in {output_folder}")


if __name__ == "__main__":
    main()
