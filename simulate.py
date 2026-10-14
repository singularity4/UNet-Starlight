"""simulate.py - simulated Euclid VIS stamps of galaxy-scale strong lenses and non-lenses, with arc masks and
Einstein radii

UNet-Starlight.

Two data sets are simulated with one forward model:
    benchmark    the simulated benchmark: 1,000 lenses and 1,000 non-lenses with Einstein radii from 0.6" to 2.0".
                 The U-Nets are trained and tested on it (train_simulations.py).
    radius-set   the second simulated set: 5,000 lenses and 5,000 non-lenses with Einstein radii from 0.3" to 4.5"
                 and neighbouring galaxies. UNet-Radius-mini is trained on it before its Einstein radius output is
                 applied to the real lenses (einstein_radius.py).

Forward model (lenstronomy)
    Lens galaxy light   elliptical Sersic profile: magnitude 19-22, Sersic index 3-5, Sersic radius 0.5"-1.5".
    Lens mass           singular isothermal ellipsoid (SIE) whose ellipticity is that of the light plus a scatter of
                        0.03 in each component, and an external shear with components drawn from N(0, 0.03^2).
                        Non-lenses have an Einstein radius of zero, so that no light is deflected.
    Source light        elliptical Sersic profile with Sersic index 1-2.5. The source of a lens lies within half an
                        Einstein radius of the lens centre, so that it is imaged into arcs or a ring.
    Observation         the Euclid VIS settings of lenstronomy: Gaussian point-spread function, 0.101" pixels,
                        stamps of 101 x 101 pixels, light profiles evaluated on a grid three times finer than the
                        pixels, zero point 25.72 (electrons per second). Noise: Gaussian background noise of the sky
                        and the detector at the depth of the six-year survey, and Poisson noise of the source and
                        galaxy light. No sky level is added, as in the background-subtracted Q1 mosaics.
    Ellipticity components (e1, e2) are drawn from a normal distribution and limited to +-0.3. All lengths are in
    arcseconds.

Arc mask
    A pixel belongs to the arc mask if the noise-free light of the lensed source, after convolution with the
    point-spread function, exceeds the standard deviation of the background noise in one pixel (0.0109 electrons per
    second). The mask of a non-lens is empty. The threshold includes arc light fainter than the arcs visible in the
    noisy image, so the masks are wider than the visible arcs.

Differences between the two sets
                        benchmark                                   radius-set
    Einstein radius     uniform in 0.6"-2.0"                        uniform in log(theta_E), 0.3"-4.5"
    non-lenses          in half of the cases an unlensed companion   the lens galaxy and its neighbours only
                        1.5"-4" from the centre (magnitude 21-24,
                        Sersic radius 0.2"-0.6")
    neighbours          none                                        0, 1 or 2 (probabilities 0.4, 0.4 and 0.2) in
                                                                    lenses and non-lenses alike: offset 1"-5",
                                                                    magnitude 20-25, Sersic radius 0.2"-0.8",
                                                                    Sersic index 0.5-4
    The source of a lens has magnitude 22-25 and Sersic radius 0.1"-0.4" in both sets. In the benchmark a second
    galaxy occurs only in non-lenses; in the radius set neighbours occur in both classes, so that their presence
    carries no information about the label.

Reproducibility
    The parameters of all systems are drawn from one random generator (--seed, default 42) in a fixed order, first
    all lenses and then all non-lenses; the stamps are then shuffled with the same generator. Benchmark: the noise is
    drawn from NumPy's global random generator, seeded once with --noise-seed (default 1), one stamp after another.
    Radius set: the noise of stamp i is drawn with the seed (--noise-seed, i), so that the stamps can be simulated in
    parallel (--workers) and the result does not depend on the number of workers.

    The parameter seed 42 gave unet_lenses_2000.npz, the simulated benchmark of the paper.

Output (.npz, read by train_simulations.py)
    images      (N, 101, 101) float32, electrons per second (zero point 25.72)
    mask        (N, 101, 101) uint8, 1 = arc pixel
    label       (N,) 1 = lens, 0 = non-lens
    theta_E     (N,) Einstein radius in arcseconds, 0 for non-lenses
    neighbours  (N,) number of neighbouring galaxies (radius set only)

Usage
    python simulate.py benchmark --out unet_lenses_2000.npz
    python simulate.py benchmark --n-lens 100 --n-nonlens 100 --out unet_lenses_200.npz      # small set for tests
    python simulate.py radius-set --out radius_training_10000.npz
"""

import argparse
import multiprocessing
import os
import time

import numpy as np
from lenstronomy.SimulationAPI.ObservationConfig.Euclid import Euclid
from lenstronomy.SimulationAPI.sim_api import SimAPI

STAMP_SIZE_PIXELS = 101                 # 0.101" per pixel in the Euclid VIS settings of lenstronomy
SUPERSAMPLING_FACTOR = 3                # light profiles are evaluated on a grid three times finer than the pixels
MAXIMUM_ELLIPTICITY_COMPONENT = 0.3
ABSENT_MAGNITUDE = 99.0                 # a magnitude so faint that the component contributes no light
MAXIMUM_NEIGHBOURS = 2                  # radius set


# ======================================================================== 1. forward model, shared by both sets

def build_simulator(number_of_unlensed_components):
    """The lenstronomy simulator of Euclid VIS stamps: an SIE lens with external shear, one lensed source, and the
    given number of unlensed light components (the lens galaxy and, in the radius set, its neighbours).
    Returns (simulation, image_model)."""
    observation = Euclid(band="VIS", psf_type="GAUSSIAN", coadd_years=6).kwargs_single_band()
    model_components = {"lens_model_list": ["SIE", "SHEAR"],
                        "lens_light_model_list": ["SERSIC_ELLIPSE"] * number_of_unlensed_components,
                        "source_light_model_list": ["SERSIC_ELLIPSE"]}
    simulation = SimAPI(num_pix=STAMP_SIZE_PIXELS, kwargs_single_band=observation, kwargs_model=model_components)
    image_model = simulation.image_model_class({"supersampling_factor": SUPERSAMPLING_FACTOR})
    return simulation, image_model


def random_ellipticity(random_generator, spread):
    """Two ellipticity components, each drawn from a normal distribution with standard deviation `spread` and
    limited to +-0.3."""
    first_component = float(np.clip(random_generator.normal(0, spread),
                                    -MAXIMUM_ELLIPTICITY_COMPONENT, MAXIMUM_ELLIPTICITY_COMPONENT))
    second_component = float(np.clip(random_generator.normal(0, spread),
                                     -MAXIMUM_ELLIPTICITY_COMPONENT, MAXIMUM_ELLIPTICITY_COMPONENT))
    return first_component, second_component


def draw_lens_galaxy(random_generator, is_lens, draw_einstein_radius):
    """Light and mass of the central galaxy, drawn in a fixed order. The mass follows the light: the ellipticity of
    the SIE is that of the light plus a scatter of 0.03 in each component. draw_einstein_radius(random_generator)
    gives the Einstein radius of a lens; a non-lens has an Einstein radius of zero.
    Returns (light, mass, Einstein radius), light and mass as lenstronomy keyword arguments."""
    uniform = random_generator.uniform
    light_e1, light_e2 = random_ellipticity(random_generator, spread=0.1)
    light = {"magnitude": uniform(19, 22), "R_sersic": uniform(0.5, 1.5), "n_sersic": uniform(3, 5),
             "e1": light_e1, "e2": light_e2, "center_x": 0.0, "center_y": 0.0}
    einstein_radius = draw_einstein_radius(random_generator) if is_lens else 0.0
    mass_e1 = light_e1 + random_generator.normal(0, 0.03)
    mass_e2 = light_e2 + random_generator.normal(0, 0.03)
    mass = [{"theta_E": einstein_radius, "e1": mass_e1, "e2": mass_e2, "center_x": 0.0, "center_y": 0.0},
            {"gamma1": random_generator.normal(0, 0.03), "gamma2": random_generator.normal(0, 0.03)}]
    return light, mass, einstein_radius


def source_light(random_generator, offset, angle, magnitude, sersic_radius, ellipticity):
    """Keyword arguments of an elliptical Sersic source at the given offset and position angle from the centre.
    The Sersic index (1-2.5) is drawn here."""
    return {"magnitude": magnitude, "R_sersic": sersic_radius, "n_sersic": random_generator.uniform(1, 2.5),
            "e1": ellipticity[0], "e2": ellipticity[1],
            "center_x": offset * np.cos(angle), "center_y": offset * np.sin(angle)}


# ============================================================================ 2. simulated benchmark

def draw_benchmark_system(random_generator, is_lens):
    """Parameters of one system of the simulated benchmark, drawn in a fixed order."""
    uniform = random_generator.uniform
    lens_light, lens_mass, einstein_radius = draw_lens_galaxy(random_generator, is_lens,
                                                              lambda generator: generator.uniform(0.6, 2.0))
    source_ellipticity = random_ellipticity(random_generator, spread=0.2)
    source_angle = uniform(0, 2 * np.pi)
    if is_lens:
        # background source within half an Einstein radius of the centre, imaged into arcs or a ring
        source_offset = uniform(0, 0.5) * einstein_radius
        source_magnitude = uniform(22, 25)
        source_radius = uniform(0.1, 0.4)
    else:
        # an unlensed companion galaxy in half of the non-lenses; otherwise no second galaxy
        source_offset = uniform(1.5, 4.0)
        source_magnitude = uniform(21, 24) if random_generator.random() < 0.5 else ABSENT_MAGNITUDE
        source_radius = uniform(0.2, 0.6)
    source = source_light(random_generator, source_offset, source_angle, source_magnitude, source_radius,
                          source_ellipticity)
    return {"lens_light": [lens_light], "lens_mass": lens_mass, "source_light": [source],
            "theta_E": einstein_radius, "label": int(is_lens)}


def simulate_benchmark(settings):
    """Observed stamps, arc masks, labels and Einstein radii of the simulated benchmark."""
    random_generator = np.random.default_rng(settings.seed)
    systems = [draw_benchmark_system(random_generator, is_lens)
               for is_lens, count in ((True, settings.n_lens), (False, settings.n_nonlens)) for _ in range(count)]
    order = random_generator.permutation(len(systems))
    np.random.seed(settings.noise_seed)          # lenstronomy draws the noise from NumPy's global random generator

    simulation, image_model = build_simulator(number_of_unlensed_components=1)
    images, source_images = [], []
    for system in systems:
        lens_light, source, _ = simulation.magnitude2amplitude(system["lens_light"], system["source_light"])
        noise_free_image = image_model.image(system["lens_mass"], source, lens_light)
        images.append((noise_free_image + simulation.noise_for_model(model=noise_free_image)).astype(np.float32))
        source_images.append(image_model.image(system["lens_mass"], source, lens_light, lens_light_add=False))

    labels = np.array([system["label"] for system in systems])[order]
    source_images = np.stack(source_images)[order].astype(np.float32)
    is_lens_stamp = (labels == 1)[:, None, None]
    arc_mask = ((source_images > float(simulation.background_noise)) & is_lens_stamp).astype(np.uint8)
    return {"images": np.stack(images)[order], "mask": arc_mask, "label": labels,
            "theta_E": np.array([system["theta_E"] for system in systems])[order]}


# ============================================================================ 3. second simulated set (radius set)

def draw_radius_set_einstein_radius(random_generator, settings):
    """Einstein radius between --theta-e-min and --theta-e-max, uniform in log(theta_E) or in theta_E."""
    if settings.theta_e_distribution == "log-uniform":
        return float(np.exp(random_generator.uniform(np.log(settings.theta_e_min), np.log(settings.theta_e_max))))
    return float(random_generator.uniform(settings.theta_e_min, settings.theta_e_max))


def draw_radius_set_system(random_generator, is_lens, settings):
    """Parameters of one system of the radius set, drawn in a fixed order."""
    uniform = random_generator.uniform
    lens_light, lens_mass, einstein_radius = draw_lens_galaxy(
        random_generator, is_lens, lambda generator: draw_radius_set_einstein_radius(generator, settings))
    source_ellipticity = random_ellipticity(random_generator, spread=0.2)
    source_angle = uniform(0, 2 * np.pi)
    source_offset = uniform(0, 0.5) * einstein_radius
    source_magnitude = uniform(22, 25) if is_lens else ABSENT_MAGNITUDE        # non-lenses contain no source
    source = source_light(random_generator, source_offset, source_angle, source_magnitude, uniform(0.1, 0.4),
                          source_ellipticity)

    # Neighbouring galaxies: position, shape and size are drawn for both possible neighbours; a neighbour that is
    # not present has the magnitude ABSENT_MAGNITUDE.
    number_of_neighbours = int(random_generator.choice(len(settings.neighbour_probabilities),
                                                       p=settings.neighbour_probabilities))
    unlensed_light = [lens_light]
    for neighbour in range(MAXIMUM_NEIGHBOURS):
        neighbour_e1, neighbour_e2 = random_ellipticity(random_generator, spread=0.2)
        offset, angle = uniform(1.0, 5.0), uniform(0, 2 * np.pi)
        magnitude = uniform(20, 25) if neighbour < number_of_neighbours else ABSENT_MAGNITUDE
        unlensed_light.append({"magnitude": magnitude, "R_sersic": uniform(0.2, 0.8), "n_sersic": uniform(0.5, 4),
                               "e1": neighbour_e1, "e2": neighbour_e2,
                               "center_x": offset * np.cos(angle), "center_y": offset * np.sin(angle)})
    return {"lens_light": unlensed_light, "lens_mass": lens_mass, "source_light": [source],
            "theta_E": einstein_radius, "label": int(is_lens), "neighbours": number_of_neighbours}


worker_simulator = None                 # one simulator in each worker process


def start_worker():
    global worker_simulator
    worker_simulator = build_simulator(number_of_unlensed_components=1 + MAXIMUM_NEIGHBOURS)


def simulate_radius_set_stamp(task):
    """Observed stamp and arc mask of one system of the radius set. task = (index, system, noise_seed)."""
    index, system, noise_seed = task
    simulation, image_model = worker_simulator
    lens_light, source, _ = simulation.magnitude2amplitude(system["lens_light"], system["source_light"])
    source_image = image_model.image(system["lens_mass"], source, lens_light, lens_light_add=False)
    galaxy_image = image_model.image(system["lens_mass"], source, lens_light, source_add=False)
    noise_free_image = source_image + galaxy_image
    np.random.seed([noise_seed, index])          # lenstronomy draws the noise from NumPy's global random generator
    image = noise_free_image + simulation.noise_for_model(model=noise_free_image)
    arc_mask = (source_image > float(simulation.background_noise)) & bool(system["label"])
    return index, image.astype(np.float32), arc_mask.astype(np.uint8)


def simulate_radius_set(settings):
    """Observed stamps, arc masks, labels, Einstein radii and numbers of neighbours of the radius set."""
    random_generator = np.random.default_rng(settings.seed)
    systems = [draw_radius_set_system(random_generator, is_lens, settings)
               for is_lens, count in ((True, settings.n_lens), (False, settings.n_nonlens)) for _ in range(count)]
    order = random_generator.permutation(len(systems))
    tasks = [(index, system, settings.noise_seed) for index, system in enumerate(systems)]

    images = np.empty((len(systems), STAMP_SIZE_PIXELS, STAMP_SIZE_PIXELS), np.float32)
    masks = np.empty(images.shape, np.uint8)
    workers = max(1, settings.workers)
    start = time.time()
    if workers == 1:
        start_worker()
        results = map(simulate_radius_set_stamp, tasks)
    else:
        pool = multiprocessing.get_context("fork").Pool(workers, initializer=start_worker)
        results = pool.imap_unordered(simulate_radius_set_stamp, tasks, chunksize=16)
    for finished, (index, image, mask) in enumerate(results, start=1):
        images[index], masks[index] = image, mask
        if finished % 2000 == 0 or finished == len(tasks):
            elapsed = time.time() - start
            print(f"  {finished:,} of {len(tasks):,} stamps, {elapsed / 60:.1f} min "
                  f"(about {elapsed / finished * (len(tasks) - finished) / 60:.0f} min left)", flush=True)
    if workers > 1:
        pool.close()
        pool.join()

    return {"images": images[order], "mask": masks[order],
            "label": np.array([system["label"] for system in systems])[order],
            "theta_E": np.array([system["theta_E"] for system in systems], np.float32)[order],
            "neighbours": np.array([system["neighbours"] for system in systems], np.int8)[order]}


# ============================================================================ 4. command line

def parse_arguments():
    argument_parser = argparse.ArgumentParser(description="Simulated Euclid VIS stamps of lenses and non-lenses")
    data_sets = argument_parser.add_subparsers(dest="data_set", required=True)

    benchmark = data_sets.add_parser("benchmark", help="the simulated benchmark")
    benchmark.add_argument("--n-lens", type=int, default=1000, help="number of lenses (default 1000)")
    benchmark.add_argument("--n-nonlens", type=int, default=1000, help="number of non-lenses (default 1000)")
    benchmark.add_argument("--seed", type=int, default=42, help="seed of the parameters (default 42)")
    benchmark.add_argument("--noise-seed", type=int, default=1, help="seed of the noise (default 1)")
    benchmark.add_argument("--out", default="unet_lenses_2000.npz", help="output file")

    radius_set = data_sets.add_parser("radius-set", help="the second simulated set")
    radius_set.add_argument("--n-lens", type=int, default=5000, help="number of lenses (default 5000)")
    radius_set.add_argument("--n-nonlens", type=int, default=5000, help="number of non-lenses (default 5000)")
    radius_set.add_argument("--theta-e-min", type=float, default=0.3, help="smallest Einstein radius (default 0.3\")")
    radius_set.add_argument("--theta-e-max", type=float, default=4.5, help="largest Einstein radius (default 4.5\")")
    radius_set.add_argument("--theta-e-distribution", choices=["log-uniform", "uniform"], default="log-uniform")
    radius_set.add_argument("--neighbour-probabilities", type=float, nargs=MAXIMUM_NEIGHBOURS + 1,
                            default=[0.4, 0.4, 0.2], help="probabilities of 0, 1 and 2 neighbours")
    radius_set.add_argument("--seed", type=int, default=42, help="seed of the parameters (default 42)")
    radius_set.add_argument("--noise-seed", type=int, default=1, help="seed of the noise (default 1)")
    radius_set.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="parallel processes")
    radius_set.add_argument("--out", default="radius_training_10000.npz", help="output file")

    settings = argument_parser.parse_args()
    if settings.data_set == "radius-set":
        settings.neighbour_probabilities = np.asarray(settings.neighbour_probabilities, dtype=float)
        if abs(settings.neighbour_probabilities.sum() - 1) > 1e-6:
            raise SystemExit("--neighbour-probabilities must add up to 1")
    return settings


def main():
    settings = parse_arguments()
    start = time.time()
    if settings.data_set == "benchmark":
        arrays = simulate_benchmark(settings)
    else:
        arrays = simulate_radius_set(settings)
    np.savez_compressed(settings.out, **arrays)
    lens_radii = arrays["theta_E"][arrays["label"] == 1]
    print(f"Saved {settings.out}: {len(arrays['label']):,} stamps ({int(arrays['label'].sum()):,} lenses), "
          f"Einstein radii {lens_radii.min():.2f}\"-{lens_radii.max():.2f}\" (median {np.median(lens_radii):.2f}\"), "
          f"{(time.time() - start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
