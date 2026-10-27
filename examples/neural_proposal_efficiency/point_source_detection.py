"""Compare source-count inference in simulated images with a numerical reference."""

import argparse
import hashlib
import json
import platform
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.special import expit, gammaln, logsumexp, ndtr, ndtri

from pcat import NeuralElementProposal, forward_model_image
from pcat.transdimensional import sample_catalog
from pcat.diagnostics import autocorrelation_time, gelman_rubin
from tdpy.verbosity import print

IMAGE_SHAPE = (9, 9)
SOURCE_PIXELS = ((3, 4), (5, 4))
PSF_SIGMA = 1.0  # [pixel]
MAXIMUM_FLUX = 500.0  # [count]
BACKGROUND_BOUNDS = (90.0, 110.0)  # [count pixel^-1]
PIXEL_NOISE = 10.0  # [count pixel^-1]
MAXIMUM_SOURCES = 2


def image_features(residual, templates):
    centered_templates = templates - templates.mean(axis=0)
    return np.asarray(residual).ravel() @ centered_templates / PIXEL_NOISE**2


class PointSourceSelection:
    """Evaluate fixed network weights and integrate the unused within-pixel coordinate."""

    def __init__(self, network, position_count=2):
        if network.condition_dimension < 1 or network.element_dimension != 2 or position_count < 1:
            raise ValueError("Point-source selection requires image features and two source parameters.")
        self.position_count = int(position_count)
        self.layers = [(layer.weight.detach().cpu().numpy().astype(float),
                        layer.bias.detach().cpu().numpy().astype(float))
                       for layer in network.network if hasattr(layer, "weight")]
        self.mean = np.asarray(network.condition_mean, dtype=float)
        self.scale = np.asarray(network.condition_scale, dtype=float)
        self.components = network.number_components
        self.uniform_fraction = network.uniform_fraction
        self.minimum_scale = network.minimum_scale
        self.cache = OrderedDict()

    def parameters(self, features):
        features = np.asarray(features, dtype=float)
        if features.shape != self.mean.shape or not np.isfinite(features).all():
            raise ValueError("Image features must be a finite vector matching the network inputs.")
        key = features.tobytes()
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        output = (features - self.mean) / self.scale
        for index, (weight, bias) in enumerate(self.layers):
            output = weight @ output + bias
            if index < len(self.layers) - 1:
                output = output * expit(output)
        count = self.components
        weights = np.exp(output[:count] - logsumexp(output[:count]))
        means = expit(output[count:3 * count].reshape(count, 2))
        scales = self.minimum_scale + (0.5 - self.minimum_scale) * expit(output[3 * count:5 * count].reshape(count, 2))
        lower = ndtr(-means / scales)
        upper = ndtr((1.0 - means) / scales)
        edges = np.linspace(0.0, 1.0, self.position_count + 1)
        cumulative = ndtr((edges[None, :] - means[:, 0, None]) / scales[:, 0, None])
        pixel_probabilities = np.diff(cumulative, axis=1) / (upper[:, 0] - lower[:, 0])[:, None]
        parameters = (weights, means, scales, lower, upper, pixel_probabilities)
        self.cache[key] = parameters
        if len(self.cache) > 128:
            self.cache.popitem(last=False)
        return parameters

    def sample(self, features, generator):
        if generator.random() < self.uniform_fraction:
            return generator.random(2)
        weights, means, scales, lower, upper, pixel_probabilities = self.parameters(features)
        component = min(int(np.searchsorted(np.cumsum(weights), generator.random())), self.components - 1)
        pixel = min(np.searchsorted(np.cumsum(pixel_probabilities[component]), generator.random()), self.position_count - 1)
        coordinate = (pixel + generator.random()) / self.position_count
        probability = lower[component, 1] + generator.random() * (upper[component, 1] - lower[component, 1])
        probability = np.clip(probability, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
        flux = means[component, 1] + scales[component, 1] * ndtri(probability)
        return np.array([coordinate, flux])

    def log_prob(self, features, unit):
        unit = np.asarray(unit, dtype=float)
        if unit.shape != (2,) or not np.isfinite(unit).all() or np.any((unit < 0.0) | (unit > 1.0)):
            return -np.inf
        weights, means, scales, lower, upper, pixel_probabilities = self.parameters(features)
        pixel = min(int(unit[0] * self.position_count), self.position_count - 1)
        pixel_mass = pixel_probabilities[:, pixel]
        flux_density = np.exp(-0.5 * ((unit[1] - means[:, 1]) / scales[:, 1])**2)
        flux_density /= np.sqrt(2.0 * np.pi) * scales[:, 1] * (upper[:, 1] - lower[:, 1])
        density = self.position_count * np.dot(weights * pixel_mass, flux_density)
        return float(np.log((1.0 - self.uniform_fraction) * density + self.uniform_fraction))


class PixelSourceSelection:
    """Predict a discrete search pixel and its normalized conditional flux density."""

    @classmethod
    def fit(cls, conditions, targets, position_count, epochs, seed):
        import torch
        from torch import nn

        conditions = np.asarray(conditions, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        mean, scale = conditions.mean(axis=0), conditions.std(axis=0)
        scale[scale == 0.0] = 1.0
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            network = nn.Sequential(nn.Linear(conditions.shape[1], 64), nn.SiLU(),
                                    nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 3 * position_count))
        optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
        tensor = torch.as_tensor((conditions - mean) / scale)
        pixels = torch.as_tensor(np.minimum((targets[:, 0] * position_count).astype(int), position_count - 1))
        flux = torch.as_tensor(targets[:, 1])
        generator = torch.Generator().manual_seed(seed + 1)
        for _ in range(epochs):
            for indices in torch.randperm(len(conditions), generator=generator).split(256):
                output = network(tensor[indices])
                logits, means_raw, scales_raw = output.split(position_count, dim=1)
                rows = torch.arange(len(indices))
                selected = pixels[indices]
                center = torch.sigmoid(means_raw[rows, selected])
                spread = 0.005 + 0.495 * torch.sigmoid(scales_raw[rows, selected])
                normalizer = torch.special.ndtr((1.0 - center) / spread) - torch.special.ndtr(-center / spread)
                log_density = (torch.log_softmax(logits, dim=1)[rows, selected]
                               - 0.5 * ((flux[indices] - center) / spread)**2
                               - torch.log(spread * np.sqrt(2.0 * np.pi) * normalizer))
                joint = torch.logaddexp(log_density + np.log(0.9), torch.full_like(log_density, np.log(0.1 / position_count)))
                loss = -joint.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 10.0)
                optimizer.step()
        result = cls()
        result.network, result.mean, result.scale = network.eval(), mean, scale
        result.position_count, result.cache = position_count, OrderedDict()
        result.layers = [(layer.weight.detach().numpy().astype(float), layer.bias.detach().numpy().astype(float))
                         for layer in network if hasattr(layer, "weight")]
        return result

    def parameters(self, features):
        features = np.asarray(features, dtype=float)
        key = features.tobytes()
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        output = (features - self.mean) / self.scale
        for index, (weight, bias) in enumerate(self.layers):
            output = weight @ output + bias
            if index < len(self.layers) - 1:
                output = output * expit(output)
        logits, center, spread = np.split(output, 3)
        weights = np.exp(logits - logsumexp(logits))
        center = expit(center)
        spread = 0.005 + 0.495 * expit(spread)
        lower, upper = ndtr(-center / spread), ndtr((1.0 - center) / spread)
        result = weights, center, spread, lower, upper
        self.cache[key] = result
        if len(self.cache) > 128:
            self.cache.popitem(last=False)
        return result

    def sample(self, features, generator):
        if generator.random() < 0.1:
            return generator.random(2)
        weights, center, spread, lower, upper = self.parameters(features)
        pixel = min(int(np.searchsorted(np.cumsum(weights), generator.random())), self.position_count - 1)
        quantile = np.clip(lower[pixel] + generator.random() * (upper[pixel] - lower[pixel]),
                           np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
        flux = center[pixel] + spread[pixel] * ndtri(quantile)
        return np.array([(pixel + generator.random()) / self.position_count, flux])

    def log_prob(self, features, unit):
        weights, center, spread, lower, upper = self.parameters(features)
        pixel = min(int(unit[0] * self.position_count), self.position_count - 1)
        flux_density = np.exp(-0.5 * ((unit[1] - center[pixel]) / spread[pixel])**2)
        flux_density /= np.sqrt(2.0 * np.pi) * spread[pixel] * (upper[pixel] - lower[pixel])
        return float(np.log(0.9 * self.position_count * weights[pixel] * flux_density + 0.1))

    def save(self, output):
        import torch
        print(f"Writing to {output}...")
        torch.save({"state_dict": self.network.state_dict(), "condition_mean": self.mean.tolist(),
                    "condition_scale": self.scale.tolist(), "position_count": self.position_count,
                    "distribution": "categorical pixel and truncated Gaussian flux"}, output)


def effective_samples(values):
    values = np.asarray(values, dtype=float)
    if values.size < 2 or np.all(values == values[0]):
        return 0.0
    correlation_time = float(autocorrelation_time(values)[1])
    return float(min(values.size, values.size / correlation_time)) if np.isfinite(correlation_time) else 0.0


def finite_rhat(traces):
    value = float(gelman_rubin(np.column_stack(traces))) if len(traces) > 1 else np.nan
    return value if np.isfinite(value) else None


class NeuralCatalogSelection:
    """Generate count and ordered sources, then sample the conditional background."""

    @classmethod
    def fit(cls, sample_count, epochs, seed, source_pixels=None):
        import torch
        from torch import nn

        random = np.random.default_rng(seed)
        source_pixels = tuple(SOURCE_PIXELS if source_pixels is None else source_pixels)
        templates = point_source_templates(source_pixels)
        image_conditions, labels, source_conditions, targets = [], [], [], []
        for _ in range(sample_count):
            count = int(random.choice(3, p=[0.4, 0.4, 0.2]))
            unit = random.random((count, 2))
            catalog = unit * np.array([1.0, MAXIMUM_FLUX])
            observed = source_image(catalog, templates) + random.normal(0.0, PIXEL_NOISE, templates.shape[0])
            image_conditions.append(image_features(observed, templates))
            labels.append(count)
            for index in range(count):
                residual = observed - source_image(catalog[:index], templates)
                source_conditions.append(np.r_[image_features(residual, templates), (count - index) / MAXIMUM_SOURCES])
                targets.append(unit[index])
        conditions = np.asarray(image_conditions, dtype=np.float32)
        mean, scale = conditions.mean(axis=0), conditions.std(axis=0)
        scale[scale == 0.0] = 1.0
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 1)
            count_network = nn.Sequential(nn.Linear(templates.shape[1], 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 3))
        optimizer = torch.optim.AdamW(count_network.parameters(), lr=1e-3)
        tensor = torch.as_tensor((conditions - mean) / scale)
        label_tensor = torch.as_tensor(labels)
        generator = torch.Generator().manual_seed(seed + 2)
        for _ in range(epochs):
            for indices in torch.randperm(sample_count, generator=generator).split(256):
                loss = nn.functional.cross_entropy(count_network(tensor[indices]), label_tensor[indices])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        if len(source_pixels) > 2:
            component_network = PixelSourceSelection.fit(source_conditions, targets, len(source_pixels), epochs, seed + 3)
        else:
            component_network = NeuralElementProposal.fit(
                np.asarray(source_conditions), np.asarray(targets), number_components=8,
                hidden_sizes=(32, 32), epochs=epochs, batch_size=256, uniform_fraction=0.1, seed=seed + 3,
            )
        result = cls()
        result.count_layers = [(layer.weight.detach().numpy().astype(float), layer.bias.detach().numpy().astype(float))
                               for layer in count_network if hasattr(layer, "weight")]
        result.count_mean, result.count_scale = mean, scale
        result.sources = component_network if len(source_pixels) > 2 else PointSourceSelection(component_network, len(source_pixels))
        result.source_pixels = source_pixels
        result.component_network = component_network
        return result

    def bind(self, observed, *, learned=True):
        source_pixels = getattr(self, "source_pixels", SOURCE_PIXELS)
        templates = point_source_templates(source_pixels)
        observations = np.asarray(observed).ravel()
        initial_features = image_features(observations, templates)
        gram = templates.T @ (templates - templates.mean(axis=0)) / PIXEL_NOISE**2
        probability = np.array([0.4, 0.4, 0.2])
        if learned:
            output = (initial_features - self.count_mean) / self.count_scale
            for index, (weight, bias) in enumerate(self.count_layers):
                output = weight @ output + bias
                if index < len(self.count_layers) - 1:
                    output = output * expit(output)
            probability = 0.95 * np.exp(output - logsumexp(output)) + 0.05 * probability
        background_scale = PIXEL_NOISE / np.sqrt(observations.size)
        template_means = templates.mean(axis=0)
        observed_mean = observations.mean()
        background_cache = OrderedDict()
        count_cumulative = np.cumsum(probability)

        def features(catalog, remaining):
            feature = np.empty(len(source_pixels) + 1)
            feature[:-1] = initial_features
            feature[-1] = remaining / MAXIMUM_SOURCES
            for coordinate, source_flux in catalog:
                pixel = min(int(coordinate * len(source_pixels)), len(source_pixels) - 1)
                feature[:-1] -= source_flux * gram[:, pixel]
            return feature

        def background_parameters(catalog):
            key = catalog.tobytes()
            if key in background_cache:
                return background_cache[key]
            mean = observed_mean
            for coordinate, source_flux in catalog:
                pixel = min(int(coordinate * len(source_pixels)), len(source_pixels) - 1)
                mean -= source_flux * template_means[pixel]
            lower = ndtr((BACKGROUND_BOUNDS[0] - mean) / background_scale)
            upper = ndtr((BACKGROUND_BOUNDS[1] - mean) / background_scale)
            value = mean, lower, upper
            background_cache[key] = value
            if len(background_cache) > 128:
                background_cache.popitem(last=False)
            return value

        def draw(generator):
            count = min(int(np.searchsorted(count_cumulative, generator.random())), MAXIMUM_SOURCES)
            catalog = np.empty((0, 2))
            for index in range(count):
                unit = (self.sources.sample(features(catalog, count - index), generator)
                        if learned else generator.random(2))
                catalog = np.vstack((catalog, unit * np.array([1.0, MAXIMUM_FLUX])))
            mean, lower, upper = background_parameters(catalog)
            quantile = np.clip(lower + generator.random() * (upper - lower), np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
            background = mean + background_scale * ndtri(quantile)
            return np.array([background]), catalog

        def density(base, catalog):
            count = len(catalog)
            log_probability = np.log(probability[count])
            for index, component in enumerate(catalog):
                unit = component / np.array([1.0, MAXIMUM_FLUX])
                source_density = self.sources.log_prob(features(catalog[:index], count - index), unit) if learned else 0.0
                log_probability += source_density - np.log(MAXIMUM_FLUX)
            mean, lower, upper = background_parameters(catalog)
            log_probability += (-0.5 * ((base[0] - mean) / background_scale)**2
                                - np.log(np.sqrt(2.0 * np.pi) * background_scale * (upper - lower)))
            return float(log_probability)

        return draw, density


def point_source_templates(source_pixels=None):
    templates = []
    for column, row in SOURCE_PIXELS if source_pixels is None else source_pixels:
        image = np.zeros(IMAGE_SHAPE)
        image[row, column] = 1.0
        templates.append(forward_model_image(image, PSF_SIGMA)["observed_image"].ravel())
    return np.column_stack(templates)


def source_image(catalog, templates):
    catalog = np.asarray(catalog, dtype=float).reshape(-1, 2)
    if not len(catalog):
        return np.zeros(templates.shape[0])
    indices = np.minimum((catalog[:, 0] * templates.shape[1]).astype(int), templates.shape[1] - 1)
    return templates[:, indices] @ catalog[:, 1]


def simulate_image(seed, catalog, background=100.0, source_pixels=None):
    """Return Gaussian-noise pixel counts for sources selected at two known positions."""
    random = np.random.default_rng(seed)
    templates = point_source_templates(source_pixels)
    expected = background + source_image(catalog, templates)
    observed = expected + random.normal(0.0, PIXEL_NOISE, expected.size)
    return observed.reshape(IMAGE_SHAPE), expected.reshape(IMAGE_SHAPE)


def marginalized_background_log_likelihood(observed, signals):
    residual = np.asarray(observed).ravel() - np.asarray(signals)
    mean = residual.mean(axis=-1)
    centered = residual - mean[..., None]
    scale = PIXEL_NOISE / np.sqrt(residual.shape[-1])
    lower, upper = BACKGROUND_BOUNDS
    probability = ndtr((upper - mean) / scale) - ndtr((lower - mean) / scale)
    with np.errstate(divide="ignore"):
        return (-0.5 * np.sum((centered / PIXEL_NOISE) ** 2, axis=-1)
                + np.log(probability) + np.log(np.sqrt(2.0 * np.pi) * scale / (upper - lower)))


def reference_count_probabilities(observed, grid_size=101, source_pixels=None):
    """Integrate fluxes and background for a truncated Poisson(1) source-count prior.

    Each source independently chooses one of two candidate positions with equal
    probability. Two sources may share a candidate position, just as in the fitted
    catalog model. Their fluxes are independent uniform values in [0, MAXIMUM_FLUX].
    """
    if isinstance(grid_size, bool) or int(grid_size) != grid_size or grid_size < 2:
        raise ValueError("grid_size must be an integer of at least two.")
    templates = point_source_templates(source_pixels)
    flux = np.linspace(0.0, MAXIMUM_FLUX, int(grid_size))
    weights = np.ones(flux.size) / (flux.size - 1)
    weights[[0, -1]] *= 0.5
    log_weights = np.log(weights)
    observations = np.asarray(observed).ravel()
    pixel_count = observations.size
    observed_sum, observed_square = observations.sum(), observations @ observations
    sums, products = templates.sum(axis=0), templates.T @ observations
    gram = templates.T @ templates

    def integrated(first, second, first_flux, second_flux):
        residual_sum = observed_sum - first_flux * sums[first] - second_flux * sums[second]
        residual_square = (observed_square - 2.0 * first_flux * products[first] - 2.0 * second_flux * products[second]
                           + first_flux**2 * gram[first, first] + second_flux**2 * gram[second, second]
                           + 2.0 * first_flux * second_flux * gram[first, second])
        mean = residual_sum / pixel_count
        scale = PIXEL_NOISE / np.sqrt(pixel_count)
        probability = ndtr((BACKGROUND_BOUNDS[1] - mean) / scale) - ndtr((BACKGROUND_BOUNDS[0] - mean) / scale)
        with np.errstate(divide="ignore"):
            return (-0.5 * (residual_square - residual_sum**2 / pixel_count) / PIXEL_NOISE**2
                    + np.log(probability) + np.log(np.sqrt(2.0 * np.pi) * scale / np.diff(BACKGROUND_BOUNDS)[0]))

    evidence = [float(integrated(0, 0, 0.0, 0.0))]
    single = [integrated(index, index, flux, 0.0) for index in range(templates.shape[1])]
    evidence.append(float(logsumexp(np.asarray(single) + log_weights) - np.log(templates.shape[1])))
    pairs = []
    for first in range(templates.shape[1]):
        for second in range(first, templates.shape[1]):
            likelihood = integrated(first, second, flux[:, None], flux[None, :])
            multiplicity = 1.0 if first == second else 2.0
            pairs.append(logsumexp(likelihood + log_weights[:, None] + log_weights[None, :]) + np.log(multiplicity))
    evidence.append(float(logsumexp(pairs) - 2.0 * np.log(templates.shape[1])))
    log_probability = np.asarray(evidence) - gammaln(np.arange(3) + 1)
    return np.exp(log_probability - logsumexp(log_probability))


def training_pairs(sample_count, seed, source_pixels=None):
    random = np.random.default_rng(seed)
    templates = point_source_templates(source_pixels)
    conditions, objects = [], []
    for _ in range(sample_count):
        count = int(random.integers(1, MAXIMUM_SOURCES + 1))
        unit = random.uniform(size=(count, 2))
        catalog = unit * np.array([1.0, MAXIMUM_FLUX])
        target = int(random.integers(count))
        others = np.delete(catalog, target, axis=0)
        observed = source_image(catalog, templates) + random.normal(0.0, PIXEL_NOISE, templates.shape[0])
        conditions.append(image_features(observed - source_image(others, templates), templates))
        objects.append(unit[target])
    return np.asarray(conditions), np.asarray(objects)


def validation_catalogs(scenes_per_count, seed):
    if isinstance(scenes_per_count, bool) or int(scenes_per_count) != scenes_per_count or scenes_per_count < 1:
        raise ValueError("scenes_per_count must be a positive integer.")
    if scenes_per_count == 1:
        return [
            (0, np.empty((0, 2))),
            (1, np.array([[0.25, 180.0]])),
            (2, np.array([[0.25, 180.0], [0.75, 120.0]])),
        ]

    random = np.random.default_rng(seed)
    cases = []
    for count in range(MAXIMUM_SOURCES + 1):
        for _ in range(scenes_per_count):
            catalog = random.uniform(size=(count, 2)) * np.array([1.0, MAXIMUM_FLUX])
            cases.append((count, catalog))
    return cases


def compare_image(observed, network, *, number_steps=12000, chains=2, seed=0, catalog_network=None, reverse_order=False, prior_catalog=False, source_pixels=None):
    templates = point_source_templates(source_pixels)
    observations = np.asarray(observed).ravel()
    bounds = [[0.0, 1.0], [0.0, MAXIMUM_FLUX]]
    selection = PointSourceSelection(network, templates.shape[1])

    def likelihood(base, catalog):
        residual = observations - base[0] - source_image(catalog, templates)
        return -0.5 * float(np.sum((residual / PIXEL_NOISE) ** 2))

    def features(base, catalog):
        return image_features(observations - base[0] - source_image(catalog, templates), templates)

    def draw(base, catalog, generator):
        return selection.sample(features(base, catalog), generator) * np.array([1.0, MAXIMUM_FLUX])

    def density(base, catalog, component):
        unit = component / np.array([1.0, MAXIMUM_FLUX])
        return selection.log_prob(features(base, catalog), unit) - np.log(MAXIMUM_FLUX)

    reference = reference_count_probabilities(observed, grid_size=101, source_pixels=source_pixels)
    records, arrays = [], {}
    methods = ("prior", "neural") + (("catalog",) if catalog_network is not None else ())
    if prior_catalog and catalog_network is None:
        raise ValueError("The matched whole-catalog comparison requires catalog_network.")
    if prior_catalog:
        methods += ("catalog_prior",)
    if reverse_order:
        methods = methods[::-1]
    catalog_functions = {"catalog": catalog_network.bind(observed)} if catalog_network is not None else {}
    if prior_catalog:
        catalog_functions["catalog_prior"] = catalog_network.bind(observed, learned=False)
    for method in methods:
        samples = []
        posterior_traces, elapsed, count_information, posterior_information = [], 0.0, 0.0, 0.0
        for chain in range(chains):
            started = time.perf_counter()
            result = sample_catalog(
                likelihood, [BACKGROUND_BOUNDS], bounds,
                maximum_components=MAXIMUM_SOURCES, count_prior_mean=1.0,
                number_steps=number_steps, burn_steps=number_steps // 4, thinning=2,
                seed=seed + chain, proposal_fraction=0.1,
                **({"draw_component": draw, "log_component_density": density} if method == "neural" else {}),
                     **({"draw_catalog": catalog_functions[method][0], "log_catalog_density": catalog_functions[method][1]}
                         if method in catalog_functions else {}),
            )
            count = result["component_counts"]
            seconds = time.perf_counter() - started
            count_ess = effective_samples(count)
            posterior_ess = effective_samples(result["log_posterior"])
            elapsed += seconds
            count_information += count_ess
            posterior_information += posterior_ess
            posterior_traces.append(result["log_posterior"])
            samples.append(count)
            arrays[f"{method}_chain{chain}_count"] = count
            arrays[f"{method}_chain{chain}_log_posterior"] = result["log_posterior"]
            arrays[f"{method}_chain{chain}_background"] = result["base_samples"]
            arrays[f"{method}_chain{chain}_catalog"] = np.asarray([
                np.pad(catalog, ((0, MAXIMUM_SOURCES - len(catalog)), (0, 0)), constant_values=np.nan)
                for catalog in result["catalog_samples"]
            ])
            records.append({
                "method": method, "chain": chain, "seed": seed + chain,
                "seconds": seconds, "count_ess": count_ess, "log_posterior_ess": posterior_ess,
                "count_probability": (np.bincount(count, minlength=3) / count.size).tolist(),
                "acceptance_counts": {name: int(value) for name, value in result["acceptance_counts"].items()},
            })
        counts = np.concatenate(samples)
        probability = np.bincount(counts, minlength=3) / counts.size
        records.append({
            "method": method, "combined_count_probability": probability.tolist(),
            "total_variation_from_reference": 0.5 * float(np.abs(probability - reference).sum()),
            "sampling_seconds": elapsed, "count_ess": count_information,
            "count_ess_per_second": count_information / elapsed,
            "log_posterior_ess_per_second": posterior_information / elapsed,
            "count_rhat": finite_rhat(samples),
            "log_posterior_rhat": finite_rhat(posterior_traces),
        })
    return {"reference_count_probability": reference.tolist(), "runs": records}, arrays


def _bootstrap_median_interval(values, seed, draws=5000):
    values = np.asarray(values, dtype=float)
    random = np.random.default_rng(seed)
    indices = random.integers(0, values.size, size=(draws, values.size))
    medians = np.median(values[indices], axis=1)
    return [float(value) for value in np.quantile(medians, [0.025, 0.975])]


def summarize_validation_cases(cases, seed):
    summary = {"validation_images": len(cases), "by_injected_source_count": {}}
    for count in range(MAXIMUM_SOURCES + 1):
        selected = [case for case in cases if case["injected_source_count"] == count]
        if not selected:
            continue
        distances = {
            method: np.asarray([
                next(run["total_variation_from_reference"] for run in case["runs"]
                     if run["method"] == method and "total_variation_from_reference" in run)
                for case in selected
            ])
            for method in ("prior", "neural")
        }
        differences = distances["neural"] - distances["prior"]
        summary["by_injected_source_count"][str(count)] = {
            "images": len(selected),
            "prior_median_total_variation": float(np.median(distances["prior"])),
            "prior_median_total_variation_95pct_bootstrap_interval": _bootstrap_median_interval(
                distances["prior"], seed + 3 * count
            ),
            "neural_median_total_variation": float(np.median(distances["neural"])),
            "neural_median_total_variation_95pct_bootstrap_interval": _bootstrap_median_interval(
                distances["neural"], seed + 3 * count + 1
            ),
            "neural_minus_prior_median_total_variation": float(np.median(differences)),
            "neural_minus_prior_median_total_variation_95pct_bootstrap_interval": _bootstrap_median_interval(
                differences, seed + 3 * count + 2
            ),
            "neural_lower_error_images": int(np.count_nonzero(differences < 0.0)),
        }
    return summary


def summarize_efficiency(cases, training_seconds, seed):
    comparisons = {}
    for method in ("neural", "catalog", "catalog_prior"):
        rows = []
        for case in cases:
            combined = {run["method"]: run for run in case["runs"] if "combined_count_probability" in run}
            if method not in combined:
                continue
            prior, selected = combined["prior"], combined[method]
            rows.append({
                "injected_sources": case["injected_source_count"],
                "count_ess_per_second_ratio": selected["count_ess_per_second"] / prior["count_ess_per_second"] if prior["count_ess_per_second"] > 0 else None,
                "log_posterior_ess_per_second_ratio": selected["log_posterior_ess_per_second"] / prior["log_posterior_ess_per_second"] if prior["log_posterior_ess_per_second"] > 0 else None,
                "count_ess_per_evaluation_ratio": selected["count_ess"] / prior["count_ess"] if prior["count_ess"] > 0 else None,
                "training_inclusive_count_ess_per_second_ratio": (selected["count_ess"] / (selected["sampling_seconds"] + training_seconds.get(method, 0.0) / len(cases)) / prior["count_ess_per_second"]) if prior["count_ess_per_second"] > 0 else None,
                "training_inclusive_log_posterior_ess_per_second_ratio": (
                    selected["log_posterior_ess_per_second"] * selected["sampling_seconds"]
                    / (selected["sampling_seconds"] + training_seconds.get(method, 0.0) / len(cases))
                    / prior["log_posterior_ess_per_second"]
                ) if prior["log_posterior_ess_per_second"] > 0 else None,
                "total_variation_difference_from_prior": selected["total_variation_from_reference"] - prior["total_variation_from_reference"],
                "total_variation_from_reference": selected["total_variation_from_reference"],
                "count_rhat": selected["count_rhat"],
                "log_posterior_rhat": selected["log_posterior_rhat"],
            })
        if not rows:
            continue
        summary = {"paired_images": rows}
        for index, key in enumerate(("count_ess_per_second_ratio", "log_posterior_ess_per_second_ratio",
                                     "count_ess_per_evaluation_ratio", "training_inclusive_count_ess_per_second_ratio",
                                     "training_inclusive_log_posterior_ess_per_second_ratio", "total_variation_difference_from_prior")):
            values = [row[key] for row in rows]
            available = all(value is not None and np.isfinite(value) for value in values)
            summary[f"median_{key}"] = float(np.median(values)) if available else None
            summary[f"median_{key}_95pct_image_bootstrap_interval"] = _bootstrap_median_interval(values, seed + index) if available else None
        summary["median_total_variation_from_reference"] = float(np.median([row["total_variation_from_reference"] for row in rows]))
        summary["maximum_total_variation_from_reference"] = max(row["total_variation_from_reference"] for row in rows)
        summary["maximum_count_rhat"] = max((row["count_rhat"] for row in rows if row["count_rhat"] is not None), default=None)
        summary["maximum_log_posterior_rhat"] = max((row["log_posterior_rhat"] for row in rows if row["log_posterior_rhat"] is not None), default=None)
        summary["all_count_statistics_defined"] = all(row["count_rhat"] is not None and row["count_ess_per_second_ratio"] is not None for row in rows)
        comparisons[method] = summary
    return comparisons


def run(arguments):
    import torch

    root = Path(__file__).resolve().parent
    result_path = root / "data" / f"{arguments.run_id}_point_sources.json"
    arrays_path = result_path.with_suffix(".npz")
    figure_path = root / "visuals" / f"{arguments.run_id}_point_sources.{arguments.typefileplot}"
    checkpoint_path = result_path.with_suffix(".pt")
    catalog_checkpoint_path = result_path.with_name(result_path.stem + "_catalog_sources.pt")
    complete_catalog = getattr(arguments, "complete_catalog", False)
    outputs = (result_path, arrays_path, figure_path, checkpoint_path) + ((catalog_checkpoint_path,) if complete_catalog else ())
    for output in outputs:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}.")
    source_files = (Path(__file__), Path(__file__).resolve().parents[2] / "pcat" / "transdimensional.py")
    source_hashes = {source.name: hashlib.sha256(source.read_bytes()).hexdigest() for source in source_files}
    position_grid = getattr(arguments, "position_grid", 0)
    if position_grid:
        offset = (IMAGE_SHAPE[0] - position_grid) // 2
        source_pixels = tuple((column, row) for row in range(offset, offset + position_grid)
                              for column in range(offset, offset + position_grid))
    else:
        source_pixels = SOURCE_PIXELS
    conditions, objects = training_pairs(arguments.training_samples, arguments.seed, source_pixels)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        started = time.perf_counter()
        network = NeuralElementProposal.fit(
            conditions, objects, number_components=8, hidden_sizes=(64, 64),
            epochs=arguments.epochs, batch_size=128, uniform_fraction=0.1, seed=arguments.seed + 1,
        )
        training_seconds = time.perf_counter() - started
        catalog_network, catalog_training_seconds = None, 0.0
        if complete_catalog:
            started = time.perf_counter()
            catalog_network = NeuralCatalogSelection.fit(arguments.catalog_training_samples, arguments.catalog_epochs, arguments.seed + 20, source_pixels)
            catalog_training_seconds = time.perf_counter() - started
        cases = validation_catalogs(
            arguments.validation_scenes_per_count, arguments.seed + 50
        )
        if getattr(arguments, "repeat_example_catalogs", False):
            cases = [(count, catalog.copy()) for count, catalog in validation_catalogs(1, arguments.seed + 50)
                     for _ in range(arguments.validation_scenes_per_count)]
        results, arrays = [], {}
        if catalog_network is not None:
            arrays["catalog_count_mean"] = catalog_network.count_mean
            arrays["catalog_count_scale"] = catalog_network.count_scale
            for index, (weight, bias) in enumerate(catalog_network.count_layers):
                arrays[f"catalog_count_layer{index}_weight"] = weight
                arrays[f"catalog_count_layer{index}_bias"] = bias
        for index, (injected_count, catalog) in enumerate(cases):
            observed, expected = simulate_image(arguments.seed + 100 + index, catalog, source_pixels=source_pixels)
            result, chain_arrays = compare_image(
                observed, network, number_steps=arguments.steps, chains=arguments.chains,
                seed=arguments.seed + 200 + 10 * index,
                catalog_network=catalog_network,
                reverse_order=bool(index % 2),
                source_pixels=source_pixels,
                prior_catalog=getattr(arguments, "prior_catalog", False),
            )
            result["injected_source_count"] = injected_count
            result["injected_catalog"] = np.asarray(catalog).reshape(-1, 2).tolist()
            results.append(result)
            arrays[f"case{index}_observed"] = observed
            arrays[f"case{index}_expected"] = expected
            arrays.update({f"case{index}_{name}": value for name, value in chain_arrays.items()})
    finally:
        torch.set_num_threads(previous_threads)
    for source in source_files:
        if hashlib.sha256(source.read_bytes()).hexdigest() != source_hashes[source.name]:
            raise RuntimeError("Benchmark source changed during execution; refusing to publish results.")
    result = {
        "run_id": arguments.run_id, "problem": "point-source detection in a finite image",
        "image_shape": list(IMAGE_SHAPE), "candidate_pixels": source_pixels,
        "position_search": "blind discrete pixel grid" if position_grid else "two supplied candidate positions",
        "psf_sigma_pixels": PSF_SIGMA, "flux_bounds_count": [0.0, MAXIMUM_FLUX],
        "background_bounds_counts_per_pixel": BACKGROUND_BOUNDS,
        "pixel_noise_counts": PIXEL_NOISE, "count_prior": "Poisson(1), truncated to zero through two",
        "training_samples": arguments.training_samples, "epochs": arguments.epochs,
        "steps": arguments.steps, "chains": arguments.chains, "seed": arguments.seed,
        "validation_scenes_per_count": arguments.validation_scenes_per_count,
        "validation_catalog_design": "repeated fixed catalogs with independent noise" if getattr(arguments, "repeat_example_catalogs", False) else "independent simulated catalogs",
        "method_order": "alternating forward and reverse order across images",
        "training_seconds": training_seconds, "source_sha256": source_hashes,
        "training_condition_dimension": conditions.shape[1],
        "complete_catalog": complete_catalog,
        "prior_catalog": getattr(arguments, "prior_catalog", False),
        "catalog_training_samples": arguments.catalog_training_samples if complete_catalog else None,
        "catalog_epochs": arguments.catalog_epochs if complete_catalog else None,
        "catalog_training_seconds": catalog_training_seconds,
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "torch": torch.__version__},
        "assumptions": ["blind discrete pixel grid" if position_grid else "two known candidate positions", "independent Gaussian pixel noise", "known PSF", "two objects may share a candidate position"],
        "summary": summarize_validation_cases(results, arguments.seed + 500),
        "efficiency": summarize_efficiency(results, {"neural": training_seconds, "catalog": catalog_training_seconds}, arguments.seed + 800),
        "efficiency_summary": summarize_efficiency(
            results, {"neural": training_seconds, "catalog": catalog_training_seconds}, arguments.seed + 600
        ),
        "cases": results,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    network.save(checkpoint_path)
    if catalog_network is not None:
        catalog_network.component_network.save(catalog_checkpoint_path)
    print(f"Writing to {arrays_path}...")
    np.savez_compressed(arrays_path, **arrays)
    print(f"Writing to {result_path}...")
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plot_results(result, arrays, figure_path, arguments.typefileplot)
    return result


def plot_results(result, arrays, output, typefileplot="png"):
    import matplotlib.pyplot as plt
    from tdpy.plotting import save_figure

    if len(result["cases"]) > MAXIMUM_SOURCES + 1:
        return plot_replicated_results(result, arrays, output, typefileplot)

    figure, axes = plt.subplots(2, 3, figsize=(5.4, 4.8), constrained_layout=True, sharex="row", sharey="row")
    selected = [next((index, case) for index, case in enumerate(result["cases"])
                     if case["injected_source_count"] == count) for count in range(3)]
    minimum = min(arrays[f"case{case_index}_observed"].min() for case_index, _ in selected)
    maximum = max(arrays[f"case{case_index}_observed"].max() for case_index, _ in selected)
    for index, (case_index, case) in enumerate(selected):
        image = axes[0, index].imshow(arrays[f"case{case_index}_observed"], origin="lower", cmap="gray", vmin=minimum, vmax=maximum)
        count = case["injected_source_count"]
        axes[0, index].set_title(f"{count} source" + ("" if count == 1 else "s"))
        axes[0, index].set_xlabel("Column [pixel]")
        if index == 0:
            axes[0, index].set_ylabel("Row [pixel]")
        reference = case["reference_count_probability"]
        axes[1, index].plot(range(3), reference, "ko", label="Numerical reference")
        for method, marker, color in (("prior", "s", "#0072B2"), ("neural", "^", "#D55E00"), ("catalog", "x", "#009E73")):
            if not any(run["method"] == method for run in case["runs"]):
                continue
            combined = next(run for run in case["runs"] if run["method"] == method and "combined_count_probability" in run)
            axes[1, index].plot(range(3), combined["combined_count_probability"], marker=marker, linestyle="none", color=color,
                                markerfacecolor="none", markersize=8, label=method.capitalize())
        axes[1, index].set_xticks(range(3))
        axes[1, index].set_xlim(-0.18, 2.18)
        axes[1, index].set_ylim(-0.06, 1.06)
        axes[1, index].set_xlabel("Source count")
        if index == 0:
            axes[1, index].set_ylabel("Probability")
    figure.colorbar(image, ax=axes[0, :], label="Pixel value [count]", fraction=0.035)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    names = {"Numerical reference": "Reference", "Prior": "Prior", "Neural": "Source network", "Catalog": "Catalog network"}
    figure.legend(handles, [names[label] for label in labels], loc="outside lower center", ncol=2, frameon=True)
    print(f"Writing to {output}...")
    save_figure(figure, output, typefileplot, dpi=300, font_size=12)
    plt.close(figure)


def plot_replicated_results(result, arrays, output, typefileplot="png"):
    import matplotlib.pyplot as plt
    from tdpy.plotting import save_figure

    selected = {
        count: [case for case in result["cases"] if case["injected_source_count"] == count]
        for count in range(MAXIMUM_SOURCES + 1)
    }
    representatives = [cases[0] for cases in selected.values()]
    figure, axes = plt.subplots(2, 3, figsize=(6.3, 5.6), constrained_layout=True)
    minimum = min(arrays[f"case{result['cases'].index(case)}_observed"].min()
                  for case in representatives)
    maximum = max(arrays[f"case{result['cases'].index(case)}_observed"].max()
                  for case in representatives)
    methods = (
        ("prior", "Standard PCAT", "#0072B2"),
        ("neural", "Single source", "#D55E00"),
        ("catalog", "Whole catalog", "#009E73"),
    )
    available_methods = [method for method in methods
                         if all(any(run["method"] == method[0] for run in case["runs"])
                                for case in representatives)]
    all_errors = []
    for column, count in enumerate(range(MAXIMUM_SOURCES + 1)):
        case = representatives[column]
        pixels = arrays[f"case{result['cases'].index(case)}_observed"]
        image = axes[0, column].imshow(pixels, origin="lower", cmap="gray", vmin=minimum, vmax=maximum)
        axes[0, column].set_title(f"{count} source" + ("" if count == 1 else "s"))
        axes[0, column].set_xlabel("Column [pixel]")
        if column == 0:
            axes[0, column].set_ylabel("Row [pixel]")
        for position, (method, label, color) in enumerate(available_methods):
            errors = np.asarray([
                next(run["total_variation_from_reference"] for run in scene["runs"]
                     if run["method"] == method and "total_variation_from_reference" in run)
                for scene in selected[count]
            ])
            all_errors.extend(errors)
            jitter = np.linspace(-0.1, 0.1, errors.size)
            axes[1, column].scatter(
                position + jitter, errors, s=24, color=color, alpha=0.75,
                label=label if column == 0 else None,
            )
            axes[1, column].plot(
                [position - 0.2, position + 0.2], [np.median(errors)] * 2,
                color="black", linewidth=1.5,
                label="Median" if column == 0 and position == 0 else None,
            )
        axes[1, column].set_xticks([])
        axes[1, column].set_xlim(-0.4, len(available_methods) - 0.6)
        if column == 0:
            axes[1, column].set_ylabel("Total variation")
    upper = max(float(np.max(all_errors)) * 1.15, 0.01)
    for axis in axes[1]:
        axis.set_ylim(0.0, upper)
    figure.colorbar(image, ax=axes[0, :], label="Pixel value [count]", fraction=0.035)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    handle_by_label = dict(zip(labels, handles))
    ordered_labels = [method[1] for method in available_methods] + ["Median"]
    figure.legend(
        [handle_by_label[label] for label in ordered_labels], ordered_labels,
        loc="outside lower center", ncol=len(ordered_labels), fontsize=12,
        frameon=True, fancybox=True, framealpha=1.0, edgecolor="black",
    )
    print(f"Writing to {output}...")
    save_figure(figure, output, typefileplot, dpi=300, font_size=12)
    plt.close(figure)


def plot_efficiency(result, output, typefileplot="png"):
    import matplotlib.pyplot as plt
    from tdpy.plotting import save_figure

    combined = [
        {run["method"]: run for run in case["runs"] if "combined_count_probability" in run}
        for case in result["cases"]
    ]
    figure, axes = plt.subplots(1, 2, figsize=(6.5, 3.2), constrained_layout=True)
    methods = (("prior", "Prior", "#0072B2"), ("neural", "Single\nsource", "#D55E00"),
               ("catalog", "Whole\ncatalog", "#009E73"))
    if all("catalog_prior" in rows for rows in combined):
        methods += (("catalog_prior", "Whole\nprior", "#666666"),)
    jitter = np.linspace(-0.1, 0.1, len(combined))
    for position, (method, label, color) in enumerate(methods):
        errors = [rows[method]["total_variation_from_reference"] for rows in combined]
        axes[1].scatter(position + jitter, errors, color=color, s=12, alpha=0.65)
        axes[1].plot([position - 0.23, position + 0.23], [np.median(errors)] * 2,
                     color="black", linewidth=1.5)
        if method != "prior":
            ratios = [rows[method]["log_posterior_ess_per_second"]
                      / rows["prior"]["log_posterior_ess_per_second"] for rows in combined]
            axes[0].scatter(position - 1 + jitter, ratios, color=color, s=12, alpha=0.65)
            axes[0].plot([position - 1.23, position - 0.77], [np.median(ratios)] * 2,
                         color="black", linewidth=1.5)
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Log-posterior ESS/s\nrelative to prior")
    axes[0].set_xticks(range(len(methods) - 1), [method[1] for method in methods[1:]])
    axes[0].set_xlim(-0.4, len(methods) - 1.6)
    axes[1].set_ylabel("Count-posterior TV")
    axes[1].set_xticks(range(len(methods)), [method[1] for method in methods])
    axes[1].set_xlim(-0.4, len(methods) - 0.6)
    axes[1].set_ylim(bottom=-0.001)
    for axis in axes:
        axis.grid(False)
    print(f"Writing to {output}...")
    save_figure(figure, output, typefileplot, dpi=300, font_size=12)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="point_sources_" + datetime.now(ZoneInfo("America/Chicago")).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--training-samples", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--complete-catalog", action="store_true")
    parser.add_argument("--prior-catalog", action="store_true")
    parser.add_argument("--catalog-training-samples", type=int, default=12000)
    parser.add_argument("--catalog-epochs", type=int, default=80)
    parser.add_argument("--repeat-example-catalogs", action="store_true")
    parser.add_argument("--position-grid", type=int, choices=(0, 3, 5), default=0)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--validation-scenes-per-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--typefileplot", choices=("png", "pdf"), default="png")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    if arguments.smoke:
        arguments.training_samples, arguments.epochs, arguments.steps = 48, 2, 80
    if (arguments.training_samples < 2 or arguments.epochs < 1 or arguments.steps < 20
            or arguments.chains < 1 or arguments.validation_scenes_per_count < 1):
        parser.error("Require at least two training examples, one epoch, twenty steps, one chain, and one validation image per count.")
    if arguments.prior_catalog and not arguments.complete_catalog:
        parser.error("--prior-catalog requires --complete-catalog.")
    run(arguments)


if __name__ == "__main__":
    main()