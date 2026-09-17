"""Catalog association utilities."""

import numpy as np
from scipy.spatial import cKDTree


def associate_catalogs(
	coordinates_source,
	values_source,
	coordinates_target,
	values_target,
	distance_maximum,
	value_difference_maximum,
	confidence_target=None,
	significance_target=None,
):
	"""Associate sources with all nearby targets having compatible values.

	Returns a Boolean array indicating whether each source has at least one
	match. Optional target confidence and significance arrays add the maximum
	confidence and minimum significance among each source's valid matches.
	"""
	coordinates_source = np.asarray(coordinates_source, dtype=float)
	coordinates_target = np.asarray(coordinates_target, dtype=float)
	values_source = np.asarray(values_source, dtype=float)
	values_target = np.asarray(values_target, dtype=float)
	if coordinates_source.size == 0 and coordinates_source.ndim == 1:
		numbdimension = coordinates_target.shape[1] if coordinates_target.ndim == 2 else 0
		coordinates_source = coordinates_source.reshape(0, numbdimension)
	if coordinates_target.size == 0 and coordinates_target.ndim == 1:
		numbdimension = coordinates_source.shape[1] if coordinates_source.ndim == 2 else 0
		coordinates_target = coordinates_target.reshape(0, numbdimension)
	if coordinates_source.ndim != 2 or coordinates_target.ndim != 2:
		raise ValueError('Catalog coordinates must be two-dimensional arrays.')
	if coordinates_source.shape[1] != coordinates_target.shape[1]:
		raise ValueError('Catalog coordinates must have the same dimension.')
	if coordinates_source.shape[0] != values_source.size:
		raise ValueError('Source coordinates and values must have equal lengths.')
	if coordinates_target.shape[0] != values_target.size:
		raise ValueError('Target coordinates and values must have equal lengths.')

	confidence_target = _validate_optional_target(
		confidence_target, values_target.size, 'confidence_target',
	)
	significance_target = _validate_optional_target(
		significance_target, values_target.size, 'significance_target',
	)
	matched = np.zeros(values_source.size, dtype=bool)
	confidence_matched = np.zeros(values_source.size) if confidence_target is not None else None
	significance_matched = (
		np.full(values_source.size, np.inf) if significance_target is not None else None
	)

	if values_source.size and values_target.size:
		nearby = cKDTree(coordinates_source).query_ball_tree(
			cKDTree(coordinates_target), distance_maximum,
		)
		for index_source, indices_target in enumerate(nearby):
			indices_target = np.asarray(indices_target, dtype=int)
			if not indices_target.size:
				continue
			indices_target = indices_target[
				np.abs(values_source[index_source] - values_target[indices_target])
				< value_difference_maximum
			]
			if not indices_target.size:
				continue
			matched[index_source] = True
			if confidence_matched is not None:
				confidence_matched[index_source] = np.max(confidence_target[indices_target])
			if significance_matched is not None:
				significance_matched[index_source] = np.min(significance_target[indices_target])

	outputs = [matched]
	if confidence_matched is not None:
		outputs.append(confidence_matched)
	if significance_matched is not None:
		outputs.append(significance_matched)

	return outputs[0] if len(outputs) == 1 else tuple(outputs)


def _validate_optional_target(values, size_target, name):
	if values is None:
		return None
	values = np.asarray(values, dtype=float)
	if values.ndim != 1 or values.size != size_target:
		raise ValueError('%s must match the target catalog length.' % name)

	return values