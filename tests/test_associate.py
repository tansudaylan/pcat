import numpy as np
import pytest

from pcat.associate import associate_catalogs


def test_associate_catalogs_filters_distance_and_value():
	coordinates_source = [[0., 0.], [2., 0.], [5., 0.]]
	coordinates_target = [[0.1, 0.], [0.2, 0.], [2.1, 0.], [5.1, 0.]]
	values_source = [20., 21., 22.]
	values_target = [20.1, 20.8, 22., 22.1]

	matched = associate_catalogs(
		coordinates_source, values_source, coordinates_target, values_target, 0.3, 0.5,
	)

	assert np.array_equal(matched, [True, False, True])


def test_associate_catalogs_summarizes_valid_targets():
	result = associate_catalogs(
		[[0., 0.]], [20.], [[0.1, 0.], [0.2, 0.], [0.25, 0.]],
		[20.1, 20.2, 21.], 0.3, 0.5,
		confidence_target=[0.4, 0.9, 1.], significance_target=[3., 2., 1.],
	)

	matched, confidence, significance = result
	assert matched[0]
	assert confidence[0] == pytest.approx(0.9)
	assert significance[0] == pytest.approx(2.)


def test_associate_catalogs_handles_empty_catalogs():
	matched = associate_catalogs([], [], [[0., 0.]], [20.], 0.3, 0.5)

	assert matched.size == 0


def test_associate_catalogs_rejects_mismatched_lengths():
	with pytest.raises(ValueError, match='Source coordinates'):
		associate_catalogs([[0., 0.]], [], [[0., 0.]], [20.], 0.3, 0.5)