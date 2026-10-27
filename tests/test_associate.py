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


def test_neighbors_within_radius_returns_sorted_indices_and_empty_rows():
	from pcat.associate import neighbors_within_radius

	coordinates_source = [[0., 0.], [2., 0.]]
	coordinates_target = [[0.2, 0.], [-0.2, 0.], [2., 0.]]
	neighbors = neighbors_within_radius(coordinates_source, coordinates_target, 0.3)

	assert [indices.tolist() for indices in neighbors] == [[0, 1], [2]]
	assert neighbors_within_radius([], coordinates_target, 0.3) == []
	empty_rows = neighbors_within_radius(coordinates_source, [], 0.3)
	assert all(indices.size == 0 for indices in empty_rows)


def test_associate_catalogs_rejects_mismatched_lengths():
	with pytest.raises(ValueError, match='Source coordinates'):
		associate_catalogs([[0., 0.]], [], [[0., 0.]], [20.], 0.3, 0.5)


def test_one_dimensional_condensation_uses_explicit_coordinate_and_tolerance():
	from types import SimpleNamespace
	from pcat.main import retr_condcatl

	catalogs = [[dict(center=np.array(centers), flux=np.array(fluxes))]
				for centers, fluxes in [([0., 2.], [10., 100.]),
									([0.02, 2.02], [12., 102.]), ([0.01], [14.])]]
	snapshots = [[{name: values.copy() for name, values in population.items()}
				  for population in draw] for draw in catalogs]
	posterior = SimpleNamespace(
		fitt=SimpleNamespace(indxpopl=np.arange(1), numbparagenrelem=2,
						 namepara=SimpleNamespace(genrelem=[['flux', 'center']])),
		indxsamptotl=np.arange(3), numbsamptotl=3, typeverb=0,
		listpostindxelemfull=[[np.arange(len(draw[0]['center']))] for draw in catalogs],
		listpostdictelem=catalogs, dictglob={}, blimpara=SimpleNamespace(),
	)
	with pytest.raises(ValueError, match='explicit.*tolerance'):
		retr_condcatl(posterior, coordinate_names=('center',))
	assert posterior.dictglob == {}
	retr_condcatl(posterior, coordinate_names=('center',), distance_maximum=0.1)
	assert posterior.numbstkscond == 2
	summaries = posterior.dictglob['poststkscond']
	order = np.argsort([entry['center'][0] for entry in summaries])
	np.testing.assert_allclose([summaries[index]['center'][0] for index in order], [0.01, 2.01])
	np.testing.assert_allclose([summaries[index]['flux'][0] for index in order], [12., 101.])
	np.testing.assert_allclose(posterior.prvl[order], [1., 2. / 3.])
	for draw, snapshot in zip(catalogs, snapshots):
		for population, saved in zip(draw, snapshot):
			for name, values in population.items():
				np.testing.assert_array_equal(values, saved[name])


@pytest.mark.parametrize('coordinate_names', [(), 'xpos', ('xpos', 'xpos'), ('',), ('a', 'b', 'c')])
def test_condensation_rejects_invalid_coordinate_names(coordinate_names):
	from types import SimpleNamespace
	from pcat.main import retr_condcatl

	posterior = SimpleNamespace(dictglob={})
	with pytest.raises(ValueError, match='coordinate_names'):
		retr_condcatl(posterior, coordinate_names=coordinate_names, distance_maximum=0.1)
	assert posterior.dictglob == {}


@pytest.mark.parametrize('distance', [0., -1., np.nan, np.inf, -np.inf, True, [], '0.1'])
def test_condensation_rejects_invalid_explicit_tolerance(distance):
	from types import SimpleNamespace
	from pcat.main import retr_condcatl

	posterior = SimpleNamespace(dictglob={})
	with pytest.raises(ValueError, match='finite and positive'):
		retr_condcatl(posterior, coordinate_names=('center',), distance_maximum=distance)
	assert posterior.dictglob == {}


@pytest.mark.parametrize('one_dimensional', [False, True])
@pytest.mark.parametrize('crossing', [False, True])
@pytest.mark.parametrize('reverse_populations', [False, True])
@pytest.mark.parametrize('reordered_features', [False, True])
def test_condensation_keeps_distinct_populations_separate(crossing, reverse_populations, reordered_features, one_dimensional):
	from types import SimpleNamespace
	from pcat.main import retr_condcatl

	positions = [[0., 0.08], [0.09, 0.01]] if crossing else [[0., 2.], [0.01, 2.01]]
	catalogs = [
		[dict(xpos=np.array([position]), ypos=np.zeros(1), flux=np.array([flux]))
		 for position, flux in zip(draw, (10. + 2 * draw_index, 100. + 2 * draw_index))]
		for draw_index, draw in enumerate(positions)
	]
	if reverse_populations:
		catalogs = [list(reversed(draw)) for draw in catalogs]
	snapshots = [[{name: values.copy() for name, values in population.items()}
				  for population in draw] for draw in catalogs]
	features = [['xpos', 'ypos', 'flux'],
				['flux', 'ypos', 'xpos'] if reordered_features else ['xpos', 'ypos', 'flux']]
	posterior = SimpleNamespace(
		fitt=SimpleNamespace(indxpopl=np.arange(2), numbparagenrelem=3,
						 namepara=SimpleNamespace(genrelem=features)),
		indxsamptotl=np.arange(2), numbsamptotl=2, typeverb=0, anglassc=0.2,
		listpostindxelemfull=[[np.array([0]), np.array([0])] for draw in catalogs],
		listpostdictelem=catalogs, dictglob={}, blimpara=SimpleNamespace(),
	)
	if one_dimensional:
		retr_condcatl(posterior, coordinate_names=('xpos',), distance_maximum=0.2)
	else:
		retr_condcatl(posterior)
	assert posterior.numbstkscond == 2
	medians = sorted(float(entry['flux'][0]) for entry in posterior.dictglob['poststkscond'])
	np.testing.assert_allclose(medians, [11., 101.])
	np.testing.assert_array_equal(posterior.prvl, [1., 1.])
	for draw, snapshot in zip(catalogs, snapshots):
		for population, saved in zip(draw, snapshot):
			for name, values in population.items():
				np.testing.assert_array_equal(values, saved[name])


def test_condensation_keeps_exact_distance_boundary_separate():
	from types import SimpleNamespace
	from pcat.main import retr_condcatl

	catalogs = [[dict(center=np.array([center]), flux=np.array([flux]))]
				for center, flux in [(0., 10.), (0.1, 20.)]]
	posterior = SimpleNamespace(
		fitt=SimpleNamespace(indxpopl=np.arange(1), numbparagenrelem=2,
							 namepara=SimpleNamespace(genrelem=[['center', 'flux']])),
		indxsamptotl=np.arange(2), numbsamptotl=2, typeverb=0,
		listpostindxelemfull=[[np.array([0])] for _ in catalogs],
		listpostdictelem=catalogs, dictglob={}, blimpara=SimpleNamespace(),
	)

	retr_condcatl(posterior, coordinate_names=('center',), distance_maximum=0.1)

	assert posterior.numbstkscond == 2
	assert sorted(posterior.prvl) == pytest.approx([0.5, 0.5])