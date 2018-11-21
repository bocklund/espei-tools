"""Generate datasets from a reference TDB.

Currently only supports ZPF data.

Useful to generate small benchmark test sets.

"""

from pycalphad import Model, Database, calculate, equilibrium, variables as v
from pycalphad.plot.eqplot import _map_coord_to_variable, unpack_condition
from collections import OrderedDict
import numpy as np

class ModelNoIdealMix(Model):
    """Same as pycalphad's Model, but with no ideal mixing contribution."""
    contributions = [('ref', 'reference_energy'),
                     ('xsmix', 'excess_mixing_energy'), ('mag', 'magnetic_energy'),
                     ('2st', 'twostate_energy'), ('ein', 'einstein_energy'),
                     ('ord', 'atomic_ordering_energy')]




def extract_zpf_point_data(eq):
    conds = OrderedDict([(_map_coord_to_variable(key), unpack_condition(np.asarray(value)))
                         for key, value in sorted(eq.coords.items(), key=str)
                         if (key == 'T') or (key == 'P') or (key.startswith('X_'))])
    print(conds)
    indep_comps = sorted([key for key, value in conds.items() if isinstance(key, v.Composition)], key=str)
    indep_pots = [v.T]

    # determine what the type of plot will be
    if len(indep_comps) == 1 and len(indep_pots) == 1:
        projection = None
    elif len(indep_comps) == 2 and len(indep_pots) == 0:
        projection = 'triangular'
    else:
        raise ValueError('The eqplot projection is not defined and cannot be autodetected. There are {} independent compositions and {} indepedent potentials.'.format(len(indep_comps), len(indep_pots)))

    # Handle cases for different plot types
    x = None
    tielines = True
    y = None
    if projection is None:
        x = indep_comps[0] if x is None else x
        y = indep_pots[0] if y is None else y
    elif projection == 'triangular':
        x = indep_comps[0] if x is None else x
        y = indep_comps[1] if y is None else y

    # get the active phases and support loading netcdf files from disk
    phases = map(str, sorted(set(np.array(eq.Phase.values.ravel(), dtype='U')) - {''}, key=str))
    comps = map(str, sorted(np.array(eq.coords['component'].values, dtype='U'), key=str))
    eq['component'] = np.array(eq['component'], dtype='U')
    eq['Phase'].values = np.array(eq['Phase'].values, dtype='U')

    # Select all two- and three-phase regions
    three_phase_idx = np.nonzero(np.sum(eq.Phase.values != '', axis=-1, dtype=np.int) == 3)
    two_phase_idx = np.nonzero(np.sum(eq.Phase.values != '', axis=-1, dtype=np.int) == 2)

    # For both two and three phase, cast the tuple of indices to an array and flatten
    # If we found two phase regions:
    if two_phase_idx[0].size > 0:
        found_two_phase = eq.Phase.values[two_phase_idx][..., :2]
        # get tieline endpoint compositions
        two_phase_x = eq.X.sel(component=x.species.name).values[two_phase_idx][..., :2]
        # handle special case for potential
        if isinstance(y, v.Composition):
            two_phase_y = eq.X.sel(component=y.species.name).values[two_phase_idx][..., :2]
        else:
            # it's a StateVariable. This must be True
            two_phase_y = np.take(eq[str(y)].values, two_phase_idx[list(str(i) for i in conds.keys()).index(str(y))])
            # because the above gave us a shape of (n,) instead of (n,2) we are going to create it ourselves
            two_phase_y = np.array([two_phase_y, two_phase_y]).swapaxes(0, 1)

        if tielines:
            # construct and plot tielines
            two_phase_tielines = np.array([np.concatenate((two_phase_x[..., 0][..., np.newaxis], two_phase_y[..., 0][..., np.newaxis]), axis=-1),
                                           np.concatenate((two_phase_x[..., 1][..., np.newaxis], two_phase_y[..., 1][..., np.newaxis]), axis=-1)])
            two_phase_tielines = np.rollaxis(two_phase_tielines, 1)
            return (x,y), two_phase_tielines, found_two_phase


    # If we found three phase regions:
    if three_phase_idx[0].size > 0:
        found_three_phase = eq.Phase.values[three_phase_idx][..., :3]
        # get tieline endpoints
        three_phase_x = eq.X.sel(component=x.species.name).values[three_phase_idx][..., :3]
        three_phase_y = eq.X.sel(component=y.species.name).values[three_phase_idx][..., :3]
        # three phase tielines, these are tie triangles and we always plot them
        three_phase_tielines = np.array([np.concatenate((three_phase_x[..., 0][..., np.newaxis], three_phase_y[..., 0][..., np.newaxis]), axis=-1),
                                     np.concatenate((three_phase_x[..., 1][..., np.newaxis], three_phase_y[..., 1][..., np.newaxis]), axis=-1),
                                     np.concatenate((three_phase_x[..., 2][..., np.newaxis], three_phase_y[..., 2][..., np.newaxis]), axis=-1)])
        three_phase_tielines = np.rollaxis(three_phase_tielines,1)
        three_lc = mc.LineCollection(three_phase_tielines, zorder=1, colors=[1,0,0,1], linewidths=[0.5, 0.5])
        # plot three phase points and tielines
        three_phase_plotcolors = np.array(list(map(lambda x: [colorlist[x[0]], colorlist[x[1]], colorlist[x[2]]], found_three_phase)), dtype='U') # from pycalphad

def make_zpf_dataset(equilibrium_dataset, half_mode=False):
    """Turn an equilibrium dataset into an ESPEI dataset dict

    If half_mode: create two data points for each equilibrium (null)

    Assumes binary 2 phase equilibria in T-X space. Assumes P=101325 Pa.
    """
    comps = sorted(eq_res.component.values.tolist())
    phases = set(equilibrium_dataset.Phase.values.flatten().squeeze().tolist()) - {''}

    dataset = {
        "components": sorted(comps),
        "phases": sorted(phases),
        "broadcast_conditions": False,
        "conditions": {
            "T": [],
            "P": [101325.0],  # assumed
        },
        "output": "ZPF",
        "values": [],
        "reference": "",
        "comment": "",
    }
    # points is an array of shape (datapoints, tieline endpoints, (X, T)), (n, 2, 2)
    # phases is a (datapoints, phases) (n, 2)  array
    (comp, pot), points, phases = extract_zpf_point_data(equilibrium_dataset)
    component = comp.species.name

    if not half_mode:
        temps = []
        values = []
        for pt, ph in zip(points, phases):
            val = [[ph[i], [component], [pt[i, 0]]] for i in range(pt.shape[0])]
            temps.append(pt[0, 1])
            values.append(val)
    else:
        temps = []
        values = []
        for pt, ph in zip(points, phases):
            # point 1
            val = [[ph[0], [component], [pt[0, 0]]], [ph[1], [component], [None]]]
            temps.append(pt[0, 1])
            values.append(val)

            # point 2
            val = [[ph[1], [component], [pt[1, 0]]], [ph[0], [component], [None]]]
            temps.append(pt[1, 1])
            values.append(val)

    # temperatures
    dataset["conditions"]["T"] = temps
    # values
    # should be of style [[phase_0, [elements], [compositions]], [phase_1, [elements], [compositions]], ...]
    dataset["values"] = values

    return dataset



if __name__ == '__main__':
    dbf = Database('Cu-Ni.tdb')
    comps = ["CU", 'NI', 'VA']
    phases = list(dbf.phases.keys())
    eq_res = equilibrium(dbf, comps, phases, {v.P: 101325, v.T: np.linspace(1085+273, 1455+273, 20), v.X('CU'): (0, 1, 0.1)})

    zpf_dataset = make_zpf_dataset(eq_res)

    import json
    with open('tieline-zpf.json', 'w') as fp:
        json.dump(zpf_dataset, fp)

    zpf_dataset = make_zpf_dataset(eq_res, True)
    with open('half-tieline-zpf.json', 'w') as fp:
        json.dump(zpf_dataset, fp)

