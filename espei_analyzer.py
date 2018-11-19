"""Module for automated analysis of ESPEI results."""

from collections import namedtuple
import click
import numpy as np
import matplotlib.pyplot as plt
import corner
from pycalphad import Database, variables as v

from espei.datasets import load_datasets, recursive_glob
from espei.analysis import truncate_arrays
from espei.utils import database_symbols_to_fit, optimal_parameters, formatted_parameter
from espei.plot import multiplot


def parameter_labels(dbf, formatted=True):
    parameter_symbols = database_symbols_to_fit(dbf)

    if formatted:
        parameter_labels = []
        for sym in parameter_symbols:
            fp = formatted_parameter(dbf, sym)
            label = "{}({})\n{}: {}".format(fp.phase_name, fp.interaction, fp.parameter_type, fp.term_symbol)
            parameter_labels.append(label)
        return parameter_labels
    else:
        return parameter_symbols


def optimal_parameters_dict(dbf, trace, lnprob):
    return dict(zip(parameter_labels(dbf, formatted=False), optimal_parameters(trace, lnprob, 0)))


def plot_lnprob(lnprob, fname="lnprob.png", y_min=None, y_max=None, ax=None):
    """Plot lnprob vs. iterations
    
    Parameters
    ----------
    lnprob : numpy.ndarray
        NumPy array of lnprob. Shape (chains, iterations).
    fname : str, optional
        Name of the file to save the figure to (the default is 'lnprob.png')
    y_min : float, optional
        Minimum of the y-axis (the default is None, which sets the min to min(lnprob)/10)
    y_max : float, optional
        Maximum of the y-axis (the default is None, which sets the max to max(lnprob)*10)
    ax : matplotlib.axes.Axes, optional
        Axes to plot to (the default is None, which creates new Axes)
    
    Returns
    -------
    matplotlib.figure.Figure
    """
    y_min = y_min or np.min(lnprob) / 10
    y_max = y_max or np.max(lnprob) * 10

    if not ax:
        fig = plt.figure()
        ax = fig.gca()
    else:
        fig = ax.figure
    ax.set_yscale("log")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("- lnprob")

    num_chains = lnprob.shape[0]
    for i in range(num_chains):
        ax.plot(-lnprob[i, :])

    fig.savefig(fname)
    return fig


def plot_parameter_changes(dbf, trace, lnprob, fname="parameters.png"):
    """Plot the value of each parameter vs iterations.
    
    Parameters
    ----------
    dbf : pycalphad.Database
        pycalphad Database
    trace : numpy.ndarray
        Array of the trace. Shape (chains, iterations, parameters)
    lnprob : numpy.ndarray
        Array of the log probability. Shape (chains, iterations)
    fname : str, optional
        Filename to save the figure to (the default is "parameters.png")
    
    Returns
    -------
    matplotlib.figure.Figure

    """
    param_labels = parameter_labels(dbf, formatted=True)

    num_chains = trace.shape[0]
    num_parameters = trace.shape[2]
    default_figsize = plt.rcParams["figure.figsize"]
    scaled_figsize = (default_figsize[0], default_figsize[1] * num_parameters)
    fig, axes = plt.subplots(num_parameters, sharex=True, figsize=scaled_figsize)

    for parameter, ax in zip(range(num_parameters), axes):
        ax.set_title(param_labels[parameter])
        ax.set_ylabel("Parameter Value")
        for chain in range(num_chains):
            ax.plot(trace[chain, :, parameter])

    ax.set_xlabel("Iterations")
    fig.savefig(fname)
    return fig


def plot_corner(dbf, trace, fname="corner.png"):
    param_labels = parameter_labels(dbf, formatted=True)
    fig = corner.corner(trace.reshape(-1, trace.shape[-1]), labels=param_labels)
    fig.savefig(fname)
    return fig


def plot_phase_diagram(dbf, trace, lnprob, datasets, temperatures=(300, 2500, 10), fname="phase_diagram.png"):
    # enable making an initial plot
    if (trace is not None) and (lnprob is not None):
        opt_parameters = optimal_parameters_dict(dbf, trace, lnprob)
    else:
        opt_parameters = dict()
    comps = [sp.name for sp in dbf.species]
    non_va_comps = sorted(set(comps) - {"VA"})
    phases = list(dbf.phases.keys())

    ax = multiplot(
        dbf,
        comps,
        phases,
        {v.P: 101325, v.T: temperatures, v.X(non_va_comps[-1]): (0, 1, 0.05)},
        datasets,
        eq_kwargs={"parameters": opt_parameters},
    )

    fig = ax.figure
    ax.set_xlim(0, 1)
    ax.set_ylim(*(temperatures[:2]))
    fig.savefig(fname)
    return fig


def save_phase_diagram_animation(dbf, trace, lnprob, datasets, num_chunks=30, temperatures=(300, 2500, 10)):
    # create all the remaining images images
    idx_start = 0
    for count, idx_end in enumerate([0] + sorted(set(np.logspace(0,np.log10(trace.shape[1]), num_chunks, dtype=np.int)))):
        # make the 0 iterations image
        if idx_end == 0:
            plot_phase_diagram(
                dbf, None, None, datasets, temperatures=temperatures, fname="animation-{:03d}.png".format(count)
            )
        else:
            plot_phase_diagram(
                dbf,
                trace[:, idx_start:idx_end, :],
                lnprob[:, idx_start:idx_end],
                datasets,
                temperatures=temperatures,
                fname="animation-{:03d}.png".format(count),
            )
        idx_start = idx_end
    print(
        "To animate, use an external program, e.g. ImageMagick: `convert -delay 20 -loop 0 animation-*.png animation.gif`"
    )


def main(dbf, trace, lnprob, datasets, plots="blps", phase_diagram_opts=None):

    if (trace is not None) and (lnprob is not None):
        trace, lnprob = truncate_arrays(trace, lnprob)

    if "l" in plots:
        plot_lnprob(lnprob)
    if "p" in plots:
        plot_parameter_changes(dbf, trace, lnprob)
    if "c" in plots:
        plot_corner(dbf, trace)
    # TODO: plotting command to generate parameter endmembers and interactions and plot_parameters
    if "b" in plots:
        plot_phase_diagram(dbf, trace, lnprob, datasets, **phase_diagram_opts)
    if "a" in plots:
        save_phase_diagram_animation(dbf, trace, lnprob, datasets, **phase_diagram_opts)


PLOTS_HELP_STRING = """
Explictly choose plots, default = 'bpls' (all)
    b: binary phase diagram
    p: parameter changes
    l: log probability
    c: corner plot
"""


@click.command()
@click.option("--database", "-db", help="Path to a thermodynamic database.", type=click.Path())
@click.option("--tracefile", "-t", help="Path to ESPEI trace.", type=click.Path(), default=None)
@click.option("--probfile", "-p", help="Path to ESPEI lnprob.", type=click.Path(), default=None)
@click.option("--datasets", "-ds", help="Path to ESPEI lnprob.", type=click.Path(), default=None)
@click.option("--t_min", help="Minimum phase diagram temperature.", type=click.FLOAT, default=300)
@click.option("--t_max", help="Maximum phase diagram temperature.", type=click.FLOAT, default=2500)
@click.option("--t_step", help="Phase diagram temperature step size.", type=click.FLOAT, default=10)
@click.option("--plot", help=PLOTS_HELP_STRING, type=click.STRING, default="bpls")
def run(database, tracefile, probfile, datasets, t_min, t_max, t_step, plot):
    dbf = Database(database)
    trace = np.load(tracefile) if tracefile else None
    lnprob = np.load(probfile) if probfile else None
    ds = load_datasets(recursive_glob(datasets, "*.json")) if datasets else None
    phase_diagram_options = dict()
    phase_diagram_options["temperatures"] = (t_min, t_max, t_step)
    plots = plot.lower()

    main(dbf, trace, lnprob, ds, plots, phase_diagram_options)


if __name__ == "__main__":
    run()
