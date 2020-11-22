"""Command line tools for the project."""

import click

from hoqunm.data_tools.cli import analyse_data, build_model, preprocess_data
from hoqunm.optimisation.cli import compute_optimum, simulate_optimum
from hoqunm.simulation.cli import analyse_model_variants, assess_capacities


@click.group()
def cli():
    """CLI interface for HoQuNM project."""


cli.add_command(preprocess_data)
cli.add_command(analyse_data)
cli.add_command(build_model)
cli.add_command(analyse_model_variants)
cli.add_command(assess_capacities)
cli.add_command(compute_optimum)
cli.add_command(simulate_optimum)

if __name__ == '__main__':
    cli()
