import six
import click
from pyfiglet import figlet_format

from pyadcirc.viz.figuregen import config

try:
    from termcolor import colored
except ImportError:
    colored = None


@click.group()
def cli():
    if colored:
        six.print_(colored(figlet_format("FIGUREGEN", font="slant"), "blue"))
    else:
        six.print_(figlet_format("FIGUREGEN", font="slant"))
    pass


@click.command()
@click.argument("input_file", type=click.File("r"))
def run(type: str, input_file: str):
    pass


cli.add_command(config)
