"""
Pull ADCIRC Paramater Definitions

This python script can be used to create a json file with ADCIRC parameter
defintions that are web-scraped from web-docs.

"""

import json
import urllib.request

import bs4 as bs
import click

__author__ = "Carlos del-Castillo-Negrete"
__copyright__ = "Carlos del-Castillo-Negrete"
__license__ = "MIT"


ADCIRC_CONFIGS = {}
URL = "".join(
    ["https://adcirc.org/home/documentation/", "users-manual-v53/parameter-definitions"]
)


def pull_param_configs(url: str = URL):
    params = {}
    source = urllib.request.urlopen(url).read()
    rows = bs.BeautifulSoup(source, "lxml").findAll("p", {"class": "MsoNormal"})
    for row in rows:
        p_name = row.text.split()[0]
        if "(" in p_name:
            p_name = p_name.split("(")[0]
        params[p_name] = " ".join(row.text.split()[2:])

    return params


@click.command()
@click.argument("url", type=str, default=URL)
@click.argument("output_file", type=click.File("w"), default="adcirc_configs.json")
def scrape(url: str, output_file: str = "adcirc_defs.json"):
    """
    Pull ADCIRC parameter configs by web scraping URL.

    Parameters
    ----------
    url : str
        URL to web scrape configurations from.
    output_file : str
        Path to write json configuration file to.

    Returns
    ----------
    param_defs : dict
        ADCIRC parameter definitions dictionary.
    """

    # Web scrape parameter configs
    param_defs = pull_param_configs(url)

    # Write output file
    json.dump(param_defs, output_file)

    return param_defs


if __name__ == "__main__":
    """Create param defs json file."""
    configs = scrape()

    print(configs)
