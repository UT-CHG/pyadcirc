import pdb
import re
import pandas as pd
from pathlib import Path
from pyfiglet import Figlet
from termcolor import colored
from datetime import timedelta
from prettytable import PrettyTable


def make_pretty_table(df, fields, search=None,
                      match=r".", filter_fun=None, colors=None):
    """
    Makes a pretty table

    Prints dictionary keys in list `fields` for each dictionary in res,
    filtering on the search column if specified with regular expression
    if desired.

    Parameters
    ----------
    res : List[dict]
        List of dictionaries containing response of an AgavePy call
    fields : List[string]
        List of strings containing names of fields to extract for each element.
    search : string, optional
        String containing column to perform string patter matching on to
        filter results.
    match : str, default='.'
        Regular expression to match strings in search column.
    output_file : str, optional
        Path to file to output result table to.

    """
    # Initialize Table
    x = PrettyTable(float_format="0.2")
    x.field_names = fields
    if colors is None:
        colors = len(fields)*["blue"]
    else:
        if len(colors) != len(fields):
            raise ValueError('Colors must be same length as fields')

    # Build table from results
    for _, r in df.dropna().iterrows():
        if filter_fun is not None:
            r = filter_fun(r)
        if search is not None:
            if re.search(match, r[search]) is not None:
                x.add_row([colored(r[f], colors[i]) for i, f in enumerate(fields)])
        else:
            x.add_row([colored(r[f], colors[i]) for i, f in enumerate(fields)])

    return str(x)


def split_string(string, line_length):
    # Initialize an empty list to store the chunks
    chunks = []
    # Use a while loop to iterate through the string
    while len(string) > line_length:
        # Get the index of the last space before the line length
        space_index = string.rfind(" ", 0, line_length + 1)
        # If there is no space before the line length, use the line length as the index
        if space_index == -1:
            space_index = line_length
        # Append the chunk to the list and remove it from the string
        chunks.append(string[:space_index])
        string = string[space_index:].lstrip()
    # Append any remaining string to the list
    chunks.append(string)
    return chunks


def chunkify_str(text, line_length=70):
    """
    Turn a string into a list of certain length each item
    """
    text = str(text)
    text_len = len(text)
    chunk_size = text_len // line_length
    pdb.set_trace()
    text_list = []
    for i in range(0, text_len, line_length):
        text_list.append(text[i : i + chunk_size])
    return text_list


def get_help_text(
    category,
    option="all",
    key_color="yellow",
    value_color="blue",
    line_length=None
):
    """
    Parse description for CLI help
    """
    help_text = ""
    if option == "all":
        items = category.items()
    else:
        items = [(option, category[option])]
    for key, val in items:
        key_text = colored(key, key_color) if key_color is not None else key
        val_text = colored(val["desc"], value_color) if value_color is not None else val["desc"]
        help_text += f"{key_text} : {val_text} \n\n"

    if line_length is not None:
        help_text = split_string(help_text, line_length)

    return help_text


def get_banner_text(
    title="NOAA CO-OPS",
    font="slant",
    color="blue",
    attrs=["bold"],
    line_length=None,
):
    """ """
    f = Figlet(font=font)
    banner_text = f.renderText(title)
    colored_banner = colored(banner_text, color, attrs=attrs)

    if line_length is not None:
        colored_banner = split_string(colored_banner, line_length)

    return colored_banner

def create_pretty_table(data: pd.DataFrame, key_col_name:str, value_col_name:str, key_col_label:str, value_col_label:str, key_col_color:str, value_col_color:str):
    x = PrettyTable()
    x.field_names = [key_col_label, value_col_label]
    for _, row in data.iterrows():
        x.add_row([colored(row[key_col_name], key_col_color), colored(row[value_col_name], value_col_color)])
    return x.get_string()



banner_text = [
    "\x1b[1m\x1b[34m    _   ______  ___    ___       __________        ____  ____",
    "_____\n   / | / / __ \\/   |  /   |     / ____/ __ \\      / __ \\/ __ \\/",
    "___/\n  /  |/ / / / / /| | / /| |    / /   / / / /_____/ / / / /_/ /\\__",
    "\\ \n / /|  / /_/ / ___ |/ ___ |   / /___/ /_/ /_____/ /_/ / ____/___/ /",
    "/_/ |_/\\____/_/  |_/_/  |_|   \\____/\\____/      \\____/_/    /____/  \n ",
    "\x1b[0m",
]

product_help_text = [
    "\t\x1b[1m\x1b[33mmetadata\x1b[0m: \x1b[1m\x1b[34mStation metadata, like location and",
    "station name.\x1b[0m\n\t\x1b[1m\x1b[33mwater_level\x1b[0m: \x1b[1m\x1b[34mPreliminary or",
    "verified 6-minute interval water levels, depending on data",
    "availability.\x1b[0m\n\t\x1b[1m\x1b[33mhourly_height\x1b[0m: \x1b[1m\x1b[34mVerified",
    "hourly height water level data for the",
    "station.\x1b[0m\n\t\x1b[1m\x1b[33mhigh_low\x1b[0m: \x1b[1m\x1b[34mVerified high tide / low",
    "tide water level data for the station.\x1b[0m\n\t\x1b[1m\x1b[33mdaily_mean\x1b[0m:",
    "\x1b[1m\x1b[34mVerified daily mean water level data for the",
    "station.\nNote!Great Lakes stations only. Only available with",
    "time_zone=LST\x1b[0m\n\t\x1b[1m\x1b[33mmonthly_mean\x1b[0m: \x1b[1m\x1b[34mVerified",
    "monthly mean water level data for the",
    "station.\x1b[0m\n\t\x1b[1m\x1b[33mone_minute_water_level\x1b[0m:",
    "\x1b[1m\x1b[34mPreliminary 1-minute interval water level data for the",
    "station.\x1b[0m\n\t\x1b[1m\x1b[33mpredictions\x1b[0m: \x1b[1m\x1b[34mWater level / tide",
    "prediction data for the station.\nNote!See Interval for available data",
    "interval options and data length",
    "limitations.\x1b[0m\n\t\x1b[1m\x1b[33mdatums\x1b[0m: \x1b[1m\x1b[34mObserved tidal datum",
    "values at the station for the present National Tidal Datum Epoch",
    "(NTDE).\x1b[0m\n\t\x1b[1m\x1b[33mair_gap\x1b[0m: \x1b[1m\x1b[34mAir Gap (distance between",
    "a bridge and the water's surface) at the",
    "station.\x1b[0m\n\t\x1b[1m\x1b[33mair_temperature\x1b[0m: \x1b[1m\x1b[34mAir temperature",
    "as measured at the station.\x1b[0m\n\t\x1b[1m\x1b[33mwater_temperature\x1b[0m:",
    "\x1b[1m\x1b[34mWater temperature as measured at the",
    "station.\x1b[0m\n\t\x1b[1m\x1b[33mwind\x1b[0m: \x1b[1m\x1b[34mWind speed, direction, and",
    "gusts as measured at the station.\x1b[0m\n\t\x1b[1m\x1b[33mair_pressure\x1b[0m:",
    "\x1b[1m\x1b[34mBarometric pressure as measured at the",
    "station.\x1b[0m\n\t\x1b[1m\x1b[33mconductivity\x1b[0m: \x1b[1m\x1b[34mThe water's",
    "conductivity as measured at the station.\x1b[0m\n\t\x1b[1m\x1b[33mvisibility\x1b[0m:",
    "\x1b[1m\x1b[34mVisibility (atmospheric clarity) as measured at the station.",
    "(Units of Nautical Miles or Kilometers)\x1b[0m\n\t\x1b[1m\x1b[33mhumidity\x1b[0m:",
    "\x1b[1m\x1b[34mRelative humidity as measured at the",
    "station.\x1b[0m\n\t\x1b[1m\x1b[33msalinity\x1b[0m: \x1b[1m\x1b[34mSalinity and specific",
    "gravity data for the station.\x1b[0m\n\t\x1b[1m\x1b[33mcurrents\x1b[0m:",
    "\x1b[1m\x1b[34mCurrents data for the station. NOTE:  Default data interval is",
    "6-minute interval data.Use with “interval=h” for hourly",
    "data\x1b[0m\n\t\x1b[1m\x1b[33mcurrents_predictions\x1b[0m: \x1b[1m\x1b[34mCurrents",
    "prediction data for the stations. NOTE:  See Interval for options",
    "available and data length",
    "limitations.\x1b[0m\n\t\x1b[1m\x1b[33mofs_water_level\x1b[0m: \x1b[1m\x1b[34mCurrents data",
    "for the station. NOTE:  Default data interval is 6-minute interval",
    "data.Use with “interval=h” for hourly data\x1b[0m\n",
]


DATE_TIME = {
    "begin_date": ''.join([colored("Use with either: ", color='blue'),
                           colored("end_date", color="yellow"), ' : ',
                           colored("to express an explicit date range, or with ",
                                   color='blue'),
                           colored("date_range", color="yellow"), ' : ',
                           colored(''.join(["to express a date range (in hours) of ti",
                                            "me starting from a certain date."]),
                                   color="blue")]),
    "end_date": ''.join([colored("Use with either: ", color='blue'),
                         colored("begin_date", color="yellow"), ' : ',
                         colored("to express an explicit date range, or with ",
                                 color='blue'),
                         colored("date_range", color="yellow"), ' : ',
                         colored(''.join(["to express a date range (in hours) of ti",
                                          "me ending on a certain date."]),
                                 color="blue")]),
    "range": ''.join([colored("Use either with: ", color='blue'),
                      colored("begin_date", color="yellow"), ' : ',
                      colored("to express an explicit date range, or with ",
                              color='blue'),
                      colored("date_range", color="yellow"), ' : ',
                      colored(''.join(["to express a date range (in hours) of ti",
                                       "me ending on a certain date."]),
                              color="blue"),
                      '                          ',
                      colored("alone", color="yellow"), ' : ',
                      colored(''.join(["to specify number of hours bac",
                                       "k from current time (only available for pre",
                                       "liminary water level data and meteorologica",
                                       "l data)"]),
                              color="blue")]),
    "date": ''.join([colored(''.join(["Only available for preliminary water level data",
                                      ", meteorological data and predictions.\nValid o",
                                      "ptions for the date parameter are:"]),
                             color='blue'),
                     '             ',
                     get_help_text({"today": {"desc": "24 hours starting at midnight"},
                                    "latest": {"desc": ''.join(["last data point ",
                                                                "available within the ",
                                                                "last 18 min"])},
                                    "recent": {"desc": "last 72 hours"}})])
}

PRODUCTS = {
    "metadata": {
        "group": "metadata",
        "desc": "Station metadata, like location and station name.",
    },
    "water_level": {
        "group": "tide",
        "desc": "Preliminary or verified 6-minute interval water levels, depending on data availability.",
        "max_interval": timedelta(days=29),
    },
    "hourly_height": {
        "group": "tide",
        "desc": "Verified hourly height water level data for the station.",
        "max_interval": timedelta(days=364),
    },
    "high_low": {
        "group": "tide",
        "desc": "Verified high tide / low tide water level data for the station.",
        "max_interval": timedelta(days=10 * 364 - 1),
    },
    "daily_mean": {
        "group": "tide",
        "desc": "Verified daily mean water level data for the station.\nNOTE: Great Lakes stations only. Only available with time_zone=LST",
        "max_interval": timedelta(days=10 * 364 - 1),
    },
    "monthly_mean": {
        "group": "tide",
        "desc": "Verified monthly mean water level data for the station.",
        "max_interval": timedelta(days=200 * 364 - 1),
    },
    "one_minute_water_level": {
        "group": "tide",
        "desc": "Preliminary 1-minute interval water level data for the station.",
        "max_interval": timedelta(days=4),
    },
    "predictions": {
        "group": "tide",
        "desc": "Water level / tide prediction data for the station.\nNOTE: See Interval for available data interval options and data length limitations.",
    },
    "datums": {
        "group": "tide",
        "desc": "Observed tidal datum values at the station for the present National Tidal Datum Epoch (NTDE).",
    },
    "air_gap": {
        "group": "tide",
        "desc": "Air Gap (distance between a bridge and the water's surface) at the station.",
    },
    "air_temperature": {
        "group": "met",
        "desc": "Air temperature as measured at the station.",
    },
    "water_temperature": {
        "group": "met",
        "desc": "Water temperature as measured at the station.",
    },
    "wind": {
        "group": "met",
        "desc": "Wind speed, direction, and gusts as measured at the station.",
    },
    "air_pressure": {
        "group": "met",
        "desc": "Barometric pressure as measured at the station.",
    },
    "conductivity": {
        "group": "met",
        "desc": "The water's conductivity as measured at the station.",
    },
    "visibility": {
        "group": "met",
        "desc": "Visibility (atmospheric clarity) as measured at the station. (Units of Nautical Miles or Kilometers)",
    },
    "humidity": {
        "group": "met",
        "desc": "Relative humidity as measured at the station.",
    },
    "salinity": {
        "group": "met",
        "desc": "Salinity and specific gravity data for the station.",
    },
    "currents": {
        "group": "cur",
        "desc": "Currents data for the station. NOTE:  Default data interval is 6-minute interval data.Use with “interval=h” for hourly data",
    },
    "currents_predictions": {
        "group": "cur",
        "desc": "Currents prediction data for the stations. NOTE:  See Interval for options available and data length limitations.",
    },
    "ofs_water_level": {
        "group": "ofs",
        "desc": "Currents data for the station. NOTE:  Default data interval is 6-minute interval data.Use with “interval=h” for hourly data",
    },
}

DATUMS = {
    "CRD": {
        "desc": "Columbia River Datum. NOTE: Only available for certain stations on the Columbia River, Washington/Oregon"
    },
    "IGLD": {
        "desc": "International Great Lakes Datum NOTE:  Only available for Great Lakes stations."
    },
    "LWD": {
        "desc": "Great Lakes Low Water Datum (Nautical Chart Datum for the Great Lakes). NOTE:  Only available for Great Lakes Stations"
    },
    "MHHW": {"desc": "Mean Higher High Water"},
    "MHW": {"desc": "Mean High Water"},
    "MTL": {"desc": "Mean Tide Level"},
    "MSL": {"desc": "Mean Sea Level"},
    "MLW": {"desc": "Mean Low Water"},
    "MLLW": {
        "desc": "Mean Lower Low Water (Nautical Chart Datum for all U.S. coastal waters). NOTE:  Subordinate tide prediction stations must use “datum=MLLW”"
    },
    "NAVD": {
        "desc": "North American Vertical Datum NOTE:  This datum is not available for all stations."
    },
    "STND": {
        "desc": "Station Datum - original reference that all data is collected to, uniquely defined for each station."
    },
}

UNITS = {
        "metric": {"desc": "Metric units (Celsius, meters, cm/s appropriate for the data)\nNOTE: Visibility data is kilometers (km)"},
        "english": {"desc": "English units (fahrenheit, feet, knots appropriate for the data)\nNOTE: Visibility data is Nautical Miles (nm)"},
}

TIME_ZONES = {
        "gmt": {"desc": "Greenwich Mean Time"},
        "lst": {"desc": "Local Standard Time, not corrected for Daylight Saving Time, local to the requested station."},
        "lst_ldt": {"desc": "Local Standard Time, corrected for Daylight Saving Time when appropriate, local to the requested station"},
}

INTERVALS = ["h", "hilo", "max_slack"] + [
    str(x) for x in [1, 5, 6, 10, 15, 30, 60]
]

FORMATS = {
        "xml": {"desc": "Extensible Markup Language. This format is an industry standard for data."},
        "json": {"desc": "Javascript Object Notation. This format is useful for direct import to a javascript plotting library. Parsers are available for other languages such as Java and Perl."},
        "csv": {"desc": "Comma Separated Values. This format is suitable for import into Microsoft Excel or other spreadsheet programs. "},
}

NOAA_STATIONS = pd.read_json(
    Path(Path(__file__).parents[1] / "configs/noaa_stations.json"))

STATION_IDS = [int(x) for x in NOAA_STATIONS['ID'].dropna()]

REGIONS = ['Alabama', 'Alaska', 'Bermuda', 'California',
           'Map icon Caribbean/Central America', 'Caribbean/Central America',
           'Connecticut', 'Delaware', 'District of Columbia', 'Florida',
           'Georgia', 'Great Lakes - Detroit River',
           'Map icon Great Lakes - Lake Erie', 'Great Lakes - Lake Erie',
           'Lake Huron', 'Great Lakes - Lake Michigan',
           'Great Lakes - Lake Ontario', 'Great Lakes - Lake St. Clair',
           'Great Lakes - Lake Superior', 'Great Lakes - Niagra River',
           'Great Lakes - St. Clair River',
           'Great Lakes - St. Lawrence River',
           'Great Lakes - St. Marys River', 'Hawaii', 'Louisiana', 'Maine',
           'Maryland', 'Massachusetts', 'Mississippi', 'New Jersey',
           'New York', 'North Carolina', 'Oregon', 'Pacific Islands',
           'Pennsylvania', 'Rhode Island', 'South Carolina', 'Texas',
           'Virginia', 'Washington']
