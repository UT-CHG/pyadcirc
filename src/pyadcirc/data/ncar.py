"""
ncar

Utilities for pulling data from NCAR Research Data Archive.

See NCAR's Websites for more information:
    - https://rda.ucar.edu

TODO:
    + Remove TOKEN_PATH if Authentication fails with tokene because it's gone
    stale and then retry login (so link appears). Error is:

        >       raise self.error_class(r)
E       globus_sdk.services.auth.errors.AuthAPIError: ('POST', 'https://auth.globus.org/v2/oauth2/token', None, 400, 'Error', '{"error":"invalid_grant"}')

/opt/miniconda3/envs/pyadcirc_dev/lib/python3.10/site-packages/globus_sdk/client.py:310: AuthAPIError

    + Handle error when TACC endpoint not initalized:
        E       globus_sdk.services.transfer.errors.TransferAPIError: ('POST', 'https://transfer.api.globus.org/v0.10/transfer', 'Bearer', 409, 'NoCredException', "Credentials are needed for 'tacc#142d715e-8939-11e9-b807-0a37f382de32'", 'LlfwUcHt2')

/opt/miniconda3/envs/pyadcirc_dev/lib/python3.10/site-packages/globus_sdk/client.py:310: TransferAPIError
"""
import pdb
from io import StringIO
from pathlib import Path
from typing import List, Tuple
from fnmatch import fnmatch
from fnmatch import filter as ffilter

from prettytable import PrettyTable
import globus_sdk
import pandas as pd
import requests
import xarray as xa

NCAR_ENDPOINT = "1e128d3c-852d-11e8-9546-0a6d4e044368"
TOKEN_PATH = Path.home() / ".globus_token"

# TODO: Move to utils
def sizeof_fmt(num, suffix="B"):
    """
    Formats number representing bytes to string with appropriate size unit.

    Parameters
    ----------
    num : int,float
        Number to convert to string with bytes unit.
    suffix : str, default=B
        Suffix to use for measurement. Kilobytes will be KiB with default.

    Returns
    -------
    fmt_str : str
        Formatted string with bytes units.

    Notes
    -----
    Taken from
    stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size
    """
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} Yi{suffix}"


def init_globus_client(client_id: str, token_path: str = None):
    """Initialize Globus Client

    Initialies a globus transfer client through SDK for data transfers. If
    refresh token is not found, then asks user to go to website to input token
    to initialize client and stores refresh token.


    Parameters
    ----------
    client_id : str
        Client ID of your Globus Client App to use.
    token_path : str, optional
        Path to store refresh token. Defaults to file named .globus_token in
        users home directory.

    Returns
    -------
    authorizer : globus_sdk.authorizers.refresh_token.RefreshTokenAuthorizer
        `RefreshTokenAuthorizer` object that can be used to initialize Globus
        transfer clients to different endpoints.

    Notes
    -----
    Go to https://globus-sdk-python.readthedocs.io/en/stable/tutorial.html for
    more info on how to use refresh tokens.
    """
    client = globus_sdk.NativeAppAuthClient(client_id)
    token_path = str(TOKEN_PATH) if token_path is None else token_path

    if not Path(token_path).exists():
        client.oauth2_start_flow(refresh_tokens=True)

        print(
            "Please go to this URL and login: {0}".format(
                client.oauth2_get_authorize_url()
            )
        )
        auth_code = input("Please enter the code here: ").strip()
        res = client.oauth2_exchange_code_for_tokens(auth_code)

        # let's get stuff for the Globus Transfer service
        tk = res.by_resource_server["transfer.api.globus.org"]["refresh_token"]

        with open(Path(token_path).absolute(), "w") as tk_file:
            tk_file.write(tk)
    else:
        tk = Path(token_path).read_text()

    authorizer = globus_sdk.RefreshTokenAuthorizer(tk, client)

    return client, authorizer


class NCARDataTransfer(object):

    """Docstring for NCARDataTransfer."""

    def __init__(self, client_id: str):
        """Initialize an NCARDataTransfer object"""

        # Initialize client and authorizer
        self.client, self.authorizer = init_globus_client(client_id, TOKEN_PATH)

        self.transfer_client = globus_sdk.TransferClient(authorizer=self.authorizer)

        self.transfer_data = None

    def list_files(
        self,
        ds_id: str,
        data_types: List[str],
        start_date: str,
        end_date: str,
        pp: bool = False,
    ):
        """List NCAR Dataset Files

        Returns list of data files available of given `data_type` from
        `start_date` and `end_date`.

        Parameters
        ----------
        ds_id : str
            ID of dataset to access.
        data_types : List[str]
            Data types from the dataset to access.
        start_date : str
            Data start date. Must be in format `pandas.to_datetime` recognizes.
        end_date : str
            Data end date. Must be in format `pandas.to_datetime` recognizes.

        Returns
        -------
        files : List[str]
            List of files in directory.
        """

        # Get date ranges
        date_range = pd.date_range(
            *(pd.to_datetime([start_date, end_date]) + pd.offsets.MonthEnd()), freq="M"
        )
        years = date_range.strftime("%Y").tolist()
        months = date_range.strftime("%Y%m").tolist()
        years_unique = list(set(years))
        years_unique.sort()

        f_info = []
        for year in years_unique:

            folder = f"/{ds_id}/{year}"
            files = self.transfer_client.operation_ls(NCAR_ENDPOINT, path=folder)

            for idx, m in enumerate(months):
                if years[idx] == year:
                    for d in data_types:
                        f_info += [
                            (folder, f["name"], f["size"])
                            for f in files
                            if fnmatch(f["name"], f"{d}.{m}.grb2")
                        ]

        f_info = [{"name": x[1], "size": x[2], "folder": x[0]} for x in f_info]

        if pp:
            x = PrettyTable()
            x.field_names = ["name", "size"]
            for f in f_info:
                x.add_row([f["name"], sizeof_fmt(f["size"])])
            print(x)

        return f_info

    def stage(
        self,
        ds_id: str,
        data_types: List[str],
        target_endpoint: str,
        target_path: str,
        start_date: str,
        end_date: str,
    ):
        """Stage NCAR Data Transfer

        Stages data for a globus data transfer from an NCAR dataset globus
        endpoint to a desired target globus endpoint and path.

        Parameters
        ----------
        ds_id : str
            ID of dataset to access.
        data_types : List[str]
            Data types from the dataset to access.
        target_endpoint : str
            ID of target endpoint.
        target_path : str
            Path on target endpoint to place data.
        start_date : str
            Data start date. Must be in format `pandas.to_datetime` recognizes.
        end_date : str
            Data end date. Must be in format `pandas.to_datetime` recognizes.

        Returns
        -------
        transfer_data : dict
            Dictionary containing info on data transfer to be submitted. Key
            'DATA' contains list of files to be transferred.
        """

        f_info = self.list_files(ds_id, data_types, start_date, end_date)

        source_files = [str(Path(f["folder"]) / f["name"]) for f in f_info]
        dest_files = [str(Path(target_path) / f["name"]) for f in f_info]

        transfer_data = globus_sdk.TransferData(
            self.transfer_client,
            NCAR_ENDPOINT,
            target_endpoint,
            label="NCAR weather Data",
            sync_level="checksum",
        )
        for s, d in zip(source_files, dest_files):
            transfer_data.add_item(s, d)

        return transfer_data

    def submit(self, transfer_data: dict):
        """Submit Datatransfer

        Parameters
        ----------
        transfer_data : dict
            Dictionary built by a `globus_sdk.TransferData` object, to submit
            for transfer.

        Returns
        -------


        """
        return self.transfer_client.submit_transfer(transfer_data)
