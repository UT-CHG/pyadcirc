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
import json
import os
from datetime import timedelta
from fnmatch import fnmatch
from pathlib import Path
from typing import List

import globus_sdk
import pandas as pd
import requests
from prettytable import PrettyTable
from pyadcirc.utils import sizeof_fmt, check_file_status
from pyadcirc.io import cfsv2_grib_to_adcirc_owi


class NCARGlobusDataTransfer(object):

    """Docstring for NCARDataTransfer."""

    NCAR_ENDPOINT = "1e128d3c-852d-11e8-9546-0a6d4e044368"
    TOKEN_PATH = Path.home() / ".globus_token"

    def __init__(self, client_id: str):
        """Initialize an NCARDataTransfer object"""

        # Initialize client and authorizer
        self.client, self.authorizer = self._init_globus_client(
            client_id, self.TOKEN_PATH)

        self.transfer_client = globus_sdk.TransferClient(
            authorizer=self.authorizer)

        self.transfer_data = None

    def _init_globus_client(self, client_id: str, token_path: str = None):
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
        token_path = str(self.TOKEN_PATH) if token_path is None else token_path

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
            files = self.transfer_client.operation_ls(
                self.NCAR_ENDPOINT, path=folder)

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
            self.NCAR_ENDPOINT,
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


class NCARDownloader:
    """A class for downloading NCAR files without using GLOBUS (just username/pass)"""

    login_url = "https://rda.ucar.edu/cgi-bin/login"
    base_url = "https://rda.ucar.edu/data/"

    def __init__(self, configfile=os.environ["HOME"] + "/.ncar.json"):
        """Initialize NCAR credentials"""

        with open(configfile, "r") as fp:
            data = json.load(fp)
            self.email = data["email"]
            self.pw = data["pw"]

    def download(self, dataset, variables, start_date, end_date):
        date_range = pd.date_range(
            *(pd.to_datetime([start_date, end_date]) + pd.offsets.MonthEnd()), freq="M"
        )
        years = date_range.strftime("%Y").tolist()
        months = date_range.strftime("%Y%m").tolist()

        fnames = {v: [] for v in variables}
        for y, ym in zip(years, months):
            system = "cdas1" if int(y) >= 2011 else "gdas"
            for v in variables:
                fnames[v].append(
                    self.base_url + f"/{dataset}/{y}/{v}.{system}.{ym}.grb2"
                )
                print(fnames[v])

        auth = {"email": self.email, "passwd": self.pw, "action": "login"}
        ret = requests.post(self.login_url, data=auth)
        if ret.status_code != 200:
            print("Bad Authentication")
            print(ret.text)
            return

        for v, files in fnames.items():
            for f in files:
                self._download_one(f, cookies=ret.cookies)

        return {
            v: [os.path.basename(url) for url in urls]
            for v, urls in fnames.items()
        }

    def get_adcirc_forcing(self, start_date, end_date,
                           bounding_box=None, outdir=None):
        # The NCAR wind data is actually translated one hour ahead
        # Only forecasts are downloaded, NOT the analysis
        # So we need to pad the download range by a day :/
        download_start_date = pd.to_datetime(start_date) - timedelta(days=1)
        download_end_date = pd.to_datetime(end_date) + timedelta(days=1)
        dataset = "ds094.1" if download_start_date.year >= 2011 else "ds093.1"
        fnames = self.download(
            dataset=dataset,
            variables=["prmsl", "wnd10m", "icecon"],
            start_date=download_start_date,
            end_date=download_end_date,
        )

        variables = ["wnd10m"] + ["prmsl", "icecon"]
        windgrid = []
        for var in variables:
            ncar_vars_to_adcirc = {"prmsl": "fort.221",
                                   "wnd10m": "fort.222",
                                   "icecon": "fort.225"}

            outfile = ncar_vars_to_adcirc[var]
            if outdir is not None:
                outfile = outdir + "/" + outfile

            newgrid = windgrid if var == "prmsl" else None
            arrs = cfsv2_grib_to_adcirc_owi(
                fnames[var],
                date_range=[start_date, end_date],
                bounding_box=bounding_box,
                newgrid=newgrid,
                outfile=outfile,
            )
            if "wnd" in var:
                windgrid = arrs["latitude"], arrs["longitude"]

            # clean up
            for f in fnames[var]:
                os.remove(f)

    def _download_one(self, url, cookies, overwrite=False):
        file_base = os.path.basename(url)
        if os.path.exists(file_base) and not overwrite:
            return
        print("Downloading", file_base)
        req = requests.get(url, cookies=cookies,
                           allow_redirects=True, stream=True)
        filesize = int(req.headers["Content-length"])
        with open(file_base, "wb") as outfile:
            chunk_size = 1048576
            for chunk in req.iter_content(chunk_size=chunk_size):
                outfile.write(chunk)
                if chunk_size < filesize:
                    check_file_status(file_base, filesize)
        check_file_status(file_base, filesize)
        print()


if __name__ == "__main__":
    downloader = NCARDownloader()
    start_date, end_date = "2010-09-10", "2010-09-20"
    fnames = downloader.download(
        dataset="ds093.1",
        variables=["prmsl", "wnd10m", "icecon"],
        start_date=start_date,
        end_date=end_date,
    )
    # cfsv2_grib_to_adcirc(["wnd10m.cdas1.202209.grb2"],
    #                                    date_range=[start_date, end_date],
    #                                    bounding_box=[140, 240, 40, 80],
    #                                    outfile="fort.222")
    # netcdf_to_owi("ml_runs/marbock/fort.222.nc")
