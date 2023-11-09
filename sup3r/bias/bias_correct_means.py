"""Classes to compute means from vortex and era data and compute bias
correction factors.

Vortex mean files can be downloaded from IRENA.
https://globalatlas.irena.org/workspace
"""


import calendar
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from rex import Resource
from scipy.interpolate import interp1d
from sklearn.neighbors import BallTree

from sup3r.postprocessing.file_handling import OutputHandler, RexOutputs
from sup3r.preprocessing.feature_handling import Feature
from sup3r.utilities import VERSION_RECORD

logger = logging.getLogger(__name__)


class VortexMeanPrepper:
    """Class for converting monthly vortex tif files for each height to a
    single h5 files containing all monthly means for all requested output
    heights.
    """

    def __init__(self, path_pattern, in_heights, out_heights, overwrite=False):
        """
        Parameters
        ----------
        path_pattern : str
            Pattern for input tif files. Needs to include {month} and {height}
            format keys.
        in_heights : list
            List of heights for input files.
        out_heights : list
            List of output heights used for interpolation
        overwrite : bool
            Whether to overwrite intermediate netcdf files containing the
            interpolated masked monthly means.
        """
        msg = 'path_pattern needs to have {month} and {height} format keys'
        assert '{month}' in path_pattern and '{year}' in path_pattern, msg
        self.path_pattern = path_pattern
        self.in_heights = in_heights
        self.out_heights = out_heights
        self.out_dir = os.path.dirname(path_pattern)
        self.overwrite = overwrite
        self._mask = None

    @property
    def in_features(self):
        """List of features corresponding to input heights."""
        return [f"windspeed_{h}m" for h in self.in_heights]

    @property
    def out_features(self):
        """List of features corresponding to output heights"""
        return [f"windspeed_{h}m" for h in self.out_heights]

    def get_input_file(self, month, height):
        """Get vortex tif file for given month and height."""
        return self.path_pattern.format(month=month, height=height)

    def get_height_files(self, month):
        """Get set of netcdf files for given month"""
        files = []
        for height in self.in_heights:
            infile = self.get_input_file(month, height)
            outfile = infile.replace(".tif", ".nc")
            files.append(outfile)
        return files

    @property
    def input_files(self):
        """Get list of all input files used for h5 meta."""
        files = []
        for height in self.in_heights:
            for i in range(1, 13):
                month = calendar.month_name[i]
                files.append(self.get_input_file(month, height))
        return files

    def get_output_file(self, month):
        """Get name of netcdf file for a given month."""
        return os.path.join(
            self.out_dir.replace("{month}", month), f"{month}.nc"
        )

    @property
    def output_files(self):
        """List of output monthly output files each with windspeed for all
        input heights
        """
        files = []
        for i in range(1, 13):
            month = calendar.month_name[i]
            files.append(self.get_output_file(month))
        return files

    def convert_month_height_tif(self, month, height):
        """Get windspeed mean for the given month and hub height from the
        corresponding input file and write this to a netcdf file.
        """
        infile = self.get_input_file(month, height)
        logger.info(f"Getting mean windspeed_{height}m for {month}.")
        outfile = infile.replace(".tif", ".nc")
        if os.path.exists(outfile) and self.overwrite:
            os.remove(outfile)

        if not os.path.exists(outfile) or self.overwrite:
            tmp = rioxarray.open_rasterio(infile)
            ds = tmp.to_dataset("band")
            ds = ds.rename(
                {1: f"windspeed_{height}m", "x": "longitude", "y": "latitude"}
            )
            ds.to_netcdf(outfile)
        return outfile

    def convert_month_tif(self, month):
        """Write netcdf files for all heights for the given month."""
        for height in self.in_heights:
            self.convert_month_height_tif(month, height)

    def convert_all_tifs(self):
        """Write netcdf files for all heights for all months."""
        for i in range(1, 13):
            month = calendar.month_name[i]
            logger.info(f"Converting tif files to netcdf files for {month}")
            self.convert_month_tif(month)

    @property
    def mask(self):
        """Mask coordinates without data"""
        if self._mask is None:
            with xr.open_mfdataset(self.get_height_files("January")) as res:
                mask = (res[self.in_features[0]] != -999) & (
                    ~np.isnan(res[self.in_features[0]])
                )
                for feat in self.in_features[1:]:
                    tmp = (res[feat] != -999) & (~np.isnan(res[feat]))
                    mask = mask & tmp
                self._mask = np.array(mask).flatten()
        return self._mask

    def get_month(self, month):
        """Get interpolated means for all hub heights for the given month.

        Parameters
        ----------
        month : str
            Name of month to get data for

        Returns
        -------
        data : xarray.Dataset
            xarray dataset object containing interpolated monthly windspeed
            means for all input and output heights

        """
        month_file = self.get_output_file(month)
        if os.path.exists(month_file) and self.overwrite:
            os.remove(month_file)

        if os.path.exists(month_file) and not self.overwrite:
            logger.info(f"Loading month_file {month_file}.")
            data = xr.open_dataset(month_file)
        else:
            logger.info(
                "Getting mean windspeed for all heights "
                f"({self.in_heights}) for {month}"
            )
            data = xr.open_mfdataset(self.get_height_files(month))
            logger.info(
                "Interpolating windspeed for all heights "
                f"({self.out_heights}) for {month}."
            )
            data = self.interp(data)
            data.to_netcdf(month_file)
            logger.info(
                "Saved interpolated means for all heights for "
                f"{month} to {month_file}."
            )
        return data

    def interp(self, data):
        """Interpolate data to requested output heights.

        Parameters
        ----------
        data : xarray.Dataset
            xarray dataset object containing windspeed for all input heights

        Returns
        -------
        data : xarray.Dataset
            xarray dataset object containing windspeed for all input and output
            heights
        """
        var_array = np.zeros(
            (
                len(data.latitude) * len(data.longitude),
                len(self.in_heights),
            ),
            dtype=np.float32,
        )
        lev_array = var_array.copy()
        for i, (h, feat) in enumerate(zip(self.in_heights, self.in_features)):
            var_array[..., i] = data[feat].values.flatten()
            lev_array[..., i] = h

        logger.info(
            f"Interpolating {self.in_features} to {self.out_features} "
            f"for {var_array.shape[0]} coordinates."
        )
        tmp = [
            interp1d(h, v, fill_value="extrapolate")(self.out_heights)
            for h, v in zip(lev_array[self.mask], var_array[self.mask])
        ]
        out = np.full(
            (len(data.latitude), len(data.longitude), len(self.out_heights)),
            np.nan,
            dtype=np.float32,
        )
        out[self.mask.reshape((len(data.latitude), len(data.longitude)))] = tmp
        for i, feat in enumerate(self.out_features):
            if feat not in data:
                data[feat] = (("latitude", "longitude"), out[..., i])
        return data

    def get_lat_lon(self):
        """Get lat lon grid"""
        with xr.open_mfdataset(self.get_height_files("January")) as res:
            lons, lats = np.meshgrid(
                res["longitude"].values, res["latitude"].values
            )
        return np.array(lats), np.array(lons)

    def get_all_data(self):
        """Get interpolated monthly means for all out heights as a dictionary
        to use for h5 writing.

        Returns
        -------
        out : dict
            Dictionary of arrays containing monthly means for each hub height.
            Also includes latitude and longitude. Spatial dimensions are
            flattened
        """
        data_dict = {}
        lats, lons = self.get_lat_lon()
        data_dict["latitude"] = lats.flatten()[self.mask]
        data_dict["longitude"] = lons.flatten()[self.mask]
        s_num = len(data_dict["longitude"])
        for i in range(1, 13):
            month = calendar.month_name[i]
            out = self.get_month(month)
            for feat in self.out_features:
                if feat not in data_dict:
                    data_dict[feat] = np.full((s_num, 12), np.nan)
                data = out[feat].values.flatten()[self.mask]
                data_dict[feat][..., i - 1] = data
        return data_dict

    @property
    def meta(self):
        """Get a meta data dictionary on how this data is prepared"""
        meta = {
            "input_files": self.input_files,
            "class": str(self.__class__),
            "version_record": VERSION_RECORD,
        }
        return meta

    def write_data(self, fp_out, out):
        """Write monthly means for all heights to h5 file"""
        if fp_out is not None:
            if not os.path.exists(os.path.dirname(fp_out)):
                os.makedirs(os.path.dirname(fp_out), exist_ok=True)

            if not os.path.exists(fp_out) or self.overwrite:
                with h5py.File(fp_out, "w") as f:
                    for dset, data in out.items():
                        f.create_dataset(dset, data=data)
                        logger.info(f"Added {dset} to {fp_out}.")

                    for k, v in self.meta.items():
                        f.attrs[k] = json.dumps(v)

                    logger.info(
                        f"Wrote monthly means for all out heights: {fp_out}"
                    )
            elif os.path.exists(fp_out):
                logger.info(f"{fp_out} already exists and overwrite=False.")

    @classmethod
    def run(
        cls, path_pattern, in_heights, out_heights, fp_out, overwrite=False
    ):
        """Read vortex tif files, convert these to monthly netcdf files for all
        input heights, interpolate this data to requested output heights, mask
        fill values, and write all data to h5 file.

        Parameters
        ----------
        path_pattern : str
            Pattern for input tif files. Needs to include {month} and {height}
            format keys.
        in_heights : list
            List of heights for input files.
        out_heights : list
            List of output heights used for interpolation
        fp_out : str
            Name of final h5 output file to write with means.
        overwrite : bool
            Whether to overwrite intermediate netcdf files containing the
            interpolated masked monthly means.
        """
        vprep = cls(path_pattern, in_heights, out_heights, overwrite=overwrite)
        vprep.convert_all_tifs()
        out = vprep.get_all_data()
        vprep.write_data(fp_out, out)


class EraMeanPrepper:
    """Class to compute monthly windspeed means from ERA data."""

    def __init__(self, era_pattern, years, features):
        """Parameters
        ----------
        era_pattern : str
            Pattern pointing to era files with u/v wind components at the given
            heights. Must have a {year} format key.
        years : list
            List of ERA years to use for calculating means.
        features : list
            List of features to compute means for. e.g. ['windspeed_10m']
        """
        self.era_pattern = era_pattern
        self.years = years
        self.features = features
        self.lats, self.lons = self.get_lat_lon()

    @property
    def shape(self):
        """Get shape of spatial dimensions (lats, lons)"""
        return self.lats.shape

    @property
    def heights(self):
        """List of feature heights"""
        heights = [Feature.get_height(feature) for feature in self.features]
        return heights

    @property
    def input_files(self):
        """List of ERA input files to use for calculating means."""
        return [self.era_pattern.format(year=year) for year in self.years]

    def get_lat_lon(self):
        """Get arrays of latitude and longitude for ERA domain"""
        with xr.open_dataset(self.input_files[0]) as res:
            lons, lats = np.meshgrid(
                res["longitude"].values, res["latitude"].values
            )
        return lats, lons

    def get_windspeed(self, data, height):
        """Compute windspeed from u/v wind components from given data.

        Parameters
        ----------
        data : xarray.Dataset
            xarray dataset object for a year of ERA data. Must include u/v
            components for the given height. e.g. u_{height}m, v_{height}m.
        height : int
            Height to compute windspeed for.
        """
        return np.hypot(
            data[f"u_{height}m"].values, data[f"v_{height}m"].values
        )

    def get_month_mean(self, data, height, month):
        """Get windspeed_{height}m mean for the given month.

        Parameters
        ----------
        data : xarray.Dataset
            xarray dataset object for a year of ERA data. Must include u/v
            components for the given height. e.g. u_{height}m, v_{height}m.
        height : int
            Height to compute windspeed for.
        month : int
            Index of month to get mean for. e.g. 1 = Jan, 2 = Feb, etc.

        Returns
        -------
        out : np.ndarray
            Array of time averaged windspeed data for the given month.

        """
        mask = pd.to_datetime(data["time"]).month == month
        ws = self.get_windspeed(data, height)[mask]
        return ws.mean(axis=0)

    def get_all_means(self, height):
        """Get monthly means for all months across all given years for the
        given height.

        Parameters
        ----------
        height : int
            Height to compute windspeed for.

        Returns
        -------
        means : dict
            Dictionary of windspeed_{height}m means for each month
        """
        feature = self.features[self.heights.index(height)]
        means = {i: [] for i in range(1, 13)}
        for i, year in enumerate(self.years):
            logger.info(f"Getting means for year={year}, feature={feature}.")
            data = xr.open_dataset(self.input_files[i])
            for m in range(1, 13):
                means[m].append(self.get_month_mean(data, height, month=m))
        means = {m: np.dstack(arr).mean(axis=-1) for m, arr in means.items()}
        return means

    def write_csv(self, out, out_file):
        """Write monthly means to a csv file.

        Parameters
        ----------
        out : dict
            Dictionary of windspeed_{height}m means for each month
        out_file : str
            Name of csv output file.
        """
        logger.info(f"Writing means to {out_file}.")
        out = {
            f"{str(calendar.month_name[m])[:3]}_mean": v.flatten()
            for m, v in out.items()
        }
        df = pd.DataFrame.from_dict(out)
        df["latitude"] = self.lats.flatten()
        df["longitude"] = self.lons.flatten()
        df["gid"] = np.arange(len(df["latitude"]))
        df.to_csv(out_file)
        logger.info(f"Finished writing means for {out_file}.")

    @classmethod
    def run(cls, era_pattern, years, features, out_pattern):
        """Compute monthly windspeed means for the given heights, using the
        given years of ERA data, and write the means to csv files for each
        height.

        Parameters
        ----------
        era_pattern : str
            Pattern pointing to era files with u/v wind components at the given
            heights. Must have a {year} format key.
        years : list
            List of ERA years to use for calculating means.
        features : list
            List of features to compute means for. e.g. ['windspeed_10m']
        out_pattern : str
            Pattern pointing to csv files to write means to. Must have a
            {feature} format key.
        """
        em = cls(era_pattern=era_pattern, years=years, features=features)
        for height, feature in zip(em.heights, em.features):
            means = em.get_all_means(height)
            out_file = out_pattern.format(feature=feature)
            em.write_csv(means, out_file=out_file)
        logger.info(
            f"Finished writing means for years={years} and "
            f"heights={em.heights}."
        )


class BiasCorrectionFromMeans:
    """Class for getting bias correction factors from bias and base data files
    with precomputed monthly means.
    """

    MIN_DISTANCE = 1e-12

    def __init__(self, bias_fp, base_fp, dset, leaf_size=4):
        """Parameters
        ----------
        bias_fp : str
            Path to csv file containing means for biased data
        base_fp : str
            Path to csv file containing means for unbiased data
        dset : str
            Name of dataset to compute bias correction factor for
        leaf_size : int
            Leaf size for ball tree used to match bias and base grids
        """
        self.dset = dset
        self.bias_fp = bias_fp
        self.base_fp = base_fp
        self.leaf_size = leaf_size
        self.bias_means = pd.read_csv(bias_fp)
        self.base_means = Resource(base_fp)
        self.bias_meta = self.bias_means[["latitude", "longitude"]]
        self.base_meta = pd.DataFrame(columns=["latitude", "longitude"])
        self.base_meta["latitude"] = self.base_means["latitude"]
        self.base_meta["longitude"] = self.base_means["longitude"]
        self._base_tree = None
        logger.info(
            "Finished initializing BiasCorrectionFromMeans for "
            f"bias_fp={bias_fp}, base_fp={base_fp}, dset={dset}."
        )

    @property
    def base_tree(self):
        """Build ball tree from source_meta"""
        if self._base_tree is None:
            logger.info("Building ball tree for regridding.")
            self._base_tree = BallTree(
                np.deg2rad(self.base_meta),
                leaf_size=self.leaf_size,
                metric="haversine",
            )
        return self._base_tree

    @property
    def meta(self):
        """Get a meta data dictionary on how these bias factors were
        calculated
        """
        meta = {
            "base_fp": self.base_fp,
            "bias_fp": self.bias_fp,
            "dset": self.dset,
            "class": str(self.__class__),
            "version_record": VERSION_RECORD,
            "NOTES": ("scalar factors computed from base_data / bias_data."),
        }
        return meta

    @property
    def height(self):
        """Get feature height"""
        return Feature.get_height(self.dset)

    @property
    def u_name(self):
        """Get corresponding u component for given height"""
        return f"u_{self.height}m"

    @property
    def v_name(self):
        """Get corresponding v component for given height"""
        return f"v_{self.height}m"

    def get_base_data(self, knn=1):
        """Get means for baseline data."""
        logger.info(f"Getting base data for {self.dset}.")
        dists, gids = self.base_tree.query(np.deg2rad(self.bias_meta), k=knn)
        mask = dists < self.MIN_DISTANCE
        if mask.sum() > 0:
            logger.info(
                f"{np.sum(mask)} of {np.product(mask.shape)} "
                "distances are zero."
            )
        dists[mask] = self.MIN_DISTANCE
        weights = 1 / dists
        norm = np.sum(weights, axis=-1)
        out = self.base_means[self.dset, gids]
        out = np.einsum("ijk,ij->ik", out, weights) / norm[:, np.newaxis]
        return out

    def get_bias_data(self):
        """Get means for biased data."""
        logger.info(f"Getting bias data for {self.dset}.")
        cols = [col for col in self.bias_means.columns if "mean" in col]
        bias_data = self.bias_means[cols].to_numpy()
        return bias_data

    def get_corrections(self, global_scalar=1, knn=1):
        """Get bias correction factors."""
        logger.info(f"Getting correction factors for {self.dset}.")
        base_data = self.get_base_data(knn=knn)
        bias_data = self.get_bias_data()
        scaler = global_scalar * base_data / bias_data
        adder = np.zeros(scaler.shape)

        out = {
            "latitude": self.bias_meta["latitude"],
            "longitude": self.bias_meta["longitude"],
            f"base_{self.dset}_mean": base_data,
            f"bias_{self.dset}_mean": bias_data,
            f"{self.dset}_adder": adder,
            f"{self.dset}_scalar": scaler,
            f"{self.dset}_global_scalar": global_scalar,
        }
        return out

    def get_uv_corrections(self, global_scalar=1, knn=1):
        """Write windspeed bias correction factors for u/v components"""
        u_out = self.get_corrections(global_scalar=global_scalar, knn=knn)
        v_out = u_out.copy()
        u_out[f"{self.u_name}_scalar"] = u_out[f"{self.dset}_scalar"]
        v_out[f"{self.v_name}_scalar"] = v_out[f"{self.dset}_scalar"]
        u_out[f"{self.u_name}_adder"] = u_out[f"{self.dset}_adder"]
        v_out[f"{self.v_name}_adder"] = v_out[f"{self.dset}_adder"]
        return u_out, v_out

    def write_output(self, fp_out, out):
        """Write bias correction factors to h5 file."""
        logger.info(f"Writing correction factors to file: {fp_out}.")
        with h5py.File(fp_out, "w") as f:
            for dset, data in out.items():
                f.create_dataset(dset, data=data)
                logger.info(f"Added {dset} to {fp_out}.")
            for k, v in self.meta.items():
                f.attrs[k] = json.dumps(v)
        logger.info(f"Finished writing output to {fp_out}.")

    @classmethod
    def run(
        cls,
        bias_fp,
        base_fp,
        dset,
        fp_out,
        leaf_size=4,
        global_scalar=1.0,
        knn=1,
        out_shape=None,
    ):
        """Run bias correction factor computation and write.

        Parameters
        ----------
        bias_fp : str
            Path to csv file containing means for biased data
        base_fp : str
            Path to csv file containing means for unbiased data
        dset : str
            Name of dataset to compute bias correction factor for
        fp_out : str
            Name of output file containing bias correction factors
        leaf_size : int
            Leaf size for ball tree used to match bias and base grids
        global_scalar : float
            Optional global scalar to use for multiplying all bias correction
            factors. This can be used to improve systemic bias against
            observation data. This is just written to output files, not
            included in the stored bias correction factor values.
        knn : int
            Number of nearest neighbors to use when matching bias and base
            grids. This should be based on difference in resolution. e.g. if
            bias grid is 30km and base grid is 3km then knn should be 100 to
            aggregate 3km to 30km.
        out_shape : tuple | None
            Optional 2D shape for output. If this is provided then the bias
            correction arrays will be reshaped to this shape. If not provided
            the arrays will stay flattened. When using this to write bc files
            that will be used in a forward-pass routine this shape should be
            the same as the spatial shape of the forward-pass input data.
        """
        bc = cls(bias_fp=bias_fp, base_fp=base_fp, dset=dset,
                 leaf_size=leaf_size)
        out = bc.get_corrections(global_scalar=global_scalar, knn=knn)
        if out_shape is not None:
            out = cls._reshape_output(out, out_shape)
        bc.write_output(fp_out, out)

    @classmethod
    def _reshape_output(cls, out, out_shape):
        """Reshape output according to given output shape"""
        for k, v in out.items():
            if k in ("latitude", "longitude"):
                out[k] = np.array(v).reshape(out_shape)
            elif not isinstance(v, (int, float)):
                out[k] = np.array(v).reshape((*out_shape, 12))
        return out

    @classmethod
    def run_uv(
        cls,
        bias_fp,
        base_fp,
        dset,
        fp_pattern,
        global_scalar=1.0,
        knn=1,
        out_shape=None,
    ):
        """Run bias correction factor computation and write.

        Parameters
        ----------
        bias_fp : str
            Path to csv file containing means for biased data
        base_fp : str
            Path to csv file containing means for unbiased data
        dset : str
            Name of dataset to compute bias correction factor for
        fp_pattern : str
            Pattern for output file. Should contain {feature} format key.
        leaf_size : int
            Leaf size for ball tree used to match bias and base grids
        global_scalar : float
            Optional global scalar to use for multiplying all bias correction
            factors. This can be used to improve systemic bias against
            observation data. This is just written to output files, not
            included in the stored bias correction factor values.
        knn : int
            Number of nearest neighbors to use when matching bias and base
            grids. This should be based on difference in resolution. e.g. if
            bias grid is 30km and base grid is 3km then knn should be 100 to
            aggregate 3km to 30km.
        out_shape : tuple | None
            Optional 2D shape for output. If this is provided then the bias
            correction arrays will be reshaped to this shape. If not provided
            the arrays will stay flattened. When using this to write bc files
            that will be used in a forward-pass routine this shape should be
            the same as the spatial shape of the forward-pass input data.
        """
        bc = cls(bias_fp=bias_fp, base_fp=base_fp, dset=dset)
        out_u, out_v = bc.get_uv_corrections(
            global_scalar=global_scalar, knn=knn
        )
        if out_shape is not None:
            out_u = cls._reshape_output(out_u, out_shape)
            out_v = cls._reshape_output(out_v, out_shape)
        bc.write_output(fp_pattern.format(feature=bc.u_name), out_u)
        bc.write_output(fp_pattern.format(feature=bc.v_name), out_v)


class BiasCorrectUpdate:
    """Class for bias correcting existing files and writing corrected files."""

    @classmethod
    def get_bc_factors(cls, bc_file, dset, month, global_scalar=1):
        """Get bias correction factors for the given dset and month

        Parameters
        ----------
        bc_file : str
            Name of h5 file containing bias correction factors
        dset : str
            Name of dataset to apply bias correction factors for
        month : int
            Index of month to bias correct
        global_scalar : float
            Optional global scalar to multiply all bias correction
            factors. This can be used to improve systemic bias against
            observation data.

        Returns
        -------
        factors : ndarray
            Array of bias correction factors for the given dset and month.
        """
        with Resource(bc_file) as res:
            logger.info(
                f"Getting {dset} bias correction factors for month {month}."
            )
            bc_factor = res[f"{dset}_scalar", :, month - 1]
            factors = global_scalar * bc_factor
            logger.info(
                f"Retrieved {dset} bias correction factors for month {month}. "
                f"Using global_scalar={global_scalar}."
            )
        return factors

    @classmethod
    def _correct_month(
        cls, fh_in, month, out_file, dset, bc_file, global_scalar
    ):
        """Bias correct data for a given month.

        Parameters
        ----------
        fh_in : Resource()
            Resource handler for input file being corrected
        month : int
            Index of month to be corrected
        out_file : str
            Name of h5 file containing bias corrected data
        dset : str
            Name of dataset to bias correct
        bc_file : str
            Name of file containing bias correction factors for the given dset
        global_scalar : float
            Optional global scalar to multiply all bias correction
            factors. This can be used to improve systemic bias against
            observation data.
        """
        with RexOutputs(out_file, "a") as fh:
            mask = fh.time_index.month == month
            mask = np.arange(len(fh.time_index))[mask]
            mask = slice(mask[0], mask[-1] + 1)
            bc_factors = cls.get_bc_factors(
                bc_file=bc_file,
                dset=dset,
                month=month,
                global_scalar=global_scalar,
            )
            logger.info(f"Applying bias correction factors for month {month}")
            fh[dset, mask, :] = bc_factors * fh_in[dset, mask, :]

    @classmethod
    def update_file(
        cls,
        in_file,
        out_file,
        dset,
        bc_file,
        global_scalar=1,
        max_workers=None,
    ):
        """Update the in_file with bias corrected values for the given dset
        and write to out_file.

        Parameters
        ----------
        in_file : str
            Name of h5 file containing data to bias correct
        out_file : str
            Name of h5 file containing bias corrected data
        dset : str
            Name of dataset to bias correct
        bc_file : str
            Name of file containing bias correction factors for the given dset
        global_scalar : float
            Optional global scalar to multiply all bias correction
            factors. This can be used to improve systemic bias against
            observation data.
        max_workers : int | None
            Number of workers to use for parallel processing.
        """
        tmp_file = out_file.replace(".h5", ".h5.tmp")
        logger.info(f"Bias correcting {dset} in {in_file} with {bc_file}.")
        with Resource(in_file) as fh_in:
            OutputHandler._init_h5(
                tmp_file, fh_in.time_index, fh_in.meta, fh_in.global_attrs
            )
            OutputHandler._ensure_dset_in_output(tmp_file, dset)

            if max_workers == 1:
                for i in range(1, 13):
                    try:
                        cls._correct_month(
                            fh_in,
                            month=i,
                            out_file=tmp_file,
                            dset=dset,
                            bc_file=bc_file,
                            global_scalar=global_scalar,
                        )
                    except Exception as e:
                        raise RuntimeError(
                            f"Bias correction failed for month {i}."
                        ) from e

                    logger.info(
                        f"Added {dset} for month {i} to output file "
                        f"{tmp_file}."
                    )
            else:
                futures = {}
                with ThreadPoolExecutor(max_workers=max_workers) as exe:
                    for i in range(1, 13):
                        future = exe.submit(
                            cls._correct_month,
                            fh_in=fh_in,
                            month=i,
                            out_file=tmp_file,
                            dset=dset,
                            bc_file=bc_file,
                            global_scalar=global_scalar,
                        )
                        futures[future] = i

                        logger.info(
                            f"Submitted bias correction for month {i} "
                            f"to {tmp_file}."
                        )

                    for future in as_completed(futures):
                        _ = future.result()
                        i = futures[future]
                        logger.info(
                            f"Completed bias correction for month {i} "
                            f"to {tmp_file}."
                        )

        os.replace(tmp_file, out_file)
        msg = f"Saved bias corrected {dset} to: {out_file}"
        logger.info(msg)

    @classmethod
    def run(
        cls,
        in_file,
        out_file,
        dset,
        bc_file,
        overwrite=False,
        global_scalar=1,
        max_workers=None
    ):
        """Run bias correction update.

        Parameters
        ----------
        in_file : str
            Name of h5 file containing data to bias correct
        out_file : str
            Name of h5 file containing bias corrected data
        dset : str
            Name of dataset to bias correct
        bc_file : str
            Name of file containing bias correction factors for the given dset
        overwrite : bool
            Whether to overwrite the output file if it already exists.
        global_scalar : float
            Optional global scalar to multiply all bias correction
            factors. This can be used to improve systemic bias against
            observation data.
        max_workers : int | None
            Number of workers to use for parallel processing.
        """
        if os.path.exists(out_file) and not overwrite:
            logger.info(
                f"{out_file} already exists and overwrite=False. Skipping."
            )
        else:
            if os.path.exists(out_file) and overwrite:
                logger.info(
                    f"{out_file} exists but overwrite=True. "
                    f"Removing {out_file}."
                )
                os.remove(out_file)
            cls.update_file(
                in_file, out_file, dset, bc_file, global_scalar=global_scalar,
                max_workers=max_workers
            )
