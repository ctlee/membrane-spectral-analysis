import numpy as np
import warnings
from .surface import (
    get_z_surface,
    get_interpolated_z_surface,
)
from .leaflet_finder import determine_leaflets

import MDAnalysis
from MDAnalysis.analysis.base import AnalysisBase

import logging

MDAnalysis.start_logging()

logger = logging.getLogger("MDAnalysis.MDAKit.membrane_spectral_analysis")

leaflets = ["upper", "lower"]


class MembraneSpectralAnalysis(AnalysisBase):
    def __init__(
        self,
        universe,
        select="all",
        n_x_bins=100,
        n_y_bins=100,
        x_range=None,
        y_range=None,
        wrap=True,
        interpolate=False,
        **kwargs,
    ):

        super().__init__(universe.universe.trajectory, **kwargs)

        self.ag = determine_leaflets(universe, select)
        # self.ag = universe.select_atoms(select)

        self.wrap = wrap
        self.interpolate = interpolate
        self.n_x_bins = n_x_bins
        self.n_y_bins = n_y_bins
        self.x_range = x_range if x_range else (0, universe.dimensions[0])
        self.y_range = y_range if y_range else (0, universe.dimensions[1])

        self.x_step = (x_range[1] - x_range[0]) / self.n_x_bins
        self.y_step = (y_range[1] - y_range[0]) / self.n_y_bins

        # Initialize mesh grid and wave vectors
        self.P, self.Q = np.mgrid[
            x_range[0] : x_range[1] : self.x_step,
            y_range[0] : y_range[1] : self.y_step,
        ]
        self.qx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(self.n_x_bins, self.x_step))
        self.qy = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(self.n_y_bins, self.y_step))

        # Raise if selection doesn't exist
        if len(self.ag) == 0:
            raise ValueError("Invalid selection. Empty AtomGroup.")

        # Only checks the first frame. NPT simulations not properly covered here.
        # Warning message if range doesn't cover entire dimensions of simulation box
        for dim_string, dim_range, num in [
            ("x", self.x_range, 0),
            ("y", self.y_range, 1),
        ]:
            if dim_range[1] < universe.dimensions[num]:
                msg = (
                    f"Grid range in {dim_string} does not cover entire "
                    "dimensions of simulation box.\n Minimum dimensions "
                    "must be equal to simulation box."
                )
                warnings.warn(msg)
                logger.warn(msg)

        # Apply wrapping coordinates
        if not self.wrap:
            # Warning
            msg = (
                " `wrap == False` may result in inaccurate calculation "
                "of membrane curvature. Surfaces will be derived from "
                "a reduced number of atoms. \n "
                " Ignore this warning if your trajectory has "
                " rotational/translational fit rotations! "
            )
            warnings.warn(msg)
            logger.warn(msg)

    def _prepare(self):
        # Initialize empty np.array with results
        self.results.z_surface = dict(
            zip(
                leaflets,
                [
                    np.full((self.n_frames, self.n_x_bins, self.n_y_bins), np.nan),
                    np.full((self.n_frames, self.n_x_bins, self.n_y_bins), np.nan),
                ],
            )
        )
        self.results.mean = dict(
            zip(
                leaflets,
                [
                    np.full((self.n_frames, self.n_x_bins, self.n_y_bins), np.nan),
                    np.full((self.n_frames, self.n_x_bins, self.n_y_bins), np.nan),
                ],
            )
        )
        self.results.gaussian = dict(
            zip(
                leaflets,
                [
                    np.full((self.n_frames, self.n_x_bins, self.n_y_bins), np.nan),
                    np.full((self.n_frames, self.n_x_bins, self.n_y_bins), np.nan),
                ],
            )
        )
        self.results.thickness = np.full(
            (self.n_frames, self.n_x_bins, self.n_y_bins), np.nan
        )
        self.results.height = np.full(
            (self.n_frames, self.n_x_bins, self.n_y_bins), np.nan
        )
        self.results.height_power_spectrum = np.full(
            (self.n_frames, self.n_x_bins, self.n_y_bins), np.nan
        )

    def _single_frame(self):
        # Apply wrapping coordinates
        if self.wrap:
            for leaflet in leaflets:
                self.ag[leaflet].wrap()

        cog = (self.ag["upper"] | self.ag["lower"]).center_of_geometry()

        for leaflet in leaflets:
            # Populate a slice with np.arrays of surface, mean, and gaussian per frame

            if self.interpolate:
                self.results.z_surface[leaflet][
                    self._frame_index
                ] = get_interpolated_z_surface(
                    self.ag[leaflet].positions,
                    self.P,
                    self.Q,
                    ag=self.ag[leaflet],
                )

            else:
                self.results.z_surface[leaflet][self._frame_index] = get_z_surface(
                    self.ag[leaflet].positions,
                    n_x_bins=self.n_x_bins,
                    n_y_bins=self.n_y_bins,
                    x_range=self.x_range,
                    y_range=self.y_range,
                )

            Zy, Zx = np.gradient(self.results.z_surface[leaflet][self._frame_index])
            Zxy, Zxx = np.gradient(Zx)
            Zyy, _ = np.gradient(Zy)

            # self.results.gaussian[self._frame_index] = gaussian_curvature(self.results.z_surface[self._frame_index])
            self.results.gaussian[leaflet][self._frame_index] = (
                Zxx * Zyy - (Zxy**2)
            ) / (1 + (Zx**2) + (Zy**2)) ** 2

            # self.results.mean[self._frame_index] = mean_curvature(self.results.z_surface[self._frame_index])
            self.results.mean[leaflet][self._frame_index] = (
                (1 + Zx**2) * Zyy + (1 + Zy**2) * Zxx - 2 * Zx * Zy * Zxy
            )
            self.results.mean[leaflet][self._frame_index] /= 2 * (
                1 + Zx**2 + Zy**2
            ) ** (1.5)

        self.results.thickness[self._frame_index] = (
            self.results.z_surface["upper"][self._frame_index]
            - self.results.z_surface["lower"][self._frame_index]
        )

        self.results.height[self._frame_index] = (
            self.results.z_surface["upper"][self._frame_index]
            + self.results.z_surface["lower"][self._frame_index]
        ) / 2.0 - cog[2]

        # Height fluctuation spectra
        FFT_height = self.x_step * np.fft.fft2(self.results.height[self._frame_index], norm='ortho')
        self.results.height_power_spectrum[self._frame_index] = (
            np.abs(np.fft.fftshift(FFT_height)) ** 2
        )  # Angstrom^4

    def _conclude(self):
        pass
