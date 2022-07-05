"""
MembraneCurvature
=======================================

:Author: Estefania Barreto-Ojeda
:Year: 2021
:Copyright: GNU Public License v3

MembraneCurvature calculates the mean and Gaussian curvature of
surfaces derived from a selection of reference.

Mean curvature is calculated in units of Å :sup:`-1` and Gaussian curvature
in units of Å :sup:`-2`.
"""

import numpy as np
import warnings
from .surface import get_z_surface, get_interpolated_z_surface
from .leaflet_finder import determine_leaflets

# from .curvature import mean_curvature, gaussian_curvature

import MDAnalysis
from MDAnalysis.analysis.base import AnalysisBase

import logging

MDAnalysis.start_logging()

logger = logging.getLogger("MDAnalysis.MDAKit.membrane_curvature")

leaflets = ["upper", "lower"]


class MembraneCurvature(AnalysisBase):
    r"""
    MembraneCurvature is a tool to calculate membrane curvature.

    Parameters
    ----------
    universe : Universe or AtomGroup
        An MDAnalysis Universe object.
    select : str or iterable of str, optional.
        The selection string of an atom selection to use as a
        reference to derive a surface.
    wrap : bool, optional
        Apply coordinate wrapping to pack atoms into the primary unit cell.
    interpolate : bool, optional
        Obtain surfaces using cubic interpolation
    n_x_bins : int, optional, default: '100'
        Number of bins in grid in the x dimension.
    n_y_bins : int, optional, default: '100'
        Number of bins in grid in the y dimension.
    x_range : tuple of (float, float), optional, default: (0, `universe.dimensions[0]`)
        Range of coordinates (min, max) in the x dimension.
    y_range : tuple of (float, float), optional, default: (0, `universe.dimensions[1]`)
        Range of coordinates (min, max) in the y dimension.

    Attributes
    ----------
    results.z_surface : ndarray
        Surface derived from atom selection in every frame.
        Array of shape (`n_frames`, `n_x_bins`, `n_y_bins`)
    results.mean_curvature : ndarray
        Mean curvature associated to the surface.
        Array of shape (`n_frames`, `n_x_bins`, `n_y_bins`)
    results.gaussian_curvature : ndarray
        Gaussian curvature associated to the surface.
        Arrays of shape (`n_frames`, `n_x_bins`, `n_y_bins`)
    results.average_z_surface : ndarray
        Average of the array elements in `z_surface`.
        Each array has shape (`n_x_bins`, `n_y_bins`)
    results.average_mean_curvature : ndarray
        Average of the array elements in `mean_curvature`.
        Each array has shape (`n_x_bins`, `n_y_bins`)
    results.average_gaussian_curvature: ndarray
        Average of the array elements in `gaussian_curvature`.
        Each array has shape (`n_x_bins`, `n_y_bins`)


    See also
    --------
    :class:`~MDAnalysis.transformations.wrap.wrap`
        Wrap/unwrap the atoms of a given AtomGroup in the unit cell.

    Notes
    -----
    Use `wrap=True` to translates the atoms of your `mda.Universe` back
    in the unit cell. Use `wrap=False` for processed trajectories where
    rotational/translational fit is performed.

    For more details on when to use `wrap=True`, check the :ref:`usage`
    page.


    The derived surface and calculated curvatures are available in the
    :attr:`results` attributes.

    The attribute :attr:`~MembraneCurvature.results.average_z_surface` contains
    the derived surface averaged over the `n_frames` of the trajectory.

    The attributes :attr:`~MembraneCurvature.results.average_mean_curvature` and
    :attr:`~MembraneCurvature.results.average_gaussian_curvature` contain the
    computed values of mean and Gaussian curvature averaged over the `n_frames`
    of the trajectory.

    Example
    -----------
    You can pass a universe containing your selection of reference::

        import MDAnalysis as mda
        from membrane_curvature.base import MembraneCurvature

        u = mda.Universe(coordinates, trajectory)
        mc = MembraneCurvature(u).run()

        surface =  mc.results.average_z_surface
        mean_curvature =  mc.results.average_mean_curvature
        gaussian_curvature = mc.results.average_gaussian_curvature

    The respective 2D curvature plots can be obtained using the `matplotlib`
    package for data visualization via :func:`~matplotlib.pyplot.contourf` or
    :func:`~matplotlib.pyplot.imshow`.

    For specific examples visit the :ref:`usage` page.
    Check the :ref:`visualization` page for more examples to plot
    MembraneCurvature results using :func:`~matplotlib.pyplot.contourf`
    and :func:`~matplotlib.pyplot.imshow`.

    """

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

        self.P, self.Q = np.mgrid[
            x_range[0] : x_range[1] : self.x_step,
            y_range[0] : y_range[1] : self.y_step,
        ]

        self.qx = 2 * np.pi * np.fft.fftfreq(n_x_bins, self.x_step)
        self.qy = 2 * np.pi * np.fft.fftfreq(n_y_bins, self.y_step)

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

        self.results.thickness_power_spectrum = np.full(
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

        # cog = (self.ag['upper'] | self.ag['lower']).center_of_geometry()

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

            Zy, Zx = np.gradient(self.results.z_surface[self._frame_index])
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
            self.results.z_surface[self._frame_index]["upper"]
            - self.results.z_surface[self._frame_index]["lower"]
        )

        self.results.height[self._frame_index] = (
            self.results.z_surface[self._frame_index]["upper"]
            + self.results.z_surface[self._frame_index]["lower"]
        ) / 2.0  # - cog[2]

        ### Assumes x, y have same step...
        FFT = np.fft.fft2(
            self.results.thickness[self._frame_index]
            - np.nanmean(self.results.thickness[self._frame_index])
        )
        FFT *= self.x_step / len(FFT)
        self.results.thickness_power_spectrum = np.square(p.abs(np.fft.fftshift(FFT)))

        FFT = np.fft.fft2(
            self.results.height[self._frame_index]
            - np.nanmean(self.results.height[self._frame_index])
        )
        FFT *= self.x_step / len(FFT)
        self.results.height_power_spectrum = np.square(p.abs(np.fft.fftshift(FFT)))

    def _conclude(self):
        pass
        # self.results.average_z_surface = np.nanmean(self.results.z_surface, axis=0)
        # self.results.average_mean = np.nanmean(self.results.mean, axis=0)
        # self.results.average_gaussian = np.nanmean(self.results.gaussian, axis=0)
