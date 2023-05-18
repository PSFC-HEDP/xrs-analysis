from __future__ import annotations

import argparse
import os.path
from math import cos, sin, nan

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LinearLocator
from numpy.typing import NDArray
from scipy import interpolate, optimize

from cmap import CMAP

PERIODIC_TABLE = {
	2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
	10: "Ne", 13: "Al", 14: "Si", 18: "Ar", 36: "Kr", 54: "Xe",
}


def analyze_xrs(filename: str, energy_min: float, energy_max: float, atomic_numbers: set[int]):
	""" load an XRS image plate scan as a HDF5 file and, with the user’s help, extract and plot an x-ray spectrum
	    :param filename: either the path to the file (absolute or relative), or a distinct substring of a filename in ./data/
	    :param energy_min: the nominal minimum energy that XRS sees
	    :param energy_max: the nominal maximum energy that XRS sees
	    :param atomic_numbers: the set of elements whose lines we expect to see
	"""
	scan = find_scan_file(filename)
	spectrum = rotate_image(scan, energy_min, energy_max)
	spectrum = align_spectrum(spectrum, atomic_numbers)
	plot_and_save_spectrum(spectrum)


def find_scan_file(filename: str) -> h5py.Dataset:
	""" open the file if the filename is a complete path. otherwise, search for a file containing the given string in data/
	    :param filename: either the path to the file (absolute or relative), or a distinct substring of a filename in ./data/
	    :return: the opened h5py File object
	    :raise FileNotFoundError: if neither of the attempted methods produced a readable HDF5 file
	"""
	if os.path.isfile(filename):
		return h5py.File(filename)["PSL_per_px"]
	else:
		for found_filename in os.listdir("data/"):
			if found_filename.endswith(".h5") and filename in found_filename:
				return h5py.File(f"data/{found_filename}")["PSL_per_px"]
	raise FileNotFoundError(f"I could not find the file {filename!r}, nor could I find "
	                        f"any .h5 files matching {filename!r} in data/")


def rotate_image(scan: h5py.Dataset, energy_min: float, energy_max: float) -> SpatialSpectrum:
	""" take a raw scan file, infer how much it has been rotated and shifted, and correct it to
	    obtain an allined spacio-energo-image
	    :param scan: the scan data taken directly from the h5py File
	    :param energy_min: the nominal minimum energy that XRS sees
	    :param energy_max: the nominal maximum energy that XRS sees
	    :return: the PSL data resloved in space and energy
	"""
	# load in the image
	pixel_size = scan.attrs["pixelSizeX"]/1e4
	assert scan.attrs["pixelSizeX"] == scan.attrs["pixelSizeY"], "why aren’t the pixels square?"
	raw_image = np.transpose(scan[:, :])  # convert from (y,x) indexing to (i,j) indexing

	# convert it to an interpolator
	x = pixel_size*np.arange(-raw_image.shape[0]/2 + 1/2, raw_image.shape[0]/2)
	y = pixel_size*np.arange(-raw_image.shape[1]/2 + 1/2, raw_image.shape[1]/2)
	X, Y = np.meshgrid(x, y, indexing="ij")
	image_interpolator = interpolate.RegularGridInterpolator(
		(x, y), raw_image, bounds_error=False, fill_value=0)

	# rotate the image to maximize the maximum of the spectrally integrated image
	def negative_spacial_peak_given_angle(angle: float) -> float:
		rotated_X = X*cos(angle) + Y*sin(angle)
		rotated_Y = -X*sin(angle) + Y*cos(angle)
		rotated_image = image_interpolator((rotated_X, rotated_Y))
		spacial_image = np.sum(rotated_image, axis=0)
		return -np.max(spacial_image)
	angle = optimize.minimize_scalar(negative_spacial_peak_given_angle, [-.1, .1], method="golden").x
	rotated_x = X*cos(angle) + Y*sin(angle)
	rotated_y = -X*sin(angle) + Y*cos(angle)
	rotated_image = image_interpolator((rotated_x, rotated_y))

	# show the result of the rotation
	extent = (x[0] - pixel_size/2, x[-1] + pixel_size/2,
	          y[0] - pixel_size/2, y[-1] + pixel_size/2)
	fig, (ax_upper, ax_lower) = plt.subplots(2, 1, sharex="col", figsize=(8, 3.5))
	for ax, image, title in [(ax_upper, raw_image, "Raw image"),
	                         (ax_lower, rotated_image, "Rotated image")]:
		ax.set_title(title)
		ax.imshow(image.T, extent=extent, origin="lower", aspect="equal", cmap=CMAP["psl"])
		ax.yaxis.set_major_locator(LinearLocator(11))
		ax.yaxis.set_ticklabels([])
		ax.grid(color="w", linewidth=0.4, axis="y")

	# identify a background region and perform ramped background subtraction
	min_significant_psl = np.median(rotated_image)/100
	on_image_plate = erode(rotated_image > min_significant_psl, 10)  # type: ignore
	integrated_image = np.nanmean(np.where(on_image_plate, rotated_image, nan), axis=0)
	signal_cutoff = .9*np.nanquantile(integrated_image, .1) + .1*np.nanmax(integrated_image)
	out_of_signal_region = integrated_image < signal_cutoff
	in_background_region = on_image_plate & out_of_signal_region
	coefs = fit_2d_polynomial(
		X[in_background_region], Y[in_background_region], rotated_image[in_background_region])
	subtracted_image = rotated_image - (coefs[0]*X**2 + coefs[1]*X + coefs[2]*Y + coefs[3])
	subtracted_image[~on_image_plate] = 0  # don’t background-subtract the off-image-plate areas

	# show the result of the background subtraction
	fig, (ax_upper, ax_middle, ax_lower) = plt.subplots(3, 1, sharex="col", figsize=(8, 5))
	for ax, image, title in [(ax_upper, np.where(in_background_region, rotated_image, nan), "Background region"),
	                         (ax_middle, rotated_image, "Without background subtraction"),
	                         (ax_lower, subtracted_image, "With background subtraction (red is negative)")]:
		ax.set_title(title)
		vmax = np.max(rotated_image, where=in_background_region, initial=0)
		ax.imshow(image.T, extent=extent, origin="lower", aspect="equal",
		          cmap=CMAP["psl"], vmin=0, vmax=vmax)
		ax.imshow(np.where(image < 0, -image, nan).T, extent=extent, origin="lower", aspect="equal",
		          cmap=CMAP["blood"], vmin=0, vmax=vmax/2)
		if not np.any(np.isnan(image)):
			ax.contour(x, y, in_background_region.T, levels=[1/2], colors=["w"], linewidths=[0.4])
	plt.show()

	# crop it to the spacial data region
	integrated_image = subtracted_image.sum(axis=0)
	j_peak = np.argmax(integrated_image)
	cutoff = 0.1*integrated_image[j_peak]
	j_min = np.nonzero(integrated_image[:j_peak] < cutoff)[0][-1]  # these are inclusive
	j_max = np.nonzero(integrated_image[j_peak:] < cutoff)[0][0] + j_peak
	j_min, j_max = j_min - (j_max - j_min)//2, j_max + (j_max - j_min)//2  # abritrarily expand the bounds a bit
	j_min = max(0, j_min)
	j_max = min(j_max, integrated_image.size - 1)

	# crop it to the minimum energy edge and maximum energy edge
	integrated_spectrum = subtracted_image[:, j_min:j_max + 1].sum(axis=1)
	cutoff = np.quantile(integrated_spectrum, .7)/6
	i_min = np.nonzero(integrated_spectrum > cutoff)[0][0]  # these are inclusive
	i_max = np.nonzero(integrated_spectrum > cutoff)[0][-1]

	# show the results of the cropping
	fig, (ax_upper, ax_lower) = plt.subplots(2, 1, sharex="col", figsize=(8, 3.5),
	                                         gridspec_kw=dict(hspace=0))
	ax_upper.set_title("Data bounds")
	ax_upper.plot(x, subtracted_image[:, j_min:j_max + 1].sum(axis=1))
	ax_upper.grid()
	ax_upper.set_ylim(0, None)
	ax_upper.axhline(cutoff, color="k", linestyle="dashed", linewidth=1.)
	ax_upper.axvline(x[i_min] - pixel_size/2, color="k", linestyle="dashed", linewidth=1.)
	ax_upper.axvline(x[i_max] + pixel_size/2, color="k", linestyle="dashed", linewidth=1.)
	ax_upper.grid()
	ax_lower.imshow(image.T, extent=extent, origin="lower", aspect="auto", cmap=CMAP["psl"], vmin=0)
	ax_lower.axvline(x[i_min] - pixel_size/2, color="w", linestyle="dashed", linewidth=1.)
	ax_lower.axvline(x[i_max] + pixel_size/2, color="w", linestyle="dashed", linewidth=1.)
	ax_lower.axhline(y[j_min] - pixel_size/2, color="w", linestyle="dashed", linewidth=1.)
	ax_lower.axhline(y[j_max] + pixel_size/2, color="w", linestyle="dashed", linewidth=1.)
	ax_lower.set_ylabel("Spatial direction (cm)")
	ax_lower.set_xlabel("Energy direction (cm)")

	plt.show()

	cropped_image = subtracted_image[i_min:i_max + 1, :]
	return SpatialSpectrum(cropped_image, energy_min, energy_max, pixel_size)


def align_spectrum(spectrum: SpatialSpectrum, atomic_numbers: set[int]) -> SpatialSpectrum:
	""" get the user to help shift and scale the spectrum to make the observed features line up
	    with expected atomic line emission
	    :param spectrum: the PSL resolved in space and energy, which might be a bit misallined
	    :param atomic_numbers: the set of elements whose lines we expect to see
	    :return: the spectrum resolved in space and energy, now correctly allined
	"""
	return spectrum


def plot_and_save_spectrum(spectrum: SpatialSpectrum) -> None:
	""" display the results of the analysis
	    :param spectrum: the spectrum resolved in space and energy
	"""
	x = spectrum.pixel_size*np.arange(0.5, spectrum.num_x)
	energy = np.linspace(spectrum.energy_min, spectrum.energy_max, spectrum.num_energies)
	centroid = np.average(x, weights=np.sum(spectrum.values, axis=0))
	x = (x - centroid)*1e4

	plt.figure(figsize=(8, 4))
	plt.plot(x, np.sum(spectrum.values, axis=0))
	plt.xlabel("Position (μm)")
	plt.xlim(x[0], x[-1])
	plt.grid()
	plt.tight_layout()

	plt.figure(figsize=(8, 4))
	plt.plot(energy, np.sum(spectrum.values, axis=1)*spectrum.pixel_size)
	plt.xlabel("Photon energy (keV)")
	plt.xlim(energy[0], energy[-1])
	plt.grid()
	plt.tight_layout()

	plt.show()


def fit_2d_polynomial(x: NDArray[float], y: NDArray[float], z: NDArray[float]) -> tuple[float, float, float, float]:
	""" fit an extruded parabola to a cloud of points
	    :param x: the x coordinates at which z is known
	    :param y: the y coordinates at which z is known
	    :param z: the actual, noisy height values at each point where there is data
	    :return a, b, c, and d such that z ≈ a*x^2 + b*x + c*y + d
	"""
	matrix = np.array([[np.sum(x**4),   np.sum(x**3), np.sum(y*x**2), np.sum(x**2)],
	                   [np.sum(x**3),   np.sum(x**2), np.sum(y*x),    np.sum(x)   ],
	                   [np.sum(x**2*y), np.sum(x*y),  np.sum(y**2),   np.sum(y)   ],
	                   [np.sum(x**2),   np.sum(x),    np.sum(y),      z.size      ]])
	vector = np.array([np.sum(x**2*z), np.sum(x*z), np.sum(y*z), np.sum(z)])
	a, b, c, d = np.linalg.solve(matrix, vector)
	return a, b, c, d


def erode(array: NDArray[bool], distance: int) -> NDArray[bool]:
	for i in range(distance):
		array[1:, :] &= array[0:-1, :]
		array[:, 1:] &= array[:, 0:-1]
		array[0:-1, :] &= array[1:, :]
		array[:, 0:-1] &= array[:, 1:]
	return array


def main():
	parser = argparse.ArgumentParser(
		prog="analyze_xrs",
		description="Load an XRS image plate scan as a HDF5 file and, with the user’s help, extract "
		            "and plot an x-ray spectrum.")
	parser.add_argument("filename", type=str,
	                    help="either the path to the file (absolute or relative), or a distinct substring of a filename in ./data")
	parser.add_argument("minimum_energy", type=float,
	                    help="the nominal minimum energy that XRS sees")
	parser.add_argument("maximum_energy", type=float,
	                    help="the nominal maximum energy that XRS sees")
	for atomic_number, atomic_symbol in PERIODIC_TABLE.items():
		parser.add_argument(f"--{atomic_symbol}", action="store_true",
		                    help=f"whether to expect {atomic_symbol} lines")
	args = vars(parser.parse_args())
	atomic_numbers = {1}
	for atomic_number, atomic_symbol in PERIODIC_TABLE.items():
		if args[atomic_symbol]:
			atomic_numbers.add(atomic_number)
	analyze_xrs(args["filename"], args["minimum_energy"], args["maximum_energy"], atomic_numbers)


class SpatialSpectrum:
	def __init__(self, values: NDArray[float], energy_min: float, energy_max: float, pixel_size: float):
		""" a spacially- and temporally-resolved x-ray spectrum, including the axis information needed to interpret it.
		    :param values: the spectral data, indexed along energy on axis 0 and space on axis 1
		    :param energy_min: the photon energy (keV) corresponding to the leftmost value
		    :param energy_max: the photon energy (keV) corresponding to the rightmost value
		    :param pixel_size: the distance (μm) corresponding to a step between two adjacent rows
		"""
		self.values = values
		self.num_energies = values.shape[0]
		self.num_x = values.shape[1]
		self.energy_min = energy_min
		self.energy_max = energy_max
		self.pixel_size = pixel_size


if __name__ == "__main__":
	main()
