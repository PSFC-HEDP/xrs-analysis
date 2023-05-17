from __future__ import annotations

import argparse
import os.path
from math import cos, sin

import h5py
import matplotlib.pyplot as plt
import numpy as np
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
	pixel_size = scan.attrs["pixelSizeX"]
	assert scan.attrs["pixelSizeX"] == scan.attrs["pixelSizeY"], "why aren’t the pixels square?"
	raw_image = np.transpose(scan[:, :])  # convert from (y,x) indexing to (i,j) indexing

	# perform crude background subtraction
	subtracted_image = np.maximum(0, raw_image - np.quantile(raw_image[raw_image > 0], .5))

	# convert it to an interpolator
	x = pixel_size*np.arange(-subtracted_image.shape[0]/2 + 1/2, subtracted_image.shape[0]/2)
	y = pixel_size*np.arange(-subtracted_image.shape[1]/2 + 1/2, subtracted_image.shape[1]/2)
	X, Y = np.meshgrid(x, y, indexing="ij", sparse=True)
	image_interpolator = interpolate.RegularGridInterpolator(
		(x, y), subtracted_image, bounds_error=False, fill_value=0)

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

	# crop it to the edges that look like the minimum and maximum energy bounds
	integrated_spectrum = rotated_image.sum(axis=1)
	cutoff = np.quantile(integrated_spectrum, .7)/2
	i_min = np.nonzero(integrated_spectrum > cutoff)[0][0]
	i_max = np.nonzero(integrated_spectrum > cutoff)[0][-1]

	for title, image, include_lines in [("raw", raw_image, False), ("corrected", rotated_image, True)]:
		fig, ((ax_tl, ax_tr), (ax_bl, ax_br)) = plt.subplots(2, 2, sharex="col", sharey="row",
		                                                     gridspec_kw=dict(wspace=0, hspace=0))
		ax_tl.set_title(title)
		ax_tl.plot(x, image.sum(axis=1))
		ax_tl.grid()
		ax_tl.set_ylim(0, None)
		if include_lines:
			ax_tl.axhline(cutoff, color="k", linestyle="dashed")
			ax_tl.axvline(x[i_min] - pixel_size/2, color="k", linestyle="dashed")
			ax_tl.axvline(x[i_max] + pixel_size/2, color="k", linestyle="dashed")
		ax_br.plot(image.sum(axis=0), y)
		ax_br.grid()
		ax_br.set_xlim(0, None)
		ax_bl.imshow(image.T, origin="lower", cmap=CMAP["psl"],
		             extent=(x[0] - pixel_size/2, x[-1] + pixel_size/2,
		                     y[0] - pixel_size/2, y[-1] + pixel_size/2))
		ax_bl.set_xlabel("Spatial direction (μm)")
		ax_bl.set_xlabel("Energy direction (μm)")
	plt.show()

	cropped_image = rotated_image[i_min:i_max + 1, :]
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
	centroid = np.average(x, weights=np.sum(spectrum.values, axis=1))

	plt.figure()
	plt.plot(x - centroid, np.sum(spectrum.values, axis=1))
	plt.xlabel("Position (μm)")
	plt.grid()
	plt.tight_layout()

	plt.figure()
	plt.plot(energy, np.sum(spectrum.values, axis=0)*spectrum.pixel_size)
	plt.xlabel("Photon energy (keV)")
	plt.grid()
	plt.tight_layout()

	plt.show()


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
