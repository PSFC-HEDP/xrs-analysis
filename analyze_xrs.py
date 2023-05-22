from __future__ import annotations

import argparse
import os.path
import re
import sys
from math import cos, sin, nan

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LinearLocator
from matplotlib.widgets import Slider
from numpy.typing import NDArray
from scipy import interpolate, optimize

from cmap import CMAP
from image_plate import Filter, xray_sensitivity

PERIODIC_TABLE = {
	1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
	10: "Ne", 13: "Al", 14: "Si", 18: "Ar", 36: "Kr", 54: "Xe",
}


def analyze_xrs(filename: str, energy_min: float, energy_max: float,
                filter_stack: list[Filter], detector_type: str, atomic_numbers: set[int]):
	""" load an XRS image plate scan as a HDF5 file and, with the user’s help, extract and plot an x-ray spectrum
	    :param filename: either the path to the file (absolute or relative), or a distinct substring of a filename in ./data/
	    :param energy_min: the nominal minimum energy that XRS sees (keV)
	    :param energy_max: the nominal maximum energy that XRS sees (keV)
	    :param filter_stack: the list of layers of material between the source and the image plate
	    :param detector_type: the type of FujiFilm BAS image plate (one of 'MS', 'SR', or 'TR')
	    :param atomic_numbers: the set of elements whose lines we expect to see
	"""
	filename = find_scan_file(filename)
	scan = h5py.File(filename)["PSL_per_px"]
	distribution = rotate_image(scan, energy_min, energy_max)
	distribution = align_data(distribution, filter_stack, detector_type, atomic_numbers)
	plot_and_save_spectrum(distribution, filename, filter_stack, detector_type)


def find_scan_file(filename: str) -> str:
	""" open the file if the filename is a complete path. otherwise, search for a file containing the given string in data/
	    :param filename: either the path to the file (absolute or relative), or a distinct substring of a filename in ./data/
	    :return: the opened h5py File object
	    :raise FileNotFoundError: if neither of the attempted methods produced a readable HDF5 file
	"""
	if os.path.isfile(filename):
		return filename
	else:
		for found_filename in os.listdir("data/"):
			if found_filename.endswith(".h5") and filename in found_filename:
				return f"data/{found_filename}"
	raise FileNotFoundError(f"I could not find the file {filename!r}, nor could I find "
	                        f"any .h5 files matching {filename!r} in data/")


def rotate_image(scan: h5py.Dataset, energy_min: float, energy_max: float) -> SpatialEnergyDistribution:
	""" take a raw scan file, infer how much it has been rotated and shifted, and correct it to
	    obtain an allined spacio-energo-image
	    :param scan: the scan data taken directly from the h5py File
	    :param energy_min: the nominal minimum energy that XRS sees (keV)
	    :param energy_max: the nominal maximum energy that XRS sees (keV)
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
	integrated_distribution = subtracted_image[:, j_min:j_max + 1].sum(axis=1)
	cutoff = np.quantile(integrated_distribution, .7)/6
	i_min = np.nonzero(integrated_distribution > cutoff)[0][0]  # these are inclusive
	i_max = np.nonzero(integrated_distribution > cutoff)[0][-1]

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

	cropped_image = subtracted_image[i_max:i_min - 1:-1, j_min:j_max + 1]
	return SpatialEnergyDistribution(cropped_image, energy_min, energy_max, pixel_size)


def align_data(distribution: SpatialEnergyDistribution, filter_stack: list[Filter], detector_type: str, atomic_numbers: set[int]) -> SpatialEnergyDistribution:
	""" get the user to help shift and scale the spectrum to make the observed features line up
	    with expected atomic line emission
	    :param distribution: the PSL resolved in space and energy, which might be a bit misallined
	    :param filter_stack: the list of layers of material between the source and the image plate
	    :param detector_type: the type of FujiFilm BAS image plate (one of 'MS', 'SR', or 'TR')
	    :param atomic_numbers: the set of elements whose lines we expect to see
	    :return: the PSL resolved in space and energy, now correctly allined
	"""
	integrated_psl = distribution.values.sum(axis=1)
	# set up the figure
	fig, ax = plt.subplots(figsize=(8, 4.5))
	curve, = ax.plot([], [], color="k")
	ax.grid()
	ax.set_xlabel("Energy (keV)")
	ax.set_title("Adjust to match the lines, then close the window.")
	ax.set_xlim(distribution.energy_min, distribution.energy_max)

	# # include the places where we expect atomic lines
	# for color_index, z in enumerate(atomic_numbers):
	# 	for n1 in range(2, 6):
	# 		for n2 in range(1, n1):
	# 			hv = .0136*(z - 1)**2 * (1/n2**2 - 1/n1**2)
	# 			# if .8 < hv < 24:
	# 			print(f"does {PERIODIC_TABLE[z]} have a line at {hv*1000} eV?")  # http://hyperphysics.phy-astr.gsu.edu/hbase/Tables/kxray.html or https://xdb.lbl.gov/Section1/Table_1-2.pdf
	# 			ax.axhline(hv, color=f"C{color_index}", linewidth=1.0, linestyle="dashed")

	# add sliders
	fig.subplots_adjust(.08, .30, .97, .93)
	ax_shift = fig.add_axes([.08, .12, .89, .04])
	ax_scale = fig.add_axes([.08, .04, .89, .04])
	shift_slider = Slider(ax=ax_shift, label="Shift", valmin=-.5, valmax=+.5, valinit=0)
	scale_slider = Slider(ax=ax_scale, label="Scale", valmin=0.9, valmax=1.1, valinit=1)

	# define how to respond to the sliders
	def get_energy_bounds_from_sliders() -> tuple[float, float]:
		energy_center = (distribution.energy_min + distribution.energy_max)/2 + shift_slider.val
		energy_range = (distribution.energy_max - distribution.energy_min)*scale_slider.val
		energy_min = energy_center - energy_range/2
		energy_max = energy_center + energy_range/2
		return energy_min, energy_max

	def update_plot(*_):
		energies = np.linspace(*get_energy_bounds_from_sliders(), distribution.num_energies)
		curve.set_xdata(energies)
		curve.set_ydata(integrated_psl/xray_sensitivity(energies, filter_stack, detector_type))
		ax.set_ylim(0, 1.1*np.max(curve.get_ydata()))
		fig.canvas.draw_idle()

	scale_slider.on_changed(update_plot)
	shift_slider.on_changed(update_plot)

	# wait for the user to finish
	update_plot()
	plt.show()

	# update the PSL distribution according to the new energy bounds
	energy_min, energy_max = get_energy_bounds_from_sliders()
	return SpatialEnergyDistribution(distribution.values, energy_min, energy_max, distribution.pixel_size)


def plot_and_save_spectrum(distribution: SpatialEnergyDistribution, filename: str, filter_stack: list[Filter], detector_type: str) -> None:
	""" display the results of the analysis
	    :param distribution: the PSL resolved in space and energy
	    :param filename: the filename of the scan file, which will be used to create the output filename
	    :param filter_stack: the list of layers of material between the source and the image plate
	    :param detector_type: the type of FujiFilm BAS image plate (one of 'MS', 'SR', or 'TR')
	"""
	filename, _ = os.path.splitext(filename.replace("-[phosphor]", ""))

	x = distribution.pixel_size*np.arange(0.5, distribution.num_x)
	energy = np.linspace(distribution.energy_min, distribution.energy_max, distribution.num_energies)
	spectrum = distribution.values/xray_sensitivity(energy, filter_stack, detector_type)[:, np.newaxis]
	centroid = np.average(x, weights=np.sum(spectrum, axis=0))
	x = (x - centroid)*1e4

	with h5py.File(f"{filename}-analyzed.h5", "w") as f:
		f["photon energy"] = energy
		f["photon energy"].attrs["units"] = "keV"
		f["photon energy"].make_scale()
		f["position"] = x
		f["position"].attrs["units"] = "μm"
		f["position"].make_scale()
		f["energy density"] = spectrum
		f["energy density"].attrs["units"] = "unclear"
		f["energy density"].dims[0].attach_scale(f["photon energy"])
		f["energy density"].dims[1].attach_scale(f["position"])

	plt.figure(figsize=(8, 4), facecolor="none")
	plt.plot(x, np.sum(spectrum, axis=0))
	plt.xlabel("Position (μm)")
	plt.xlim(x[0], x[-1])
	plt.ylim(0, None)
	plt.grid()
	plt.tight_layout()
	plt.savefig(f"{filename}-image.png", dpi=300)

	plt.figure(figsize=(8, 4), facecolor="none")
	plt.plot(energy, np.sum(spectrum, axis=1)*distribution.pixel_size)
	plt.xlabel("Photon energy (keV)")
	plt.xlim(energy[0], energy[-1])
	plt.ylim(0, None)
	plt.grid()
	plt.tight_layout()
	plt.savefig(f"{filename}-spectrum.png", dpi=300)

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
	                    help="comma-separated list of either filepaths (absolute or relative) or distinct filename substrings (if the files are in ./data)")
	parser.add_argument("minimum_energy", type=float,
	                    help="the nominal minimum energy that XRS sees (keV)")
	parser.add_argument("maximum_energy", type=float,
	                    help="the nominal maximum energy that XRS sees (keV)")
	parser.add_argument("blast_shield_thickness", type=float,
	                    help="the thickness of the Be blast shield (mils)")
	parser.add_argument("--filter_material", type=str, required=False, default="Ta",
	                    help="the additional filtering material (e.g. 'Al' or 'Ta')")
	parser.add_argument("--filter_thickness", type=float, required=False, default=0.,
	                    help="the thickness of the additional filtering (mils)")
	parser.add_argument("--detector_type", type=str, required=False, default="BAS_MS",
	                    help="the type of FujiFilm BAS image plate (one of 'BAS_MS', 'BAS_SR', or 'BAS_TR')")
	for atomic_number, atomic_symbol in PERIODIC_TABLE.items():
		parser.add_argument(f"--{atomic_symbol}", action="store_true",
		                    help=f"whether to expect {atomic_symbol} lines")
	args = vars(parser.parse_args())
	atomic_numbers = set()
	for atomic_number, atomic_symbol in PERIODIC_TABLE.items():
		if args[atomic_symbol]:
			atomic_numbers.add(atomic_number)
	filter_stack = [(args["blast_shield_thickness"]*25.4, "Be"),
	                (args["filter_thickness"]*25.4, args["filter_material"])]
	detector_type_parsing = re.fullmatch(
		r"(BAS[-_ ]?)?(MS|SR|TR)([-_ ]?(IP|image ?plate))?", args["detector_type"], re.IGNORECASE)
	if detector_type_parsing is None:
		print("The detector type must be one of MS, SR, or TR.", file=sys.stderr)
	else:
		for filename in args["filename"].split(","):
			print(f"Analyzing {filename}...")
			analyze_xrs(filename, args["minimum_energy"], args["maximum_energy"],
			            filter_stack, detector_type_parsing.group(2), atomic_numbers)
		print("Done!")


class SpatialEnergyDistribution:
	def __init__(self, values: NDArray[float], energy_min: float, energy_max: float, pixel_size: float):
		""" a function of position (in 1D) and energy, bundled with the axis information needed to interpret it.
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
