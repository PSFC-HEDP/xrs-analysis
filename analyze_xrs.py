import argparse
import os.path
import re
import sys
from math import cos, sin, nan, inf

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseEvent, MouseButton
from matplotlib.ticker import LinearLocator
from matplotlib.widgets import Slider
from numpy.typing import NDArray
from scipy import interpolate, optimize

from cmap import CMAP
from image_plate import Filter, xray_sensitivity
from util import find_scan_file, SpatialEnergyDistribution

plt.rcParams.update({"font.size": 14})

SPECTRAL_LINES = {
	"Ne": [.84861],
	"Al": [1.486295, 1.486708, 1.55745],
	"Si": [1.739394, 1.739985, 1.83594],
	"Ar": [2.955566, 2.957682, 3.1905],
	"Kr": [12.595424, 12.648002, 14.112, 1.5860, 1.6366],
	"Zr": [15.6909, 15.7751, 17.6678, 2.0399, 2.04236, 2.2194, 2.1244, 2.3027],
	"Xe": [4.1099],
}  # source: https://xdb.lbl.gov/Section1/Table_1-2.pdf


def analyze_xrs(filename: str, energy_min: float, energy_max: float,
                filter_stack: list[Filter], detector_type: str, elements: set[str]):
	""" load an XRS image plate scan as a HDF5 file and, with the user’s help, extract and plot an x-ray spectrum
	    :param filename: either the path to the file (absolute or relative), or a distinct substring of a filename in ./data/
	    :param energy_min: the nominal minimum energy that XRS sees (keV)
	    :param energy_max: the nominal maximum energy that XRS sees (keV)
	    :param filter_stack: the list of layers of material between the source and the image plate
	    :param detector_type: the type of FujiFilm BAS image plate (one of 'MS', 'SR', or 'TR')
	    :param elements: the symbols of the elements whose lines we expect to see
	"""
	filename = find_scan_file(filename, forbidden={"-analyzed"})
	try:
		scan = h5py.File(filename)["PSL_per_px"]
	except KeyError:
		print(f"{filename!r} does not appear to be an image plate scan.")
	else:
		distribution = rotate_image(scan, energy_min, energy_max)
		distribution = align_data(distribution, filter_stack, detector_type, elements)
		plot_and_save_spectrum(distribution, filename, filter_stack, detector_type)


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
	plt.show()

	# identify regions off the image plate
	min_significant_psl = np.median(rotated_image)*.01
	on_image_plate = erode(rotated_image > min_significant_psl, 10)  # type: ignore
	rotated_image[~on_image_plate] = nan

	# ask the user to identify the bounds
	integrated_image = np.nanmean(rotated_image, axis=0)
	y_min, y_max = ask_user_for_bounds(y, integrated_image)
	y_min, y_max = 2*y_min - y_max, 2*y_max - y_min  # expand it by a factor of 3
	within_y_bounds = (y >= y_min) & (y <= y_max)
	y = y[within_y_bounds]
	rotated_image = rotated_image[:, within_y_bounds]

	j_peak = np.argmax(np.nansum(rotated_image, axis=0))
	central_spectrum = rotated_image[:, j_peak]
	x_min, x_max = ask_user_for_bounds(x, central_spectrum)
	x_min, x_max = x_min - .500, x_max + .500  # expand it by 1 mm
	within_x_bounds = (x >= x_min) & (x <= x_max)
	x = x[within_x_bounds]
	rotated_image = rotated_image[within_x_bounds, :]

	extent = (x[0] - pixel_size/2, x[-1] + pixel_size/2,
	          y[0] - pixel_size/2, y[-1] + pixel_size/2)

	# perform background subtraction
	integrated_image = np.nanmean(rotated_image, axis=0)
	signal_cutoff = .9*np.nanquantile(integrated_image, .67) + .1*np.nanmax(integrated_image)
	out_of_signal_region = integrated_image < signal_cutoff
	in_background_region = np.isfinite(rotated_image) & out_of_signal_region
	X, Y = np.meshgrid(x, y, indexing="ij")
	coefs = fit_2d_polynomial(
		X[in_background_region], Y[in_background_region], rotated_image[in_background_region])
	subtracted_image = rotated_image - (coefs[0]*X**2 + coefs[1]*X + coefs[2]*Y + coefs[3])
	subtracted_image[np.isnan(subtracted_image)] = 0  # un-nanify these now that there’s a meaningful fill value

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
	plt.show()

	# crop it to the spacial data region
	integrated_image = subtracted_image.sum(axis=0)
	j_peak = np.argmax(integrated_image)
	cutoff = 0.1*integrated_image[j_peak]
	j_min = np.nonzero(integrated_image[:j_peak] < cutoff)[0][-1]  # these are inclusive
	j_max = np.nonzero(integrated_image[j_peak:] < cutoff)[0][0] + j_peak
	j_min, j_max = j_min - (j_max - j_min)//2, j_max + (j_max - j_min)//2  # abritrarily expand the bounds a bit
	j_min = max(0, j_min)  # it’s possible for the j bounds to stay the same
	j_max = min(j_max, integrated_image.size - 1)

	# crop it to the minimum energy edge and maximum energy edge
	integrated_distribution = subtracted_image[:, j_min:j_max + 1].sum(axis=1)
	cutoff = 0.1*np.quantile(integrated_distribution, .7)
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
	ax_lower.imshow(subtracted_image.T, extent=extent, origin="lower", aspect="auto", cmap=CMAP["psl"], vmin=0)
	ax_lower.axvline(x[i_min] - pixel_size/2, color="w", linestyle="dashed", linewidth=1.)
	ax_lower.axvline(x[i_max] + pixel_size/2, color="w", linestyle="dashed", linewidth=1.)
	ax_lower.axhline(y[j_min] - pixel_size/2, color="w", linestyle="dashed", linewidth=1.)
	ax_lower.axhline(y[j_max] + pixel_size/2, color="w", linestyle="dashed", linewidth=1.)
	ax_lower.set_ylabel("Spatial direction (cm)")
	ax_lower.set_xlabel("Energy direction (cm)")

	plt.show()

	i_min, i_max = i_min + int(.1/pixel_size), i_max - int(.1/pixel_size)  # contract the bounds a bit
	final_image = subtracted_image[i_max:i_min - 1:-1, j_min:j_max + 1]
	return SpatialEnergyDistribution(final_image, energy_min, energy_max, pixel_size)


def ask_user_for_bounds(x: NDArray[float], y: NDArray[float]) -> tuple[float, float]:
	""" prompt the user to click on a plot to choose lower and upper limits
	    :param x: the x coordinates to plot
	    :param y: the y coordinates to plot
	    :return: the minimum x and the maximum x that the user selected
	"""
	fig = plt.figure("selection", figsize=(8, 4))
	plt.locator_params(steps=[1, 2, 5, 10])
	plt.grid()
	plt.plot(x, y)
	plt.xlim(np.min(x, where=np.isfinite(y), initial=inf),
	         np.max(x, where=np.isfinite(y), initial=-inf))
	plt.title("click on the lower and upper bounds of this signal, then close this plot")
	plt.tight_layout()

	lines = [plt.plot([0, 0], [np.nanmin(y), np.nanmax(y)], "k--")[0] for _ in range(2)]
	for line in lines:
		line.set_visible(False)

	selected_points: list[float] = []

	def on_click(event: MouseEvent):
		nonlocal selected_points
		# whenever the user clicks...
		if type(event) is MouseEvent and event.xdata is not None:
			# if it's a right-click, delete a point
			if event.button == MouseButton.RIGHT:
				if len(selected_points) > 0:
					selected_points.pop()
			# otherwise, save a new point
			else:
				selected_points.append(event.xdata)
			# then update the plot
			for i in range(len(lines)):
				if i < len(selected_points):
					x = selected_points[-i - 1]
					lines[i].set_xdata([x, x])
					lines[i].set_visible(True)
				else:
					lines[i].set_visible(False)
	fig.canvas.mpl_connect('button_press_event', on_click)

	while plt.fignum_exists("selection"):
		plt.pause(0.1)
	if len(selected_points) < 2:
		raise ValueError("you didn't specify both limits.")

	# once the user is done, arrange the results
	return min(selected_points[-2:]), max(selected_points[-2:])


def align_data(distribution: SpatialEnergyDistribution, filter_stack: list[Filter], detector_type: str,
               elements: set[str]) -> SpatialEnergyDistribution:
	""" get the user to help shift and scale the spectrum to make the observed features line up
	    with expected atomic line emission
	    :param distribution: the PSL resolved in space and energy, which might be a bit misallined
	    :param filter_stack: the list of layers of material between the source and the image plate
	    :param detector_type: the type of FujiFilm BAS image plate (one of 'MS', 'SR', or 'TR')
	    :param elements: the symbols of the elements whose lines we expect to see
	    :return: the PSL resolved in space and energy, now correctly allined
	"""
	integrated_psl = distribution.values.sum(axis=1)

	# set up the figure
	fig, ax = plt.subplots(figsize=(8, 4.5))
	curve, = ax.plot([], [], color="k")
	ax.grid()
	ax.locator_params(steps=[1, 2, 5, 10])
	ax.set_xlabel("Energy (keV)")
	ax.set_title("Adjust to match the lines, then close the window.")
	ax.set_xlim(distribution.energy_min, distribution.energy_max)
	ax.axvline(distribution.energy_min, color="k", linewidth=1.0, linestyle="dashed")
	ax.axvline(distribution.energy_max, color="k", linewidth=1.0, linestyle="dashed")

	# include the places where we expect atomic lines
	for color_index, symbol in enumerate(elements):
		for hν in SPECTRAL_LINES[symbol]:
			ax.axvline(hν, color=f"C{color_index}", linewidth=1.0, linestyle="dashed")
		ax.text(SPECTRAL_LINES[symbol][0], 0, symbol, color=f"C{color_index}", horizontalalignment="right")

	# add sliders
	fig.subplots_adjust(.08, .30, .97, .93)
	ax_shift = fig.add_axes([.08, .12, .85, .04])
	ax_scale = fig.add_axes([.08, .04, .85, .04])
	shift_slider = Slider(ax=ax_shift, label="Shift", valmin=-.5, valmax=+.5, valinit=0)
	scale_slider = Slider(ax=ax_scale, label="Scale", valmin=0.8, valmax=1.2, valinit=1)

	# define how to respond to the sliders
	def get_energy_bounds_from_sliders() -> tuple[float, float]:
		energy_center = (distribution.energy_min + distribution.energy_max)/2 + shift_slider.val
		energy_range = (distribution.energy_max - distribution.energy_min)*scale_slider.val
		energy_min = energy_center - energy_range/2
		energy_max = energy_center + energy_range/2
		return energy_min, energy_max

	def update_plot(*_):
		energy_min, energy_max = get_energy_bounds_from_sliders()
		energies = np.linspace(energy_min, energy_max, distribution.num_energies)
		curve.set_xdata(energies)
		curve.set_ydata(integrated_psl/xray_sensitivity(energies, filter_stack, detector_type))
		ax.set_xlim(energy_min, energy_max)
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
	plt.title(os.path.basename(filename))
	plt.xlim(x[0], x[-1])
	plt.ylim(0, None)
	plt.grid()
	plt.tight_layout()
	plt.savefig(f"{filename}-image.png", dpi=300)

	plt.figure(figsize=(8, 4), facecolor="none")
	plt.plot(energy, np.sum(spectrum, axis=1)*distribution.pixel_size)
	plt.xlabel("Photon energy (keV)")
	plt.title(os.path.basename(filename))
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
	                    help="the nominal minimum energy that XRS sees (eV)")
	parser.add_argument("maximum_energy", type=float,
	                    help="the nominal maximum energy that XRS sees (eV)")
	parser.add_argument("blast_shield_thickness", type=float,
	                    help="the thickness of the Be blast shield (mils)")
	parser.add_argument("--filter_material", type=str, required=False, default="Ta",
	                    help="the additional filtering material (e.g. 'Al' or 'Ta')")
	parser.add_argument("--filter_thickness", type=float, required=False, default=0.,
	                    help="the thickness of the additional filtering (mils)")
	parser.add_argument("--detector_type", type=str, required=False, default="BAS_MS",
	                    help="the type of FujiFilm BAS image plate (one of 'BAS_MS', 'BAS_SR', or 'BAS_TR')")
	for symbol in SPECTRAL_LINES.keys():
		parser.add_argument(f"--{symbol}", action="store_true",
		                    help=f"whether to expect {symbol} lines")
	args = vars(parser.parse_args())
	elements = set()
	for symbol in SPECTRAL_LINES.keys():
		if args[symbol]:
			elements.add(symbol)
	filter_stack = [(args["blast_shield_thickness"]*25.4, "Be"),
	                (args["filter_thickness"]*25.4, args["filter_material"])]
	detector_type_parsing = re.fullmatch(
		r"(BAS[-_ ]?)?(MS|SR|TR)([-_ ]?(IP|image ?plate))?", args["detector_type"], re.IGNORECASE)
	if detector_type_parsing is None:
		print("The detector type must be one of MS, SR, or TR.", file=sys.stderr)
	else:
		for filename in args["filename"].split(","):
			print(f"Analyzing {filename}...")
			analyze_xrs(filename, args["minimum_energy"]/1e3, args["maximum_energy"]/1e3,
			            filter_stack, detector_type_parsing.group(2), elements)
		print("Done!")


if __name__ == "__main__":
	main()
