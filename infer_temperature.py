import argparse
import os
from math import sqrt

import h5py
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy import optimize

from util import find_scan_file

plt.rcParams.update({"font.size": 14})


def infer_temperature(filename: str):
	""" Load an analyzed XRS spectrum and infer a spacially-integrated electron temperature from it
	    :param filename: either the path to the file (absolute or relative), or a distinct substring of a filename in ./data/
	"""
	# find the file
	full_filename = find_scan_file(filename, required={"-analyzed"})
	data = h5py.File(full_filename)

	# extract the energy and integrated spectrum
	if "energy density" not in data:
		print(f"{full_filename!r} does not appear to be an analyzed spectrum.")
		return
	energy = data["photon energy"][:]
	spectrum = np.sum(data["energy density"][:, :], axis=1)

	# fit a brems spectrum
	(temperature, yeeld), cov = optimize.curve_fit(
		brems_spectrum,
		energy, spectrum, sigma=spectrum*.10,
		p0=[3., np.sum(spectrum*np.gradient(energy))])
	temperature_error = sqrt(1.96**2*cov[0, 0] + .05**2)  # temperature error should never be less than 5%

	# print and plot the results
	print(f"Te = {temperature:.3f} ± {temperature_error:.3f} keV")

	plt.figure(figsize=(8, 4), facecolor="none")
	plt.plot(energy, spectrum, "C0-")
	plt.plot(energy, brems_spectrum(energy, temperature, yeeld), "C1--")
	plt.grid()
	plt.yscale("log")
	plt.xlim(energy[0], energy[-1])
	plt.xlabel("Photon energy (keV)")
	plt.title(f"{filename} best fit; $T_\\mathrm{{e}}$ = {temperature:.3f} ± {temperature_error:.3f} keV")
	plt.tight_layout()
	plt.savefig(os.path.splitext(full_filename)[0] + ".png", dpi=300)
	plt.show()


def brems_spectrum(E: NDArray[float], temperature: float, yeeld: float) -> NDArray[float]:
	""" a function to evaluate the exponential approximation of the bremsstrahlung spectrum """
	return yeeld/temperature*np.exp(-E/temperature)


def main():
	parser = argparse.ArgumentParser(
		prog="python infer_temperature.py",
		description="Load an analyzed XRS spectrum and infer a spacially-integrated electron temperature from it")
	parser.add_argument("filename", type=str,
	                    help="comma-separated list of either filepaths (absolute or relative) or distinct filename substrings (if the files are in ./data)")
	args = vars(parser.parse_args())
	for filename in args["filename"].split(","):
		print(f"Analyzing {filename}...")
		infer_temperature(filename)
	print("Done!")


if __name__ == "__main__":
	main()
