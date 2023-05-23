import os
from typing import Iterable

from numpy.typing import NDArray


def find_scan_file(filename: str, required: Iterable[str] = (), forbidden: Iterable[str] = ()) -> str:
	""" open the file if the filename is a complete path. otherwise, search for a file containing the given string in data/
	    :param filename: either the path to the file (absolute or relative), or a distinct substring of a filename in ./data/
	    :param required: any substrings we’re expecting to find in the filename, in addition to the filename parameter
	    :param forbidden: any substrings we must not see in the filename
	    :return: the opened h5py File object
	    :raise FileNotFoundError: if neither of the attempted methods produced a readable HDF5 file
	"""
	if os.path.isfile(filename):
		return filename
	else:
		for found_filename in os.listdir("data/"):
			if found_filename.endswith(".h5") and filename in found_filename:
				if all(key in found_filename for key in required):
					if not any(key in found_filename for key in forbidden):
						return f"data/{found_filename}"
	raise FileNotFoundError(f"I could not find the file {filename!r}, nor could I find "
	                        f"any .h5 files matching {filename!r} in data/")


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
