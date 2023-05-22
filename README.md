# xrs-analysis

A script for extracting x-ray spectra from XRS data.

To use it, drop your HDF5 file into the `data/` subdirectory and call
~~~bash
python analyze_xrs.py {FILENAME} {MINIMUM_ENERGY} {MAXIMUM_ENERGY} {BLAST_SHIELD_THICKNESS} [--filter_material={MATERIAL}] [--filter_thickness={THICKNESS}] [--detector_type={DETECTOR}] [--{ELEMENT}...]
~~~
The arguments are as follows:
1. `{FILENAME}` is a substring of the filename (such as the shot number).
   The script will search `data/` and analyze the first HDF5 filename it finds containing `{FILENAME}`.
   Alternatively, you can specify a full filepath, in which case you don’t have to put your scan files in `data/`.
   If you have multiple files to analyze, you can do multiple `{FILENAME}`s separated by commas.
2. `{MINIMUM_ENERGY}` is the lower bound of the energy range specified in the SRF, give in eV.
3. `{MAXIMUM_ENERGY}` is the upper bound of the energy range specified in the SRF, given in eV.
4. `{BLASH_SHIELD_THICKNESS}` is the thickness of the berylium blast shield given in thousandths of an inch.
5. `--filter_material` lets you specify the material of any additional filtering.
6. `--filter_thickness` lets you specify the thickness of any additional filtering in thousandths of an inch.
7. `--detector_type` lets you specify the type of detector used. Right now, only "BAS-MS", "BAS-SR", and "BAS-TR" are supported.
   If no detector type is given, it defaults to "BAS-MS".
8. For the purpose of aligning the spectrum, you may finally supply any number of atomic symbols as flags.
   These elements’ emission lines will appear on the screen where you adjust the spectral axis.
   For example, if you intend to align your spectrum to K-alpha emission from silicon and argon, you should add `--Si --Ar`.

The script will automatically rectify the image if it’s rotated and subtract out any background.
You will be prompted to identify the bounds of the data in space and energy.
Don’t worry about being precise with those, tho, as it will adjust the bounds itself later.
The most important part is the afformentioned alignment to elemental emission lines.
Skipping this and trusting XRS to view the energy range you asked for will typically produce an error of about 100–200 eV,<sup>1</sup>
which may be acceptable depending on your use-case.
The final spectrum will be saved as a plot and a HDF5 file next to the scan file.

<sup>1</sup>Personal communication from Dan Barnak.
