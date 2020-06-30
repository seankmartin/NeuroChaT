=============================
NeuroChaT Version Information
=============================

Version 1.1.2
=============
Many significant changes made to address reviewer concerns and suggested improvements.

Changes
-------
- Large updates to docs to adhere to pep8 documentation and fix stray documentation errors.
- Project wide autopep8 formatting with one level of aggression flagged.
- UI rescaling to fit text on Windows and Linux without flowing over the borders and requiring horizontal scrolling.
- Added Sphinx documentation to migrate from pdoc3.
- Added Read the Docs configuration.
- Added printing support for main NeuroChaT data types.
- Version number is now included in the UI.
- Control object reuses loaded files in the main loop if they were previously loaded.
- Set up UI to have a browse button for each of the main loaded file types in NeuroChaT. Namely, spatial files, LFP files, and position files. Also updated the backend code so that analysis could be performed without all file types being present. For example, LFP only needs to be loaded to perform spectrum analysis. The program warns if a needed file is not present.
- Results file name nc_results.xslx has been changed to name based off the file used if possible, and timestamped. If this fails, it reverts back to the old default.
- Added a spike interface sorting loader, which supports any Sorting Extractor, and can save this to NWB format for later use.
- Added option to UI to clear the selected files.
- Added button to UI to append the currently selected files to an excel spreadsheet to facilitate batch analysis.
- Added further options to the UI the control the lfp spectral analysis and place field analysis.
- Added support for colorblind friendly perceptually uniform colormaps from the NeuroChaT UI. These have been changed to the default output type.
- Can control number of levels in contour plots from the UI. For example, on place cell analysis.
- Added a legend to polar plots of head direction firing.
- Excel results file in UI is automatically named based on the input files.

Bugs
----
- All results methods now return the same number of keys regardless of the output. This was primarily a problem in grid cell analysis.
- Fixed parsing units on spike selection in UI sometimes failing unnecessarily.
- Fix bug with loading time information from Neuralynx incorrectly on certain formats.
- Reading an excel file for a list of files can now support blank cells in the spreadsheet.
- Fixed rare bug with ISI being a float after ceil operation instead of int.
- Fixed bug with outputting the wrong statistics for the head direction firing rate analysis split into Clockwise and CounterClockwise firing.
- Fixed bug with angular velocity information not being correctly loaded after the spatial file changes if angular velocity information was previously loaded.

Version 1.1.1a0
===============
Support loading blank eeg files in Axona and also log crashes from the UI.

Changes
-------
- Support loading blank eeg files in Axona.
- Crashes in the UI get logged to a file which can be reported to us.
- Added peak firing rate output from place.

Bugs
----
- Fixed crash on file select if hit cancel or done.

Version 1.1.1
=============
Various small updates to NeuroChaT from usage in our lab.

Changes
-------
- Improved spatial cell summary plot. 
- Added a .stm loader for Axona stimulation files to store in NEvent.
- Support loading blank Axona .eeg files.
- Added simple artefact detection based on standard deviation of LFP signal. More complex methods should be preferred for handling artefacts, such as ICA decomposition.
- Update documentation.

Bugs
----
- Fixed problem with smoothing circular information around 0 degrees.
- Fixed waveform slope calculation for one spike.
- Fixed issues with filenames having extra dots.

Version 1.1.0
=============
Updates to NeuroChaT before paper submission to Wellcome Trust.

Changes
-------
- Improved spatial cell summary plot. 
- Added unit parser to UI to only show existing units.
- Added scripts to reproduce files in paper.
- Update documentation.

Bugs
----
- Fixed loading h5 from UI.
- Fix rare error in recursive analysis of spatial cells.

Version 1.0
===========
Initial release of NeuroChaT.
The main functionality exists at this point.
This has numerous small changes from the version in Md. Nurul Islam's thesis, but most of these are around installing NeuroChaT, building documentation for NeuroChaT, README information, examples of usage etc.
The backend code was still very similar at this point to the first version of NeuroChaT by Md. Nurul Islam.