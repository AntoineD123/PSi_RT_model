# PSi_RT_model (written as a Python package)
Tested with Python 3.12
Dependancies: pathlib, os, numpy, matplotlib.pyplot, scipy.optimize

Project used to compute the reflectance of a porous silicon multilayered stack, measured perpendicularly to the surface (1D problem). 
Classes:
    - PSiRefractiveIndexMethod: defines the method used to compute the refractive index of a layer with homogeneous properties (porosity, type of material). 
    - PSiOpticalStack: defines a multilayered 1D stack of several layers, produced from a single wafer. The properties (characterization) of the wafer must be computed in advanced and passed to the object creating the stack. 
    - PSiReflectanceSimulator: computes or extracts the values of reflectance and transmittance. Can also be used to plot and save the data as a picture (saved in the '/results' folder).


