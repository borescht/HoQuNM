# HoQuNM
Hospital queuing network modelling.
An implementation of a DES for a mathematical queueing network model for hospital ward modelling.

## Setup
The following is necessary for using the code:
1. Install Python 3.7. 

1. Go to the project folder.

1. Create a [pip virtual env](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/). 
1. Activate the virtual environment. 
1. If you run on a windows machine install `py-make` via `pip install py-make`. In the follwing always type `pymake` instead of `make`.
1. Run `make install`.

1. For saving trees from CART analysis, `graphviz` has to be installed
on your machine and added to system PATH.

Run the functionalities inside your virtual environment.


If you want to push code run `make format` and `make lint` before and resolve possible issues.

## Available command line tools
All command line tools can be accessed over `hoqunm-cli`. For help, see `hoqunm-cli --help`.
 - `hoqunm-cli preprocess-data`: Preprocess raw data (as given by the hospital). This tool is very specific to the 
 partner hospital.
 - `hoqunm-cli analyse-data`: Analysis of preprocessed data. This tool is very specific to the 
 partner hospital.
 - `hoqunm-cli build-model`: Build the model from preprocessed data. This tool is very specific to the 
 partner hospital.
 - `hoqunm-cli analyse-model-variants`: For computed parameters from given data, analyse the different models (1,2,3).
 - `hoqunm-cli assess-capacities`: Assess capacities for a given model.
 - `hoqunm-cli simulate-optimum`: Simulate the model for different capacity combinations and compute the optimum. The simulation results
 are saved for future use.
 - `hoqunm-cli compute-optimum`: Compute the optimum from already simulated results by `simulate optimum`. This might be useful 
 if different optimisation problems (with different parameters) are to be assessed, thus simulations will
 not have to be reeated each time.

