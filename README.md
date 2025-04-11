# Codes for "Modeling glioma-induced impairments on the glymphatic system"
## Usage

Once installed, the code can be used to run the scripts corresponding to the different test cases described in the paper. 
Just run the following command in a terminal (with the conda environment activated):
```
python3 reference_clearance.py
```

To obtain the figures of the paper, you should run the 4 test cases. Then, you can use the different scripts in the folder "figure_scripts" to generate the figures. 
For the pressure and concentration fields, you have to change the variable "exp_name" in the scripts "create_slices_pressure.py" and "slices_multicomp_concentration.py" for the 4 test cases. 

## Installation

The easiest way to install and use the code is to create a conda environment using the "environment.yml" file that lists all the dependencies. 
