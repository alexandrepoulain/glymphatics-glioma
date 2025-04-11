#!/usr/bin/env python
# coding: utf-8
"""
Multicompartment in the human brain.
This script is used to simulate the 3-compartment case.
This is the non-vasogenic case.

The equations are
1) equations for the fluid pressures
2) advection-diffusion equations for the tracer concentration within the pores

For this script we consider 3 compartments:
 - interstitial space (0)
 - PVS arteries (1)
 - PVS veins (2)

Units are: mm for space and second for time
Standard use for this script is

python3 name_script.py

authors: Alexandre Poulain, Jørgen Riseth
"""
from dolfin import *
import numpy as np
import pandas as pd
from pathlib import Path
from parameters import multicompartment_parameters
from simulation_pipeline import launch_script, fixed_parameters, modulated_parameters, prepare_simu

# experiment name
exp_name = "nonvasogenic"

# Save images ? 
save_images = True

# Load parameters
comp = ["e", "pa", "pv", "pc"] # define compartments
ncomp = len(comp)-1 # -1 because no pc in computations
solute = "gadobutrol"

# load parameters
coefficients = multicompartment_parameters(comp,solute)

# Boundary conditions for PVS
dirichlet_solute = False
dirichlet_pressure = True

# Fe type 
finite_element_type = "P1"

# Load mesh
res = 64
meshfile = "./data/mesh/synthetic_mesh_"+str(res)+".h5"

# Path to results
results_path = Path("results/"+exp_name)
results_path.mkdir(parents=True, exist_ok=True)

# Temporal parameters
dt = 3600
nb_days = 14
T = 3600.*(24.*nb_days)

# Prepare simulation
mesh, geo, Q, VV, dx, ds, brain_volume, Vcsf, white_matter_vol, grey_matter_vol, volume_brain_healthy, tumor_volume, cytotoxic_volume, vasogenic_volume, n, SD  = prepare_simu(meshfile, finite_element_type, ncomp)


# PHYSICAL PARAMETERS
# fixed 
phi0, nu, Kappa_f, gamma, w_vpv, D_free, D_eff, D_eff_arr, lmbd, l_e_pial = fixed_parameters(coefficients, ncomp)

# modulated
mod_phi_cytotoxic_e = .7
mod_phi_cytotoxic_pa = .7
mod_phi_cytotoxic_pv = .7

mod_phi_vasogenic_e = 1.
mod_phi_vasogenic_pa = 1.
mod_phi_vasogenic_pv = 1.

mod_phi_tumor_e = 1.
mod_phi_tumor_pa = 1.
mod_phi_tumor_pv = 1.

mod_permea_veins = 0.
mod_fluid_transfer_cytotoxic_pa_e = 1.
mod_fluid_transfer_cytotoxic_pv_e = .5

mod_fluid_transfer_vasogenic_pa_e = 1.
mod_fluid_transfer_vasogenic_pv_e = 1.

mod_fluid_transfer_tumor_pa_e = 1.
mod_fluid_transfer_tumor_pv_e = 0.5

mod_diff_transfer_cytotoxic_pa_e = 1.
mod_diff_transfer_cytotoxic_pv_e = .5

mod_diff_transfer_vasogenic_pa_e = 1.
mod_diff_transfer_vasogenic_pv_e = 1.

mod_diff_transfer_tumor_pa_e = 1.
mod_diff_transfer_tumor_pv_e = .5

# modulated parameters
phi0_cytotoxic, phi0_vasogenic, phi0_tumor, Kappa_f_cytotoxic, Kappa_f_vasogenic, Kappa_f_tumor, gamma_disrupt_veins, gamma_cytotoxic, gamma_vasogenic, gamma_tumor, D_eff_cytotoxic_arr , D_eff_vasogenic_arr, D_eff_tumor_arr, lmbd_cytotoxic, lmbd_vasogenic, lmbd_tumor = modulated_parameters(coefficients, ncomp, nu, 
                        lmbd, gamma, w_vpv, D_free, 
                        mod_phi_cytotoxic_e, mod_phi_cytotoxic_pa, mod_phi_cytotoxic_pv, 
                        mod_phi_vasogenic_e, mod_phi_vasogenic_pa, mod_phi_vasogenic_pv, 
                        mod_phi_tumor_e, mod_phi_tumor_pa, mod_phi_tumor_pv,
                        mod_permea_veins, 
                        mod_fluid_transfer_cytotoxic_pa_e, mod_fluid_transfer_cytotoxic_pv_e, 
                        mod_fluid_transfer_vasogenic_pa_e, mod_fluid_transfer_vasogenic_pv_e, 
                        mod_fluid_transfer_tumor_pa_e, mod_fluid_transfer_tumor_pv_e,
                        mod_diff_transfer_cytotoxic_pa_e,mod_diff_transfer_cytotoxic_pv_e, 
                        mod_diff_transfer_vasogenic_pa_e,mod_diff_transfer_vasogenic_pv_e, 
                        mod_diff_transfer_tumor_pa_e, mod_diff_transfer_tumor_pv_e
                        )
                        

N_healthy, N_edema, N_tumor, conc_SAS_arr, timevec  = launch_script(
                            mesh, dx, ds, geo, Q, VV, T, dt, results_path, finite_element_type, 
                            phi0, phi0_cytotoxic, phi0_vasogenic, phi0_tumor, 
                            Kappa_f, Kappa_f_cytotoxic, Kappa_f_vasogenic, Kappa_f_tumor, 
                            gamma, gamma_cytotoxic, gamma_vasogenic, gamma_tumor, gamma_disrupt_veins, 
                            D_eff_arr, D_eff_cytotoxic_arr, D_eff_vasogenic_arr, D_eff_tumor_arr, 
                            lmbd, lmbd_cytotoxic, lmbd_vasogenic, lmbd_tumor, l_e_pial,
                            comp, ncomp, solute, dirichlet_solute, dirichlet_pressure, volume_brain_healthy, cytotoxic_volume, vasogenic_volume, tumor_volume, SD, save = save_images)

# save CSV with number of molecules and intrinsic mean concentration
df = pd.DataFrame({"massin_tumor":N_tumor, "massin_edema": N_edema, "massin_healthy": N_healthy })
print(df)
df.to_csv(str(results_path) +"/clearance_solute.csv")

