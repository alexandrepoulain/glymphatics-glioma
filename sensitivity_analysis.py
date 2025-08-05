#!/usr/bin/env python
# coding: utf-8

"""
- N is the number of simulations to run in order to perform the sensitivity analysis
- P is the number of varying parameters

"""

from dolfin import *
import numpy as np
import pandas as pd
from pathlib import Path
from parameters import multicompartment_parameters
from simulation_pipeline import launch_script, fixed_parameters, modulated_parameters, prepare_simu
import scipy.stats as st


def fun_simu(X, mesh_reso, dt, T, param_set_index):
    """Function to be used in "model_execution" of SAFE package
    
    Inputs:
        - X (np.ndarray, size = (N,P)): matrix of parameters, each row is a set of new parameters
        - other_params (np.ndarray): fixed parameters

    Outupts: 
        - mass_tumor, mass_edema (float): mass of solutes in the edema and tumor 

    """
    # experiment name
    exp_name = "mixed_edema"

    # Save images ? 
    save_images = False

    # Load parameters
    comp = ["e", "pa", "pv", "pc"] # define compartments
    ncomp = len(comp)-1 # -1 because no pc in computations
    solute = "gadobutrol"

    # load parameters
    coefficients = multicompartment_parameters(comp,solute)

    # Boundary conditions for PVS
    dirichlet_solute = False
    dirichlet_pressure = True

    # FE type 
    finite_element_type = "P1"

    # Load mesh
    res = mesh_reso
    meshfile = "./data/mesh/synthetic_mesh_"+str(res)+".h5"

    # Path to results
    results_path = Path("results_sensitivity/" + exp_name + str(param_set_index))
    results_path.mkdir(parents=True, exist_ok=True)

    # Prepare simulation
    mesh, geo, Q, VV, dx, ds, brain_volume, Vcsf, white_matter_vol, grey_matter_vol, volume_brain_healthy, tumor_volume, cytotoxic_volume, vasogenic_volume, n, SD  = prepare_simu(meshfile, finite_element_type, ncomp)


    # PHYSICAL PARAMETERS
    # fixed 
    phi0, nu, Kappa_f, gamma, w_vpv, D_free, D_eff, D_eff_arr, lmbd, l_e_pial = fixed_parameters(coefficients, ncomp) # fixed parameters

    # modulated
    mod_phi_cytotoxic_e = X[0]
    mod_phi_cytotoxic_pa = X[1]
    mod_phi_cytotoxic_pv = X[2]

    mod_phi_vasogenic_e = X[3]
    mod_phi_vasogenic_pa = X[4]
    mod_phi_vasogenic_pv = X[5]

    mod_phi_tumor_e = X[6]
    mod_phi_tumor_pa = X[7]
    mod_phi_tumor_pv = X[8]

    
    mod_fluid_transfer_cytotoxic_pa_e = X[9]
    mod_fluid_transfer_cytotoxic_pv_e = X[10]

    mod_fluid_transfer_vasogenic_pa_e = X[11]
    mod_fluid_transfer_vasogenic_pv_e = X[12]

    mod_fluid_transfer_tumor_pa_e = X[13]
    mod_fluid_transfer_tumor_pv_e = X[14]

    mod_diff_transfer_cytotoxic_pa_e = X[15]
    mod_diff_transfer_cytotoxic_pv_e = X[16]

    mod_diff_transfer_vasogenic_pa_e = X[17]
    mod_diff_transfer_vasogenic_pv_e = X[18]

    mod_diff_transfer_tumor_pa_e = X[19]
    mod_diff_transfer_tumor_pv_e = X[20]

    mod_permea_veins_vaso = X[21]
    mod_permea_veins_tumo = X[22]

    # modulated parameters
    phi0_cytotoxic, phi0_vasogenic, phi0_tumor, Kappa_f_cytotoxic, Kappa_f_vasogenic, Kappa_f_tumor, gamma_disrupt_veins_vaso,gamma_disrupt_veins_tumo , gamma_cytotoxic, gamma_vasogenic, gamma_tumor, D_eff_cytotoxic_arr , D_eff_vasogenic_arr, D_eff_tumor_arr, lmbd_cytotoxic, lmbd_vasogenic, lmbd_tumor = modulated_parameters(coefficients, ncomp, nu, 
                            lmbd, gamma, w_vpv, D_free, 
                            mod_phi_cytotoxic_e, mod_phi_cytotoxic_pa, mod_phi_cytotoxic_pv, 
                            mod_phi_vasogenic_e, mod_phi_vasogenic_pa, mod_phi_vasogenic_pv, 
                            mod_phi_tumor_e, mod_phi_tumor_pa, mod_phi_tumor_pv,
                            mod_permea_veins_vaso, mod_permea_veins_tumo,
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
                                gamma, gamma_cytotoxic, gamma_vasogenic, gamma_tumor, gamma_disrupt_veins_vaso, gamma_disrupt_veins_tumo, 
                                D_eff_arr, D_eff_cytotoxic_arr, D_eff_vasogenic_arr, D_eff_tumor_arr, 
                                lmbd, lmbd_cytotoxic, lmbd_vasogenic, lmbd_tumor, l_e_pial,
                                comp, ncomp, solute, dirichlet_solute, dirichlet_pressure, volume_brain_healthy, cytotoxic_volume, vasogenic_volume, tumor_volume, SD, save = save_images)

    # save CSV with number of molecules and intrinsic mean concentration
    df = pd.DataFrame({"massin_tumor":N_tumor, "massin_edema": N_edema, "massin_healthy": N_healthy })
    print(df)
    df.to_csv(str(results_path) +"/clearance_solute.csv")

    return N_edema + N_tumor


def GSA_fun():
    """
        This function performs the GSA analysis using Morris Method

    """

    # Define the thresholds for the modulations
    X_Labels = ['$\phi_e^\text{cyto}$', '$\phi_{pa}^\text{cyto}$', '$\phi_{pv}^\text{cyto}$', 
                '$\phi_e^\text{vaso}$', '$\phi_{pa}^\text{vaso}$', '$\phi_{pv}^\text{vaso}$',
                '$\phi_e^\text{tumo}$', '$\phi_{pa}^\text{tumo}$', '$\phi_{pv}^\text{tumo}$',
                '$\gamma_{e,pa}^\text{cyto}$','$\gamma_{e,pv}^\text{cyto}$',
                '$\gamma_{e,pa}^\text{vaso}$','$\gamma_{e,pv}^\text{vaso}$',
                '$\gamma_{e,pa}^\text{tumo}$','$\gamma_{e,pv}^\text{tumo}$',
                '$\lambda{e,pa}^\text{cyto}$','$\lambda{e,pv}^\text{cyto}$',
                '$\lambda{e,pa}^\text{vaso}$','$\lambda{e,pv}^\text{vaso}$',
                '$\lambda{e,pa}^\text{tumo}$','$\lambda{e,pv}^\text{tumo}$',
                '$\gamma_{v,pv}^\text{vaso}$', '$\gamma_{v,pv}^\text{tumo}$'] # Name of parameters (used to customize plots)
    M = len(X_Labels) # Number of varying parameters
    distr_fun = st.uniform # Parameter distributions
    xmin = [0.]*M # Parameter ranges (lower bound)
    xmax = [2.]*M # Parameter ranges (upper bound)
    # Save lower and upper bound in the appropriate format to be passed on to the sampling function:
    distr_par = [np.nan] * M
    for i in range(M):
        distr_par[i] = [xmin[i], xmax[i] - xmin[i]]
    # Choose sampling strategy and size:
    samp_strat = 'lhs' # Latin Hypercube
    design_type = 'radial'
    # other options for design type:
    # design_type  = 'trajectory'
    from safepython.sampling import OAT_sampling
    r = 1 # number of Elementary effects
    X = OAT_sampling(r, M, distr_fun, distr_par, samp_strat, design_type)

    print(X)

    # Execute the model against all the samples in 'X':
    from safepython.model_execution import model_execution # Module to execute the model

    # Temporal parameters
    dt = 3600.
    nb_days = 2.
    T = 3600.*(24.*nb_days)
    # mesh resolution
    mesh_reso = 32

    Y = model_execution(fun_simu, X, mesh_reso, dt, T, 1)

    np.savetxt("sensitivity_Y.csv", Y, delimiter=",")



    # Plot results in the plane (mean(EE), std(EE)):
    import safepython.EET as EET # module to perform the EET
    import matplotlib.pyplot as plt

    mi, sigma, _ = EET.EET_indices(r, xmin, xmax, X, Y, design_type)


    EET.EET_plot(mi, sigma, X_Labels)
    plt.show()


if __name__ == '__main__':
    GSA_fun()