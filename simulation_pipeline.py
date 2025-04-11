"""This file contains the function to prepare and run the simulations
"""

import numpy as np
import pandas as pd
from dolfin import *
from parameters import multicompartment_parameters
from ts_storage import TimeSeriesStorage
from pressure_model import pressure_model, K_multcomp
from solute_model import var_form_advection_diffusion_solute, solve_time_dependent_advection_diffusion


def fixed_parameters(coefficients, ncomp):
    """This function constructs the baseline parameters
    Parameters:
        - coefficients (dict): dictionary of baseline parameters computed with parameters.py
        - ncomp (int): number of compartments
    Returns:
        - phi0 (list of floats): porosities for each compartment 
        - nu (list of floats): CSF and IF dynamic viscosity 
        - Kappa_f (list of floats): effective permeability (kappa/nu) for each compartment 
        - gamma (ndarray of floats): fluid transfer coefficients 
        - w_vpv (float): permeability coefficient for fluid transfer between veins and vPVS 
        - D_free (float): free diffusion coefficient of Gadobutrol 
        - D_eff (float): effective diffusion of Gadobutrol in ECS 
        - D_eff_arr(list of floats): effect diffusion of Gd in each compartment 
        - lmbd (ndarray of floats): diffusive transfer between compartments 
        - l_e_pial (float): diffusive permeability of ECS at the pial surface
    """
    # porosity
    phi0 = ncomp*[Constant(0.)]  # Porosity V_i/V_Total
    phi0[0] = coefficients["porosity"]["e"] 
    phi0[1] = coefficients["porosity"]["pa"] + coefficients["porosity"]["pc"]/2.
    phi0[2] = coefficients["porosity"]["pv"] + coefficients["porosity"]["pc"]/2.
    
    # viscosity
    nu = np.array([0.0]*ncomp)
    nu[0] = coefficients["viscosity"]["e"]
    nu[1] = coefficients["viscosity"]["pa"]
    nu[2] = coefficients["viscosity"]["pv"]
    
    # hydraulic conductivity of fluid kappa/nu
    C_bar_e = 2.5e-9
    C_bar_pa = 0.00021459753
    C_bar_pv = 0.00007406933
    
    Kappa_f = np.copy([0.0]*ncomp)
    Kappa_f[0] = C_bar_e * phi0[0]**3/nu[0]
    Kappa_f[1] = C_bar_pa * phi0[1]**2/nu[1]
    Kappa_f[2] = C_bar_pv * phi0[2]**2/nu[2]

    # transfer coefficients
    w_vpv = 1.26e-10 # transfer between v and pv
    w_pae = coefficients["convective_fluid_transfer"][("e","pa")]
    w_pve = coefficients["convective_fluid_transfer"][("e","pv")]
    # fluid transfer between comps
    gamma = np.array([[0,w_pae,w_pve],
                  [w_pae,0,0],
                  [w_pve,0,0]])
    
    # free and effective diffusion          
    D_free = coefficients["free_diffusion"]["e"]
    D_eff =  coefficients["effective_diffusion"]["e"]
    D_eff_arr = [D_free*(phi0[0]**2/  0.34**2), phi0[1]*D_free, phi0[2]*D_free]     
    
    # permeability for Robin boundary condition
    l_e_pial = 3.5e-4 # Given in "Estimates of the permeability of the ECs pathways through the astrocyte endfoot sheath"  
                  
    # diffusive transfer between comps            
    l_pae = coefficients["diffusive_solute_transfer"][("e","pa")]
    l_pve = coefficients["diffusive_solute_transfer"][("e","pv")]
    lmbd = np.array([[0,l_pae,l_pve],
                     [l_pae,0,0],
                     [l_pve,0,0]
                     ])

    return phi0, nu, Kappa_f, gamma, w_vpv, D_free, D_eff, D_eff_arr, lmbd, l_e_pial

def modulated_parameters(coefficients, ncomp, nu, 
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
                        mod_diff_transfer_tumor_pa_e, mod_diff_transfer_tumor_pv_e):
    """This function implements the change of parameters due to porosity changes
    Parameters:
        - coefficients (dict): dictionary of baseline parameters computed with parameters.py
        - ncomp (int): number of compartments
        - nu (list of floats): CSF and IF dynamic viscosity 
        - lmbd (ndarray of floats): diffusive transfer between compartments 
        - gamma (ndarray of floats): fluid transfer coefficients 
        - w_vpv (float): permeability coefficient for fluid transfer between veins and vPVS 
        - D_free (float): free diffusion coefficient of Gadobutrol 
        - mod_phi_cytotoxic_e, mod_phi_cytotoxic_pa, mod_phi_cytotoxic_pv (floats): modulation of porosity in cytotoxic edema
        - mod_phi_vasogenic_e, mod_phi_vasogenic_pa, mod_phi_vasogenic_pv (floats): modulation of porosity in vasogenic edema
        - mod_phi_tumor_e, mod_phi_tumor_pa, mod_phi_tumor_pv (floats): modulation of porosity in tumor
        - mod_permea_veins (float) : 
        - mod_fluid_transfer_cytotoxic_pa_e, mod_fluid_transfer_cytotoxic_pv_e (floats): modulation of fluid transfer in cytotoxic region 
        - mod_fluid_transfer_vasogenic_pa_e, mod_fluid_transfer_vasogenic_pv_e (floats): modulation of fluid transfer in vasogenic region 
        - mod_fluid_transfer_tumor_pa_e, mod_fluid_transfer_tumor_pv_e (floats): modulation of fluid transfer in tumor region 
        - mod_diff_transfer_cytotoxic_pa_e,mod_diff_transfer_cytotoxic_pv_e (floats): modulation of solute transfer in cytotoxic region 
        - mod_diff_transfer_vasogenic_pa_e,mod_diff_transfer_vasogenic_pv_e (floats): modulation of solute transfer in vasogenic region 
        - mod_diff_transfer_tumor_pa_e, mod_diff_transfer_tumor_pv_e (floats): modulation of solute transfer in tumor region 
    Returns:
        - phi0_cytotoxic, phi0_vasogenic, phi0_tumor (lists of floats): modulated porosities in each zone
        - Kappa_f_cytotoxic, Kappa_f_vasogenic, Kappa_f_tumor (lists of floats): modulated permeability in each zone
        - gamma_cytotoxic, gamma_vasogenic, gamma_tumor (ndarrays of floats) : modulated fluid transfer in each zone
        - gamma_disrupt_veins (float): modulated fluid transfer between veins and vPVS
        - D_eff_cytotoxic_arr, D_eff_vasogenic_arr, D_eff_tumor_arr (arrays of floats): effective diffusions for each region 
        - lmbd_cytotoxic, lmbd_vasogenic, lmbd_tumor (arrays of ndarray): modulated diffusive permeability between compartments
    """

    # Changing porosities and impact of permeabilities
    phi0_cytotoxic = np.array([0.0]*ncomp) # Porosity V_i/V_Total
    phi0_cytotoxic[0] = coefficients["porosity"]["e"]*mod_phi_cytotoxic_e
    phi0_cytotoxic[1] = (coefficients["porosity"]["pa"]+ coefficients["porosity"]["pc"]/2.)*mod_phi_cytotoxic_pa
    phi0_cytotoxic[2] = (coefficients["porosity"]["pv"]+ coefficients["porosity"]["pc"]/2.)*mod_phi_cytotoxic_pv
    
    phi0_vasogenic = np.array([0.0]*ncomp) # Porosity V_i/V_Total
    phi0_vasogenic[0] = coefficients["porosity"]["e"]*mod_phi_vasogenic_e
    phi0_vasogenic[1] = (coefficients["porosity"]["pa"]+ coefficients["porosity"]["pc"]/2.)*mod_phi_vasogenic_pa
    phi0_vasogenic[2] = (coefficients["porosity"]["pv"]+ coefficients["porosity"]["pc"]/2.)*mod_phi_vasogenic_pv

    phi0_tumor = np.array([0.0]*ncomp) # Porosity V_i/V_Total
    phi0_tumor[0] = coefficients["porosity"]["e"]*mod_phi_tumor_e
    phi0_tumor[1] = (coefficients["porosity"]["pa"]+ coefficients["porosity"]["pc"]/2.)*mod_phi_tumor_pa
    phi0_tumor[2] = (coefficients["porosity"]["pv"]+ coefficients["porosity"]["pc"]/2.)*mod_phi_tumor_pv
    print("phi cytotoxic = "+str(phi0_cytotoxic))
    print("phi vasogenic = "+str(phi0_vasogenic))
    print("phi tumor = "+str(phi0_tumor))
    
    # disrupted permeabiilty
    C_bar_e = 2.5e-9
    C_bar_pa = 0.00021459753
    C_bar_pv = 0.00007406933
    
    Kappa_f_cytotoxic = np.copy([0.0]*ncomp)
    Kappa_f_cytotoxic[0] = C_bar_e * phi0_cytotoxic[0]**3/nu[0]
    Kappa_f_cytotoxic[1] = C_bar_pa * phi0_cytotoxic[1]**2/nu[1]
    Kappa_f_cytotoxic[2] = C_bar_pv * phi0_cytotoxic[2]**2/nu[2]
    
    Kappa_f_vasogenic = np.copy([0.0]*ncomp)
    Kappa_f_vasogenic[0] = C_bar_e * phi0_vasogenic[0]**3/nu[0]
    Kappa_f_vasogenic[1] = C_bar_pa * phi0_vasogenic[1]**2/nu[1]
    Kappa_f_vasogenic[2] = C_bar_pv * phi0_vasogenic[2]**2/nu[2]

    Kappa_f_tumor = np.copy([0.0]*ncomp)
    Kappa_f_tumor[0] = C_bar_e * phi0_tumor[0]**3/nu[0]
    Kappa_f_tumor[1] = C_bar_pa * phi0_tumor[1]**2/nu[1]
    Kappa_f_tumor[2] = C_bar_pv* phi0_tumor[2]**2/nu[2]

    print("kappa_f_cytotoxic = "+str(Kappa_f_cytotoxic*nu))
    print("kappa_f_vasogenic = "+str(Kappa_f_vasogenic*nu))
    print("kappa_f_tumor = "+str(Kappa_f_tumor*nu))

    # Disruption of BBB
    gamma_disrupt_veins = np.copy(w_vpv)*mod_permea_veins

    print("gamma_disrupt_veins = " +str(gamma_disrupt_veins) )
    # disruption of AEF barrier                  
    w_pae = coefficients["convective_fluid_transfer"][("e","pa")]
    w_pve = coefficients["convective_fluid_transfer"][("e","pv")]
    
    w_pae_cytotoxic = w_pae*mod_fluid_transfer_cytotoxic_pa_e
    w_pve_cytotoxic = w_pve*mod_fluid_transfer_cytotoxic_pv_e
    
    w_pae_vasogenic = w_pae*mod_fluid_transfer_vasogenic_pa_e
    w_pve_vasogenic = w_pve*mod_fluid_transfer_vasogenic_pv_e
    
    w_pae_tumor = w_pae*mod_fluid_transfer_tumor_pa_e
    w_pve_tumor = w_pve*mod_fluid_transfer_tumor_pv_e
    
    # fluid transfer between comps
    gamma_cytotoxic = np.array([[0,w_pae_cytotoxic,w_pve_cytotoxic],
                  [w_pae_cytotoxic,0,0],
                  [w_pve_cytotoxic,0,0]])
    
    gamma_vasogenic = np.array([[0,w_pae_vasogenic,w_pve_vasogenic],
                  [w_pae_vasogenic,0,0],
                  [w_pve_vasogenic,0,0]])
    
    gamma_tumor = np.array([[0,w_pae_tumor,w_pve_tumor],
                  [w_pae_tumor,0,0],
                  [w_pve_tumor,0,0]])

    print("gamma_cytotoxic = " +str(gamma_cytotoxic))
    print("gamma_vasogenic = " +str(gamma_vasogenic))
    print("gamma_tumor = " +str(gamma_tumor))

    # modulation of diffusion coefficient
    D_eff_cytotoxic_arr = [D_free*(phi0_cytotoxic[0]**2/  0.34**2), phi0_cytotoxic[1]*D_free, phi0_cytotoxic[2]*D_free]
    D_eff_vasogenic_arr = [D_free*(phi0_vasogenic[0]**2/  0.34**2), phi0_vasogenic[1]*D_free, phi0_vasogenic[2]*D_free]
    D_eff_tumor_arr = [D_free*(phi0_tumor[0]**2/  0.34**2), phi0_tumor[1]*D_free, phi0_tumor[2]*D_free]

    print("D_eff_cytotoxic_arr = " +str(D_eff_cytotoxic_arr))
    print("D_eff_vasogenic_arr = " +str(D_eff_vasogenic_arr))
    print("D_eff_tumor_arr = " +str(D_eff_tumor_arr))
    
    # disruption of diffusive transfer
    
    l_cytotoxic_pae = coefficients["diffusive_solute_transfer"][("e","pa")]*mod_diff_transfer_cytotoxic_pa_e
    l_cytotoxic_pve = coefficients["diffusive_solute_transfer"][("e","pv")]*mod_diff_transfer_cytotoxic_pv_e
    lmbd_cytotoxic = np.array([[0,l_cytotoxic_pae,l_cytotoxic_pve],
                     [l_cytotoxic_pae,0,0],
                     [l_cytotoxic_pve,0,0]
                     ])
                     
    l_vasogenic_pae = coefficients["diffusive_solute_transfer"][("e","pa")]*mod_diff_transfer_vasogenic_pa_e
    l_vasogenic_pve = coefficients["diffusive_solute_transfer"][("e","pv")]*mod_diff_transfer_vasogenic_pv_e
    lmbd_vasogenic = np.array([[0,l_vasogenic_pae,l_vasogenic_pve],
                     [l_vasogenic_pae,0,0],
                     [l_vasogenic_pve,0,0]
                     ])
                     
    l_tumor_pae = coefficients["diffusive_solute_transfer"][("e","pa")]*mod_diff_transfer_tumor_pa_e
    l_tumor_pve = coefficients["diffusive_solute_transfer"][("e","pv")]*mod_diff_transfer_tumor_pv_e
    lmbd_tumor = np.array([[0,l_tumor_pae,l_tumor_pve],
                     [l_tumor_pae,0,0],
                     [l_tumor_pve,0,0]
                     ])

    print("lmbd_cytotoxic = " + str(lmbd_cytotoxic))
    print("lmbd_vasogenic = " + str(lmbd_vasogenic))
    print("lmbd_tumor = " +str(lmbd_tumor))
    
    
    return phi0_cytotoxic, phi0_vasogenic, phi0_tumor, Kappa_f_cytotoxic, Kappa_f_vasogenic, Kappa_f_tumor, gamma_disrupt_veins, gamma_cytotoxic, gamma_vasogenic, gamma_tumor, D_eff_cytotoxic_arr, D_eff_vasogenic_arr, D_eff_tumor_arr, lmbd_cytotoxic, lmbd_vasogenic, lmbd_tumor
    
    
def prepare_simu(meshfile, finite_element_type, ncomp):
    """This function sets up the simulation.
    It loads the mesh, prepares the FE setup, defines the measures of volume and surface,
    and computes the important volumes to compute averages of solutes later on.  
    Parameters:
        - meshfile (str): the path to the mesh
        - finite_element_type (str): the finite element space type (it should be "P1")
        - ncomp (int): number of compartments
    Returns:
        - mesh (Mesh): the mesh 
        - geo (ufl_cell): the type of cell 
        - Q, VV (FE spaces): mixed finite element space and P1 space  
        - dx, ds (Measures): volume and surface measures for variational forms 
        - brain_volume, Vcsf, white_matter_vol, grey_matter_vol, volume_brain_healthy, tumor_volume, cytotoxic_volume, vasogenic_volume (floats): volumes of the regions
        - n (FacetNormal): normal to the boundary 
        - SD (MeshFunction): the subdomains
    """

    # Load mesh
    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(), meshfile, "r")
    hdf.read(mesh, "/mesh", False)
    SD = MeshFunction("size_t", mesh,mesh.topology().dim())
    hdf.read(SD, "/subdomains")
    bnd = MeshFunction("size_t", mesh,mesh.topology().dim()-1)
    hdf.read(bnd, "/boundaries")
    
    # Get the minimum cell size
    geo = mesh.ufl_cell()
    h = mesh.hmin()
    
    # Finite element functions
    if finite_element_type == "P1":
        P1 = FiniteElement('CG',geo,1)
        ME = MixedElement(ncomp*[P1])
        Q = FunctionSpace(mesh,ME)
        VV = FunctionSpace(mesh,P1)
    else:
        print("Not implemented yet")
        exit(0)
        
    # Load measure data
    dx = Measure("dx", domain=VV.mesh(),  subdomain_data=SD)
    ds = Measure("ds", domain=VV.mesh(),  subdomain_data=SD) # surface
    n = FacetNormal(mesh) # normal vector

    # Compute surface area of the mesh and its volume
    surface_area = assemble(1.0*ds)  # in mm^2
    print('Surface area: ', surface_area, ' mm^2')
    brain_volume = assemble(1.*dx)
    print('brain volume: ', brain_volume, ' mm^3')
    Vcsf = 0.1 * brain_volume
    print('csf volume: ', Vcsf, ' mm^3')
    white_matter_vol = assemble(1.*dx(2))
    grey_matter_vol = assemble(1.*dx(1))
    volume_brain_healthy = grey_matter_vol+white_matter_vol
    tumor_volume = assemble(1.*dx(4))
    vasogenic_volume = assemble(1.*dx(3))
    cytotoxic_volume = assemble(1.*dx(6))
    print('tumor volume: ', tumor_volume, ' mm^3')
    print('cytotoxic volume: ', cytotoxic_volume, ' mm^3')
    print('vasogenic volume: ', vasogenic_volume, ' mm^3')
    
    
    return mesh, geo, Q, VV, dx, ds, brain_volume, Vcsf, white_matter_vol, grey_matter_vol, volume_brain_healthy, tumor_volume, cytotoxic_volume, vasogenic_volume, n, SD
    
def launch_script( mesh, dx, ds, geo, Q, VV, T, dt, results_path, finite_element_type, 
                            phi0, phi0_cytotoxic, phi0_vasogenic, phi0_tumor, 
                            Kappa_f, Kappa_f_cytotoxic, Kappa_f_vasogenic, Kappa_f_tumor, 
                            gamma, gamma_cytotoxic, gamma_vasogenic, gamma_tumor, gamma_disrupt_veins, 
                            D_eff_arr, D_eff_cytotoxic_arr, D_eff_vasogenic_arr, D_eff_tumor_arr, 
                            lmbd, lmbd_cytotoxic, lmbd_vasogenic, lmbd_tumor, l_e_pial,
                            comp, ncomp, solute, dirichlet_solute, dirichlet_pressure, healthy_volume, cytotoxic_volume, vasogenic_volume, tumor_volume, SD, save = True):
    """the function launch the simulation script
    
    """
    print("*********PARAMETERS*********")

    print("phi0" + str(phi0))
    print("phi0_cytotoxic" + str(phi0_cytotoxic))
    print("phi0_vasogenic" + str(phi0_vasogenic))
    print("phi0_tumor" + str(phi0_tumor))

    print("Kappa_f = " +str(Kappa_f))
    print("Kappa_f_cytotoxic = " + str(Kappa_f_cytotoxic))
    print("Kappa_f_vasogenic = " +str(Kappa_f_vasogenic))
    print("Kappa_f_tumor = " + str(Kappa_f_tumor))
    print("--------")
    
    print("D_eff_arr = " + str(D_eff_arr))
    print("D_eff_cytotoxic_arr" + str(D_eff_cytotoxic_arr))
    print("D_eff_vasogenic_arr = " + str(D_eff_vasogenic_arr))
    print("D_eff_tumor_arr = " + str(D_eff_tumor_arr))
    
    print("--------")
    
    print("gamma = " +str(gamma))
    print("gamma_cytotoxic = " +str(gamma_cytotoxic))
    print("gamma_vasogenic = " +str(gamma_vasogenic))
    print("gamma_tumor = " +str(gamma_tumor))

    print("--------")
    print("lmbd = " + str(lmbd))
    print("lmbd_cytotoxic = " +str(lmbd_cytotoxic))
    print("lmbd_vasogenic = " +str(lmbd_vasogenic))
    print("lmbd_tumor = "+str(lmbd_tumor))

    
    p = TrialFunctions(Q)
    q = TestFunctions(Q)
    
    ### Fixed parameters
    # pressure at pial surface
    p_CSF = 10.*133.333
    p_pial_pa = p_CSF + 0.03*133.33
    p_veins = 16.0*133.33
    # Fluid exchange at the pial surface
    gamma_pial_pa = Constant(1.2e-7)
    gamma_pial_pv = Constant(1.5e-7)
    gamma_pial_e = Constant(1.0e-8)
    

    ### Presure model
    p_new = pressure_model(Q, p, q, dx, ds, Kappa_f,Kappa_f_cytotoxic, Kappa_f_vasogenic, Kappa_f_tumor, gamma, gamma_cytotoxic, gamma_vasogenic, gamma_tumor, gamma_disrupt_veins, p_CSF, p_pial_pa, p_veins, gamma_pial_pa, gamma_pial_pv, gamma_pial_e, ncomp, results_path, mesh, VV,  save_pressure = save, dirichlet = dirichlet_pressure)
    
    
    if save:
        # Compute exchange between compartments
        volumetric_flux_e_pa = assemble(gamma[0,1]*(p_new[0] - p_new[1])  * dx(1) + gamma[0,1]*(p_new[0] - p_new[1])  * dx(2) )
        print("Q_{e,pa} = " +str(volumetric_flux_e_pa))
        volumetric_flux_e_pv = assemble(gamma[0,2]*(p_new[0] - p_new[2])  * dx(1) + gamma[0,2]*(p_new[0] - p_new[2])  * dx(2) )
        print("Q_{e,pv} = " +str(volumetric_flux_e_pv))
        volumetric_flux_pa_pv = assemble(gamma[2,1]*(p_new[1] - p_new[2])  * dx(1) +gamma[2,1]*(p_new[1] - p_new[2])  * dx(2) )
        print("Q_{pv,pa} = " +str(volumetric_flux_pa_pv))
        
        
        volumetric_flux_pa_e = assemble(gamma_vasogenic[0,1]*(p_new[0] - p_new[1])  * dx(3) + gamma_cytotoxic[0,1]*(p_new[0] - p_new[1])  * dx(6) +  gamma_tumor[0,1]*(p_new[0] - p_new[1])  * dx(4))
        print("Q_{e,pa} in tumor + edema = " +str(volumetric_flux_pa_e))
        
        volumetric_flux_pv_e = assemble(gamma_vasogenic[0,2]*(p_new[0] - p_new[2])  * dx(3) + gamma_cytotoxic[0,2]*(p_new[0] - p_new[2])  * dx(6) +  gamma_tumor[0,2]*(p_new[0] - p_new[2])  * dx(4))
        print("Q_{e,pv} in tumor + edema = " +str(volumetric_flux_pv_e))
        
        data = {"(e,pa)": [volumetric_flux_pa_e], "(e,pv)":[volumetric_flux_pv_e]}
        df = pd.DataFrame(data)
        df.to_csv(str(results_path) +"/fluid_exchange_ECS_PVE_tumor_and_edema.csv")
        
        average_pressure_in_tumor = assemble(p_new[0]*dx(3)+p_new[0]*dx(4)+ p_new[0]*dx(6))/(cytotoxic_volume + vasogenic_volume + tumor_volume)
        print("average pressure inside tumor and edema in mmHg = " +str(average_pressure_in_tumor))
        
        print("Compute velocities...")
        W = VectorFunctionSpace(mesh, 'P', 1)
        
        K_e_multcomp = K_multcomp(SD, Kappa_f[0], Kappa_f_cytotoxic[0], Kappa_f_vasogenic[0], Kappa_f_tumor[0])
        
        
        flux_u = project(-K_e_multcomp*Constant(1./phi0[0])*grad(p_new[0]), W, solver_type='gmres')
        flux_x, flux_y, flux_z = flux_u.split(deepcopy=True)  # extract components
        flux_x_nodal_values = flux_x.vector()[:]
        flux_y_nodal_values = flux_y.vector()[:]
        flux_z_nodal_values = flux_z.vector()[:]
        mag = np.power( np.power(flux_x_nodal_values,2.) + np.power(flux_y_nodal_values,2.) + np.power(flux_z_nodal_values,2.),1./2.)
        print("max velocity in ECS = " +str(max(mag)))
        print("mean velocity in ECS = " +str(np.mean(mag)))
        mean_vel_ecs = np.mean(mag)
        max_vel_ecs = max(mag)
        
        K_pa_multcomp =  K_multcomp(SD, Kappa_f[1], Kappa_f_cytotoxic[1], Kappa_f_vasogenic[1], Kappa_f_tumor[1])
        
        flux_u = project(-K_pa_multcomp*Constant(1./phi0[1])*grad(p_new[1]), W, solver_type='gmres')
        
        flux_x, flux_y, flux_z = flux_u.split(deepcopy=True)  # extract components
        flux_x_nodal_values = flux_x.vector()[:]
        flux_y_nodal_values = flux_y.vector()[:]
        flux_z_nodal_values = flux_z.vector()[:]
        mag = np.power( np.power(flux_x_nodal_values,2.) + np.power(flux_y_nodal_values,2.) + np.power(flux_z_nodal_values,2.),1./2.)
        print("max velocity in PVS arteries = " +str(max(mag)))
        print("mean velocity in PVS arteries = " +str(np.mean(mag)))
        mean_vel_pa = np.mean(mag)
        max_vel_pa = max(mag)
        
        K_pv_multcomp =  K_multcomp(SD, Kappa_f[2], Kappa_f_cytotoxic[2], Kappa_f_vasogenic[2], Kappa_f_tumor[2])
        
        flux_u = project(-K_pv_multcomp*Constant(1./phi0[2])*grad(p_new[2]), W, solver_type='gmres')
        flux_x, flux_y, flux_z = flux_u.split(deepcopy=True)  # extract components
        flux_x_nodal_values = flux_x.vector()[:]
        flux_y_nodal_values = flux_y.vector()[:]
        flux_z_nodal_values = flux_z.vector()[:]
        mag = np.power( np.power(flux_x_nodal_values,2.) + np.power(flux_y_nodal_values,2.) + np.power(flux_z_nodal_values,2.),1./2.)
        print("max velocity in PVS veins = " +str(max(mag)))
        print("mean velocity in PVS veins = " +str(np.mean(mag)))
        mean_vel_pv = np.mean(mag)
        max_vel_pv = max(mag)
        
        data = {"e":[mean_vel_ecs, max_vel_ecs], "pa": [mean_vel_pa, max_vel_pa], "pv":[mean_vel_pv, max_vel_pv]}
        
        df = pd.DataFrame(columns=["mean_vel", "max_vel"], index = ["e","pa","pv"])
        df.loc["e"] = [mean_vel_ecs, max_vel_ecs]
        df.loc["pa"] = [mean_vel_pa, max_vel_pa]
        df.loc["pv"] = [mean_vel_pv, max_vel_pv]

        df.to_csv(str(results_path) +"/velocity.csv")
        
    
       
    ### Solute diffusion part 
    p = TrialFunctions(Q)
    q = TestFunctions(Q)
    
    # Zero initial condition.
    czero = Expression('0.0',degree=1)
    init_c = interpolate(czero,VV)
    c_ = Function(Q) # Function to save solution
    assign(c_, [init_c, init_c,init_c])
    cn = Function(Q) # function to save concentrations at the previous time step
    
    # initial concentration in SAS
    conc_SAS = Constant(0.)
    
    # model
    G, bcs_conc = var_form_advection_diffusion_solute(
                            ncomp, p,q, Q, cn, p_new, dx, ds, dt, 
                            phi0, phi0_cytotoxic, phi0_vasogenic, phi0_tumor, 
                            D_eff_arr, D_eff_cytotoxic_arr, D_eff_vasogenic_arr, D_eff_tumor_arr, 
                            Kappa_f, Kappa_f_cytotoxic, Kappa_f_vasogenic, Kappa_f_tumor, 
                            lmbd, lmbd_cytotoxic, lmbd_vasogenic, lmbd_tumor, 
                            gamma, gamma_cytotoxic, gamma_vasogenic, gamma_tumor, 
                            l_e_pial, conc_SAS, dirichlet = dirichlet_solute)

    # solve
    N_healthy, N_edema, N_tumor, SAS_conc, timevec = solve_time_dependent_advection_diffusion(
                                    Q, G, c_, cn, conc_SAS, 
                                    phi0, phi0_cytotoxic, phi0_vasogenic, phi0_tumor, 
                                    dx, dt, bcs_conc, mesh, results_path, VV, T, ncomp, dirichlet = dirichlet_solute, save = save)
    
    # prepare return
    N_healthy = np.array(N_healthy)
    N_edema = np.array(N_edema)
    N_tumor = np.array(N_tumor)
    
    
    
    return N_healthy, N_edema, N_tumor, SAS_conc, timevec