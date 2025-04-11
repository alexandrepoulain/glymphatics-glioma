"""This file contains the solute model
"""
import numpy as np
from dolfin import *
from ts_storage import TimeSeriesStorage

def L2_project(Q, dx, c, phi, phi_edema_cytotoxic, phi_edema_vasogenic, phi_tumor, ncomp, c1):
    """This function defines the macroscopic concentration fields using L2 projection
    Parameters:
       - Q (Mixed FE space): the finite element space for the concentration fields.
       - dx (Measure): volume measure.
       - phi, phi_edema_cytotoxic, phi_edema_vasogenic, phi_tumor (array of floats): porosities in each compartment and each regions
       - ncomp (int): number of compartments.
       - c1 (FE function): Function in Q to get the results after the projection. 
    """
    p = TrialFunction(Q)
    q = TestFunction(Q)

    L2_proj_var = 0
    for i in range(ncomp):
        L2_proj_var += p[i]*q[i]*dx
        L2_proj_var -= phi[i]*c[i]*q[i]*dx(1)
        L2_proj_var -= phi[i]*c[i]*q[i]*dx(2)
        L2_proj_var -= phi_edema_vasogenic[i]*c[i]*q[i]*dx(3)
        L2_proj_var -= phi_edema_cytotoxic[i]*c[i]*q[i]*dx(6)
        L2_proj_var -= phi_tumor[i]*c[i]*q[i]*dx(4)
    
    a = lhs(L2_proj_var)
    L = rhs(L2_proj_var)
    A = assemble(a)
    b = assemble(L)
    
    solve(A, c1.vector(), b, 'gmres', 'ilu')


def var_form_advection_diffusion_solute(ncomp, p,q, Q, cn, press, dx, ds, dt, 
                            phi0, phi0_cytotoxic, phi0_vasogenic, phi0_tumor, 
                            D_eff, D_eff_cytotoxic, D_eff_vasogenic, D_eff_tumor, 
                            Kappa_f, Kappa_f_cytotoxic, Kappa_f_vasogenic, Kappa_f_tumor, 
                            lmbd, lmbd_cytotoxic, lmbd_vasogenic, lmbd_tumor, 
                            gamma, gamma_cytotoxic, gamma_vasogenic, gamma_tumor, 
                            l_e_pial, conc_SAS, dirichlet = False):
    """This function constructs the variational form of the convection-diffusion equation for the solute.
    It returns the variational form and the boundary conditions. 
    
    Parameters:
        - ncomp (int): number of compartments 
        - p,q: trial and test functions
        - Q (FE space): mixed finite element space P1 * ncomp
        - cn (Function(Q)): Finite element function to store the previous time step solution
        - press (Function(Q)): the pressure fields
        - dx, ds (Measure): measures for volume and surface
        - dt (float): time step
        - phi0, phi0_edema, phi0_tumor (lists of floats): porosities
        - D_eff, D_eff_cytotoxic, D_eff_vasogenic, D_eff_tumor (lists of floats): effective diffusion 
                                                      coefficient in each compartment for all regions
        - Kappa_f, Kappa_f_cytotoxic, Kappa_f_vasogenic, Kappa_f_tumor (lists of floats): permeabilities 
                                                                   in all regions for the 3 compartments
        - lmbd, lmbd_cytotoxic, lmbd_vasogenic, lmbd_tumor (ndarrays of float): diffusive permeability between compartments
        
        - gamma, gamma_cytotoxic, gamma_vasogenic, gamma_tumor (ndarrays of float): convective permeability between compartments
        - gamma_tumor (matrix of float): convective permeability between compartments (in tumor)
        - l_pial (float): diffusive permeabity of solute at the pial surface (for the 3 compartments)
        - conc_SAS (Fenics constant): value of concentration in SAS (will change during simulation... 
            thus rhs must be updated)
    Outputs: 
        - G: the variational form
        - bcs_conc: array of boundary conditions if Dirichlet boundary conditions are used. 
    """
    # Variational form for the tracer concentration equations
    G = 0 # init variational form
    for i in range(ncomp):

        G += Constant(D_eff[i])*inner(grad(p[i]),grad(q[i]))*dx(1) 
        G += Constant(D_eff[i])*inner(grad(p[i]),grad(q[i]))*dx(2)
        G += Constant(D_eff_cytotoxic[i])*inner(grad(p[i]),grad(q[i]))*dx(6)
        G += Constant(D_eff_vasogenic[i])*inner(grad(p[i]),grad(q[i]))*dx(3)
        G += Constant(D_eff_tumor[i])*inner(grad(p[i]),grad(q[i]))*dx(4)

        G += Constant(phi0[i]/dt)*inner(p[i]-cn[i],q[i])*dx(1)
        G += Constant(phi0[i]/dt)*inner(p[i]-cn[i],q[i])*dx(2)
        G += Constant(phi0_cytotoxic[i]/dt)*inner(p[i]-cn[i],q[i])*dx(6)
        G += Constant(phi0_vasogenic[i]/dt)*inner(p[i]-cn[i],q[i])*dx(3)
        G += Constant(phi0_tumor[i]/dt)*inner(p[i]-cn[i],q[i])*dx(4)

        G += Constant(Kappa_f[i])*inner(p[i],inner(grad(press[i]),grad(q[i])))*dx(1)
        G += Constant(Kappa_f[i])*inner(p[i],inner(grad(press[i]),grad(q[i])))*dx(2)
        G += Constant(Kappa_f_cytotoxic[i])*inner(p[i],inner(grad(press[i]),grad(q[i])))*dx(6)
        G += Constant(Kappa_f_vasogenic[i])*inner(p[i],inner(grad(press[i]),grad(q[i])))*dx(3)
        G += Constant(Kappa_f_tumor[i])*inner(p[i],inner(grad(press[i]),grad(q[i])))*dx(4)
        
        # mass exchange
        for j in range(ncomp):
            if i != j:
                # diffusive transfer
                G += Constant(lmbd[i][j])*inner(p[i]-p[j],q[i])*dx(1)
                G += Constant(lmbd[i][j])*inner(p[i]-p[j],q[i])*dx(2)
                G += Constant(lmbd_cytotoxic[i][j])*inner(p[i]-p[j],q[i])*dx(6)
                G += Constant(lmbd_vasogenic[i][j])*inner(p[i]-p[j],q[i])*dx(3)
                G += Constant(lmbd_tumor[i][j])*inner(p[i]-p[j],q[i])*dx(4)
                # convective transfer
                G += Constant(gamma[i][j])*(press[i]-press[j]) *inner(p[j]+p[i],q[i])*Constant(1./2.)*dx(1)
                G += Constant(gamma[i][j])*(press[i]-press[j]) *inner(p[j]+p[i],q[i])*Constant(1./2.)*dx(2)
                G += Constant(gamma_cytotoxic[i][j])*(press[i]-press[j]) *inner(p[j]+p[i],q[i])*Constant(1./2.)*dx(6)
                G += Constant(gamma_vasogenic[i][j])*(press[i]-press[j]) *inner(p[j]+p[i],q[i])*Constant(1./2.)*dx(3)
                G += Constant(gamma_tumor[i][j])*(press[i]-press[j]) *inner(p[j]+p[i],q[i])*Constant(1./2.)*dx(4)
                
    # Robin boundary conditions for ECS
    G -= phi0[0]*Constant(l_e_pial/10.)*(conc_SAS - p[0])*q[0]*ds(1)
    G -= phi0[0]*Constant(l_e_pial/10.)*(conc_SAS - p[0])*q[0]*ds(2)
    G -= phi0_cytotoxic[0]*Constant(l_e_pial/10.)*(conc_SAS - p[0])*q[0]*ds(6)
    G -= phi0_vasogenic[0]*Constant(l_e_pial/10.)*(conc_SAS - p[0])*q[0]*ds(3)
    G -= phi0_tumor[0]*Constant(l_e_pial/10.)*(conc_SAS - p[0])*q[0]*ds(4)
    
    if dirichlet:
        bcs_conc = [DirichletBC(Q.sub(1), conc_SAS, 'on_boundary'), DirichletBC(Q.sub(2), conc_SAS, 'on_boundary')] # Dirichlet BC on PVS boundaries
    else:
        bcs_conc = []
        G -= phi0[1]*Constant(l_e_pial)*(conc_SAS - p[1])*q[1]*ds(1)
        G -= phi0[1]*Constant(l_e_pial)*(conc_SAS - p[1])*q[1]*ds(2)
        G -= phi0_cytotoxic[1]*Constant(l_e_pial)*(conc_SAS - p[1])*q[1]*ds(6)
        G -= phi0_vasogenic[1]*Constant(l_e_pial)*(conc_SAS - p[1])*q[1]*ds(3)
        G -= phi0_tumor[1]*Constant(l_e_pial)*(conc_SAS - p[1])*q[1]*ds(4)
        
        G -= phi0[2]*Constant(l_e_pial)*(conc_SAS - p[2])*q[2]*ds(1)
        G -= phi0[2]*Constant(l_e_pial)*(conc_SAS - p[2])*q[2]*ds(2)
        G -= phi0_cytotoxic[2]*Constant(l_e_pial)*(conc_SAS - p[2])*q[2]*ds(6)
        G -= phi0_vasogenic[2]*Constant(l_e_pial)*(conc_SAS - p[2])*q[2]*ds(3)
        G -= phi0_tumor[2]*Constant(l_e_pial)*(conc_SAS - p[2])*q[2]*ds(4)
        
    return G, bcs_conc


def solve_time_dependent_advection_diffusion(Q, G, c_, cn, conc_SAS, 
                                    phi0, phi0_cytotoxic, phi0_vasogenic, phi0_tumor, 
                                    dx, dt, bcs_conc, mesh, results_path, VV, T, ncomp, dirichlet = False, save = False):
    """Solves the time dependent multicompartment solute transport
    Parameters:
        - Q (Mixed FE space): the finite element space P1 * ncomp
        - G (var form): the variational form of the solute problem
        - c_ (FE function): the concentration at the time step t=0 that will be later used to compute c^{n+1}
        - cn (FE function): function to save the concentration at the previous time step c^n
        - conc_SAS (Constant): the initial SAS concentration (wil then be updated in the time loop)
        - phi0, phi0_cytotoxic, phi0_vasogenic, phi0_tumor (lists of floats): the porosities for each compartment and each region
        - dx (Measure): measure for the volume
        - dt (float): time step
        - bcs_conc (list of boundary conditions): the boundary conditions for the solute problem
        - mesh (Mesh): the mesh
        - results_path (Path): path to the results folder
        - VV (FE space): the P1 finite element space
        - T (float): final time
        - ncomp (int): the number of compartments
        - dirichlet (bool): if true it updates the dirichlet BCs for the PVSs (careful this boool should be the same as used in var_form_advection_diffusion_solute)
        - save (bool): to save of the concentration fields.
    Return:
        - c_healthy, c_edema, c_tumor (lists of floats): mass of solute in each region  
        - conc_SAS_arr (list of floats): the values of the concentration in SAS during the simulation 
        - timevec (list of floats): list containing the time points
    """
    
    # apply boundary conditions to initial time 
    [bc.apply(c_.vector()) for bc in bcs_conc]

    # assembling
    print("Assembling diffusion problem")
    a_conc = lhs(G)
    L_conc = rhs(G)
    A_conc = assemble(a_conc)
    b_conc = assemble(L_conc)
    print("Done.")
    
    c1 = Function(Q)
    
    # preparing saving
    if save:
        
        storage_cecs = TimeSeriesStorage("w", results_path, mesh=mesh, V=VV, name="ecs macro")
        storage_carteries = TimeSeriesStorage("w", results_path, mesh=mesh, V=VV, name="arteries macro")
        storage_cveins = TimeSeriesStorage("w", results_path, mesh=mesh, V=VV, name="veins macro")
        
        L2_project(Q, dx, c_, phi0, phi0_cytotoxic, phi0_vasogenic, phi0_tumor, ncomp, c1)
        c2 = c1.split()
        
        storage_cecs.write(c2[0], 0.)
        storage_carteries.write(c2[1], 0.)
        storage_cveins.write(c2[2], 0.)
        
    # initialize list to save solute masses
    c_healthy = [0]
    c_edema = [0]
    c_tumor = [0]
    MIC_healthy = [0]
    MIC_edema = [0]
    MIC_tumor = [0]
    
    conc_SAS_arr = [0]
    
    # Time steping
    t=0.0
    t+=dt
    it =1
    timevec = [0]
    
    while t< T : # Added dt/2 to ensure final time included.
        print('t = ' +str(t) +"/"+str(T))

        solute_amount_SAS = 10.*(-np.exp(-t / (0.05 * 3600.*(24.*2))) + np.exp(-t / (0.1 *3600.*(24.*2))))
        conc_SAS.assign(solute_amount_SAS)
        conc_SAS_arr.append(solute_amount_SAS)
        
        if dirichlet:
            bcs_conc = [DirichletBC(Q.sub(1), conc_SAS, 'on_boundary'),DirichletBC(Q.sub(2), conc_SAS, 'on_boundary')]
            #bcs_conc = []
        # update rhs   
        b_conc = assemble(L_conc)
        [bc.apply(A_conc,b_conc) for bc in bcs_conc]
        # Solve
        print("Solving diffusion equations...")
        solve(A_conc, c_.vector(), b_conc, 'gmres', 'ilu')
        cn.assign(c_.copy())
        #solve(A_conc, c_.vector(), b_conc, "gmres")
        c2 = c_.split()
        ### Compute the qunatities to measure the EPR effect
        c_healthy.append( assemble((phi0[0]*c2[0]+phi0[1]*c2[1]+phi0[2]*c2[2]) * dx(1) + (phi0[0]*c2[0]+phi0[1]*c2[1]+phi0[2]*c2[2]) * dx(2)  )    )
        c_edema.append( assemble((phi0_cytotoxic[0]*c2[0]+phi0_cytotoxic[1]*c2[1]+phi0_cytotoxic[2]*c2[2]) * dx(6)) + assemble((phi0_vasogenic[0]*c2[0]+phi0_vasogenic[1]*c2[1]+phi0_vasogenic[2]*c2[2]) * dx(3))    )
        c_tumor.append( assemble((phi0_tumor[0]*c2[0]+phi0_tumor[1]*c2[1]+phi0_tumor[2]*c2[2]) * dx(4))    )
        
        if save and (it == 4 or it == 12 or it == 24 or it == 48 or it == 7*24):
            print("Done, now saving.")
            
            c3 = c_.split()
            L2_project(Q, dx, c_, phi0, phi0_cytotoxic, phi0_vasogenic, phi0_tumor, ncomp, c1)
            c2 = c1.split()
            print("Max in ECS = " + str(max(c2[0].vector())))
            print("Min in ECS = " + str(min(c2[0].vector())))
            print("Max in PVSa = " + str(max(c2[1].vector())))
            print("Min in PVSa = " + str(min(c2[1].vector())))
            print("Max in PVSv = " + str(max(c2[2].vector())))
            print("Min in PVSv = " + str(min(c2[2].vector())))

            storage_cecs.write(c2[0], t) # store the ISF concentration
            storage_carteries.write(c2[1], t) # store the arterial concentration
            storage_cveins.write(c2[2], t) # store the venous concentration
            print("Mean ECS intrinsic concentration in tumor" + str(assemble(c3[0] * dx(4))))
            
        
        print("Mass tumor = " + str(c_tumor[-1]))
        print("Mass total = " +str(c_tumor[-1]+c_healthy[-1]+ c_tumor[-1]))
        print("Done, on to the next time step.")
        
        timevec.append(t)
        it += 1
        t+=dt
    
    if save:
    
        print("Done, now saving.")
            
        c3 = c_.split()
        
        L2_project(Q, dx, c_, phi0, phi0_cytotoxic, phi0_vasogenic, phi0_tumor, ncomp, c1)
        c2 = c1.split()
        print("Max in ECS = " + str(max(c2[0].vector())))
        storage_cecs.write(c2[0], t) # store the ISF concentration
        storage_carteries.write(c2[1], t) # store the arterial concentration
        storage_cveins.write(c2[2], t) # store the venous concentration

            
        # Storage
        
        storage_cecs.store_info()
        storage_carteries.store_info()
        storage_cveins.store_info()

        storage_cecs.close()
        storage_carteries.close()
        storage_cveins.close()

    return c_healthy, c_edema, c_tumor, conc_SAS_arr, timevec