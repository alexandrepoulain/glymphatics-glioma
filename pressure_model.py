"""
This fil contains the pressure system
"""
from dolfin import *
from ts_storage import TimeSeriesStorage


class K_multcomp(UserExpression):
    """
    Defines the permeability locally in each region
    """
    def __init__(self, subdomains, k0, k1, k2, k3, **kwargs):
        """
        Parameters:
           - subdomains (): subdomains of the mesh for the different regions
           - k0, k1, k2, k3 (float): permeability coefficient for each subregions
        """
        super().__init__(**kwargs)
        self.subdomains = subdomains
        self.k0 = k0
        self.k1 = k1 # cytotoxic
        self.k2 = k2 # vasogenic
        self.k3 = k3 # tumor
        
    def value_shape(self):
        return ()
        
    def eval_cell(self, values, x, cell):
        if self.subdomains[cell.index] == 1 or self.subdomains[cell.index] == 2:
            values[0] = self.k0
    
        if self.subdomains[cell.index] == 6:
            values[0] = self.k1

        if self.subdomains[cell.index] == 3:
            values[0] = self.k2
            
        if self.subdomains[cell.index] == 4:
            values[0] = self.k3
    def value_shape(self):
        return ()

def pressure_model(Q, p, q, dx, ds, Kappa_f, Kappa_f_cytotoxic, Kappa_f_vasogenic, Kappa_f_tumor, gamma, gamma_cytotoxic, gamma_vasogenic, gamma_tumor, gamma_disrupt_veins, p_CSF, p_pial_pa, p_veins, gamma_pial_pa, gamma_pial_pv, gamma_pial_e, ncomp, results_path, mesh, VV,  save_pressure =False, dirichlet = True):
    """This function defines and runs the pressure model
    Parameters:
       - Q (mixed FE space): the finite element spac for the pressure fields.
       - p,q (test and trial functions): test and trial functions for the variational pb.
       - dx, ds (Measures): volume and surface measures defined by fenics
       - Kappa_f,  Kappa_f_cytotoxic, Kappa_f_vasogenic, Kappa_f_tumor (array of floats): permeability coefficient for each region and each compartment.
       - gamma, gamma_cytotoxic, gamma_vasogenic, gamma_tumor (nd arrays): fluid transfer coefficients for each region.
       - gamma_disrupt_veins (float): fluid transfer between veins and vPVS.
       - p_CSF, p_pial_pa, p_veins (floats): CSF pressure in SAS, CSF pressure at the pial surface for aPVS, blood pressure in veins (all in Pa).
       - gamma_pial_pa, gamma_pial_pv, gamma_pial_e (floats): permeabilities at the pial surface.
       - ncomp (int): number of compartments
       - results_path (Path): path to the results folder.
       - mesh (Mesh): mesh.
       - VV (Fe space): finite element space for one pressure field.
       - save_pressure (bool): decide if you want to save the pressure fields
       - dirichlet (bool): decides if you want to use dirichlet BC or Robin BC for PVS boundaries.
    Returns: 
       - p (Q function): the pressure fields for each compartment.
    """
    # Init fluid model 
    F=0
    # Constructing the variational problem
    for i in range(ncomp):
        F += Kappa_f[i]*inner(grad(p[i]),grad(q[i]))*dx(1)
        F += Kappa_f[i]*inner(grad(p[i]),grad(q[i]))*dx(2)
        F += Kappa_f_cytotoxic[i]*inner(grad(p[i]),grad(q[i]))*dx(6)
        F += Kappa_f_vasogenic[i]*inner(grad(p[i]),grad(q[i]))*dx(3)
        F += Kappa_f_tumor[i]*inner(grad(p[i]),grad(q[i]))*dx(4)
        # transfer between compartments
        for j in range(ncomp):
            if i != j:
                F -=  Constant(gamma[i][j])*inner((p[j]-p[i]),q[i])*dx(1)
                F -=  Constant(gamma[i][j])*inner((p[j]-p[i]),q[i])*dx(2)
                F -=  Constant(gamma_cytotoxic[i][j])*inner((p[j]-p[i]),q[i])*dx(6)
                F -=  Constant(gamma_vasogenic[i][j])*inner((p[j]-p[i]),q[i])*dx(3)
                F -=  Constant(gamma_tumor[i][j])*inner((p[j]-p[i]),q[i])*dx(4)
            # addition of the fluid transfer between veins and vPVS (in tumor and vasogenic edema)
            if i == 2:
                F -=  Constant(gamma_disrupt_veins)*inner((Constant(p_veins)-p[i]), q[i])*dx(3)
                F -=  Constant(gamma_disrupt_veins)*inner((Constant(p_veins)-p[i]), q[i])*dx(4)
            
    # Robin Boundary conditions
    F -= gamma_pial_e*(Constant(p_CSF)*q[0]*ds - p[0]*q[0]*ds)

    # Dirichlet Boundary conditions on PVS surfaces
    if dirichlet:
        bcs = [DirichletBC(Q.sub(1), Constant(p_pial_pa), 'on_boundary'), DirichletBC(Q.sub(2), Constant(p_CSF), 'on_boundary')]
    else:
        F -= gamma_pial_pa*(Constant(p_pial_pa)*q[1]*ds - p[1]*q[1]*ds)
        F -= gamma_pial_pv*(Constant(p_CSF)*q[2]*ds - p[2]*q[2]*ds)
        bcs = []
        
    # Assembling 
    print("Assembling pressure equations...")
    A = assemble(lhs(F))
    b = assemble(rhs(F))
    [bc.apply(A,b) for bc in bcs]
    p_ = Function(Q)
    print("Done")

    # Solving pressure equations
    print("Solving pressure system...", sep="")
    solve(A, p_.vector(), b,"cg", "hypre_amg")
    print("Done.")
    
    p_new = p_.split(True)
    p_0 = Function(VV)
    assign(p_0,p_new[0])
    p_1 = Function(VV)
    assign(p_1,p_new[1])
    p_2 = Function(VV)
    assign(p_2,p_new[2])

    # Saving pressure fields
    if save_pressure:
        print("Saving pressure fields...")
        storage_pe = TimeSeriesStorage("w", results_path, mesh=mesh, V=VV, name="pressure ecs")
        storage_ppa = TimeSeriesStorage("w", results_path, mesh=mesh, V=VV, name="pressure PVS arteries")
        storage_ppv = TimeSeriesStorage("w", results_path, mesh=mesh, V=VV, name="pressure PVS veins")
        
        storage_pe.write(p_0, 0.)
        storage_ppa.write(p_1, 0.)
        storage_ppv.write(p_2, 0.)
        storage_pe.store_info()
        storage_ppa.store_info()
        storage_ppv.store_info()

        storage_pe.close()
        storage_ppa.close()
        storage_ppv.close()
        print("Done.")

    return p_new
