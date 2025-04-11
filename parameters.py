import numbers
from itertools import combinations

from numpy import exp, pi, sqrt
from pint import Quantity, UnitRegistry

# Define units
ureg = UnitRegistry()
mmHg = ureg.mmHg
kg = ureg.kg
Pa = ureg.Pa
m = ureg.m
cm = ureg.cm
mm = ureg.mm
um = ureg.um
nm = ureg.nm
s = ureg.s
minute = ureg.min
mL = ureg.mL


# Dictionary defining various subsets of the compartments and interfaces which
# either share a parameter, or use the same functions to compute parameters.
SHARED_PARAMETERS = {
    "all": [
        "e",
        "pa",
        "pc",
        "pv",
        "a",
        "c",
        "v",
    ],
    "pvs": ["pa", "pc", "pv"],
    "csf": ["e", "pa", "pc", "pv"],
    "blood": ["a", "c", "v"],
    "large_vessels": ["a", "v"],
    "bbb": [
        ("pa", "a"),
        ("pc", "c"),
        ("pv", "v"),
    ],
    "aef": [("e", "pa"), ("e", "pc"), ("e", "pv")],
    "membranes": [
        ("pa", "a"),
        ("pc", "c"),
        ("pv", "v"),
        ("e", "pa"),
        ("e", "pc"),
        ("e", "pv"),
    ],
    "connected": [
        ("pa", "pc"),
        ("pc", "pv"),
        ("a", "c"),
        ("c", "v"),
    ],
    "connected_blood": [("a", "c"), ("c", "v")],
    "connected_pvs": [
        ("pa", "pc"),
        ("pc", "pv"),
    ],
    "disconnected": [
        ("e", "a"),
        ("e", "c"),
        ("e", "v"),
        ("pa", "pv"),
        ("pa", "c"),
        ("pa", "v"),
        ("pc", "a"),
        ("pc", "v"),
        ("pv", "a"),
        ("pv", "c"),
        ("a", "v"),
    ],
}

# Dictionary containing parameters with values found in literature, or values for which
# we have just assumed some value. All other parameters should be derived from these.
BASE_PARAMETERS = {
    "brain_volume": 1300.0e3 * mm**3,
    "brain_surface_area": 1.750e2 * mm**2,
    "diffusion_coefficient_free": {
        "inulin": 2.98e-6 * cm**2 / s,
        "amyloid_beta": 1.8e-4 * mm**2 / s,
        "gadobutrol": 3.8e-4 * mm**2/s # Valnes (2020) ADC
    },
    "diffusion_coefficient_GM": {
        "gadobutrol": 1.6e-4* mm**2/s, # Valnes (2020) ADC
    },
    "diffusion_coefficient_WM": {
        "gadobutrol": 1.7e-4* mm**2/s, # Valnes (2020) ADC
    },
    "pressure_boundary": {
        "e": 3.74 * mmHg,
        "pa": 4.74 * mmHg,
        "pv": 3.36 * mmHg,
        "a": 60.0 * mmHg,
        "v": 7.0 * mmHg,
        "c": 13.0 * mmHg,
    },
    "tortuosity": 1.7,
    "osmotic_pressure": {"blood": 20.0 * mmHg},
    "osmotic_pressure_fraction": {
        "csf": 0.2,  # Osmotic pressure computed as this constant * osmotic_pressure-blood
    },
    "osmotic_reflection": {
        "inulin": {"aef": 0.2, "connected_blood": 1.0, "bbb": 1.0, "connected_pvs": 0.0},
        "gadobutrol": {"aef": 0.0, "connected_blood": 1.0, "bbb": 1.0, "connected_pvs": 0.0},
    },
    "porosity": {"e": 0.2},
    "vasculature_volume_fraction": 0.0329,
    "vasculature_fraction": {"a": 0.21, "c": 0.33, "v": 0.46},
    "pvs_volume_fraction": 0.015,
    "viscosity": {"blood": 2.67e-3 * Pa * s, "csf": 7.0e-4 * Pa * s},
    "permeability": {"e": 2.0e-11 * mm**2},
    "hydraulic_conductivity": {
        "a": 1.234 * mm**3 * s / kg,
        # "c": 4.28e-4 * mm ** 3 * s / kg,
        "c": 3.3e-3 * mm**3 * s / kg,
        "v": 2.468 * mm**3 * s / kg,
        ("e", "a"): 9.1e-10 * mm / (Pa * s),
        ("e", "c"): 1.0e-10 * mm / (Pa * s),
        ("e", "v"): 2.0e-11 * mm / (Pa * s),
    },
    "surface_volume_ratio": {
        ("e", "a"): 3.92 / mm,
        ("e", "c"): 11.77 / mm,
        ("e", "v"): 3.92 / mm, # new values, 11.77 mm^2/mm^3 is given in "Morphometry of the human cerebral cortex microcirculation: General characteristics and space-related profiles" whereas the 1/3 ratio to apply to veins and arteries is assumed from computations in  
    },
    "flowrate": {"blood": 2.32 * mL / minute, "csf": 3.38 * mm**3 / minute},
    "pressure_drop": {
        ("a", "c"): 40.0 * mmHg,
        ("c", "v"): 13.0 * mmHg,
        ("pa", "pc"): 1.0 * mmHg,
        ("pc", "pv"): 0.25 * mmHg,
    },
    "resistance": {
        "e": 0.57 * mmHg / (mL / minute),
        #"pa": 1.14 * mmHg / (mL / minute),
        "pa": 1.68e-3 * mmHg / (mL / minute), # if PVS arteries stop before precapillaries 
        "pc": 32.24 * mmHg / (mL / minute),
        "pv": 1.75e-3 * mmHg / (mL / minute),
    },
    "resistance_interface": {
        ("e", "pa"): 0.57 * mmHg / (mL / minute),
        ("e", "pv"): 0.64 * mmHg / (mL / minute),
        ("pc", "c"): 125.31 * mmHg / (mL / minute),
    },
    "diameter": {"a": 38.0 * um, "c": 10.0 * um, "v": 38.0 * um},
    "solute_radius": {"inulin": 15.2e-7 * mm, 
                      "amyloid_beta": 0.9 * nm,
                      "gadobutrol": 1.0/2. *nm #  1.0 nm is the hydrodynamic diameter measured in "Taylor dispersion analysis for measurement of diffusivity and size of gadolinium-based contrast agents" for Gadovist.   
                      },
    ###################################################################
    # Related to permeability of BBB. Since this work is restricted to inulin,
    # only AEF is of interest.
    "membranes": {
        "layertype": {
            "glycocalyx": "fiber",
            "inner_endothelial_cleft": "pore",
            "endothelial_junction": "pore",
            "outer_endothelial_cleft": "pore",
            "basement_membrane": "fiber",
            "aef": "pore",
        },
        "thickness": {
            "glycocalyx": {
                "a": 400.0 * nm,
                "c": 250.0 * nm,
                "v": 100.0 * nm,
            },
            "inner_endothelial_cleft": 350.0 * nm,
            "endothelial_junction": 11.0 * mm,
            "outer_endothelial_cleft": 339.0 * nm,  # Total endothelial length 700nm
            "basement_membrane": {
                "a": 80.0 * nm,
                "c": 30.0 * nm,
                "v": 20.0 * nm,
            },
            "aef": 1000.0 * nm,
        },
        "elementary_radius": {
            "glycocalyx": 6.0 * nm,
            "inner_endothelial_cleft": 9.0 * nm,
            "endothelial_junction": {
                "a": 0.5 * nm,
                "c": 2.5 * nm,
                "v": 10.0 * nm,
            },
            "outer_endothelial_cleft": 9.0 * nm,
            "basement_membrane": {
                "a": 80.0 * nm,
                "c": 30.0 * nm,
                "v": 20.0 * nm,
            },
            "aef": {
                "a": 250.0 * nm,
                "c": 10.0 * nm,
                "v": 250.0 * nm,
            },
        },
        "fiber_volume_fraction": {"glycocalyx": 0.326, "basement_membrane": 0.5},
    },
}

PARAMETER_UNITS = {
    "permeability": "mm**2",
    "viscosity": "Pa * s",
    "porosity": "",
    "hydraulic_conductivity": "mm ** 2 / (Pa * s)",
    "convective_fluid_transfer": "1 / (Pa * s)",
    "osmotic_pressure": "Pa",
    "osmotic_reflection": "",
    "diffusive_solute_transfer": "1 / s",
    "convective_solute_transfer": "1 / (Pa * s)",
    "effective_diffusion": "mm**2 / s",
    "free_diffusion": "mm**2 / s",
    "hydraulic_conductivity_bdry": "mm / (Pa * s)",
    "pressure_boundaries": "Pa",
}


def get_base_parameters():
    """Return a copy of the BASE_PARAMETERS, containing parameter values
    explicitly found in literature. Based on these values, remaining parameters
    will be computed."""
    return {**BASE_PARAMETERS}


def get_shared_parameters():
    """Return a copy of the SHARED_PARAMETERS, defining various subsets of
    compartments and interfaces."""
    return {**SHARED_PARAMETERS}


def multicompartment_parameters(compartments, solute):
    base = get_base_parameters()
    distributed = distribute_subset_parameters(base)
    computed = compute_parameters(distributed, solute)
    converted = convert_to_units(computed, PARAMETER_UNITS)
    params = make_dimless(converted)
    symmetrized = symmetrize(
        params,
        compartments,
        "convective_fluid_transfer",
        "convective_solute_transfer",
        "diffusive_solute_transfer",
    )
    return symmetrized


def pvs(v):
    return f"p{v}"


def isquantity(x):
    return isinstance(x, Quantity) or isinstance(x, numbers.Complex)


def print_quantities(p, offset, depth=0):
    """Pretty printing of dictionaries filled with pint.Quantities"""
    format_size = offset - depth * 2
    for key, value in p.items():
        if isinstance(value, dict):
            print(f"{depth*'  '}{str(key)}")
            print_quantities(value, offset, depth=depth + 1)
        else:
            if isquantity(value):
                print(f"{depth*'  '}{str(key):<{format_size+1}}: {value:.3e}")
            else:
                print(f"{depth*'  '}{str(key):<{format_size+1}}: {value}")


def distribute_subset_parameters(base, subsets=None):
    """Take any parameter entry indexed by the name of some subset (e.g. 'blood'),
    and create a new entry for each of the compartments/interfaces included in the
    given subset."""
    if subsets is None:
        subsets = get_shared_parameters()
    extended = {}
    for param_name, param_value in base.items():
        if not isinstance(param_value, dict):
            extended[param_name] = param_value
        else:
            param_dict = {**param_value}  # Copy old values

            # Check if any of the entries refer to a subset of compartments...
            for idx, val in param_value.items():
                if idx in subsets:
                    # ... and insert one entry for each of the compartment in the subset.
                    for compartment in subsets[idx]:
                        param_dict[compartment] = val
            extended[param_name] = param_dict
    return extended


def make_dimless(params):
    """Converts all quantities to a dimless number."""
    dimless = {}
    for key, val in params.items():
        if isinstance(val, dict):
            dimless[key] = make_dimless(val)
        elif isinstance(val, Quantity):
            dimless[key] = val.magnitude
        else:
            dimless[key] = val
    return dimless


def convert_to_units(params, param_units):
    """Converts all quantities to the units specified by
    param_units."""
    converted = {}
    for key, val in params.items():
        if isinstance(val, dict):
            converted[key] = convert_to_units(val, param_units[key])
        elif isinstance(val, Quantity):
            converted[key] = val.to(param_units)
        else:
            converted[key] = val
    return converted


def symmetric(param, compartments):
    out = {}
    for i, j in combinations(compartments, 2):
        if (i, j) in param and (j, i) not in param:
            out[(i, j)] = param[(i, j)]
            out[(j, i)] = param[(i, j)]
        elif (j, i) in param and (i, j) not in param:
            out[(i, j)] = param[(j, i)]
            out[(j, i)] = param[(j, i)]
        else:
            raise KeyError(f"Either both or neither of {(i, j)} and  {(j, i)} exists in {param}")
    return out


def symmetrize(params, compartments, *args):
    out = {**params}
    for param in args:
        out[param] = symmetric(params[param], compartments)
    return out


def to_constant(param, *args):
    return tuple(param[x] for x in args)


def get_effective_diffusion(params, solute):
    Dfree = params["diffusion_coefficient_free"][solute]
    tortuosity = params["tortuosity"]
    return {key: Dfree / tortuosity**2 for key in SHARED_PARAMETERS["all"]}

def get_free_diffusion(params, solute):
    Dfree = params["diffusion_coefficient_free"][solute]
    return {key: Dfree for key in SHARED_PARAMETERS["all"]}


def get_porosities(params):
    phi = {"e": params["porosity"]["e"]}
    phi_B = params["vasculature_volume_fraction"]
    phi_PV = params["pvs_volume_fraction"]
    for vi in SHARED_PARAMETERS["blood"]:
        fraction_vi = params["vasculature_fraction"][vi]
        phi[vi] = fraction_vi * phi_B
        phi[pvs(vi)] = fraction_vi * phi_PV
    return phi


def get_viscosities(params):
    viscosities = params["viscosity"]
    return viscosities


def get_resistances(params):
    R = {**params["resistance"]}
    return R


def get_permeabilities(p):
    R = get_resistances(p)
    mu = p["viscosity"]
    k = {"e": p["permeability"]["e"]}
    K = p["hydraulic_conductivity"]
    length_area_ratio = R["e"] * k["e"] / mu["e"]
    for comp in SHARED_PARAMETERS["pvs"]:
        k[comp] = length_area_ratio * mu[comp] / R[comp]
    for comp in SHARED_PARAMETERS["blood"]:
        k[comp] = K[comp] * mu[comp]
    return {key: val.to("mm^2") for key, val in k.items()}


def get_hydraulic_conductivity(params):
    K_base = params["hydraulic_conductivity"]
    K = {}
    for vi in SHARED_PARAMETERS["blood"]:
        K[vi] = K_base[vi]

    k = get_permeabilities(params)
    mu = params["viscosity"]
    for j in SHARED_PARAMETERS["csf"]:
        K[j] = k[j] / mu[j]

    return K


def get_convective_fluid_transfer(params):
    T = {}
    for vi in SHARED_PARAMETERS["blood"]:
        V = params["brain_volume"]
        L_e_vi = params["hydraulic_conductivity"][("e", vi)]
        surface_ratio = params["surface_volume_ratio"][("e", vi)]
        T_e_vi = L_e_vi * surface_ratio

        if vi == "c":
            R_pc_c = params["resistance_interface"][("pc", "c")]
            R_e_pc = 1.0 / (T_e_vi * V) - R_pc_c
            T[(pvs(vi), vi)] = 1.0 / (V * R_pc_c)
            T[("e", pvs(vi))] = 1.0 / (V * R_e_pc)
        else:
            R_e_pvi = params["resistance_interface"][("e", pvs(vi))]
            R_pvi_vi = 1.0 / (T_e_vi * V) - R_e_pvi
            T[("e", pvs(vi))] = 1.0 / (V * R_e_pvi)            
            T[(pvs(vi), vi)] = 1.0 / (V * R_pvi_vi)

    # Compute connected transfer coefficients.
    for vi, vj in SHARED_PARAMETERS["connected_blood"]:
        V = params["brain_volume"]
        Q = params["flowrate"]
        dp = params["pressure_drop"]
        T[(vi, vj)] = compute_connected_fluid_transfer(V, Q["blood"], dp[(vi, vj)])
        T[(pvs(vi), pvs(vj))] = compute_connected_fluid_transfer(
            V,
            Q["csf"],
            dp[(pvs(vi), pvs(vj))],
        )
    for i, j in SHARED_PARAMETERS["disconnected"]:
        T[(i, j)] = 0.0 * 1 / (Pa * s)
        
    return {key: val.to(1 / (Pa * s)) for key, val in T.items()}


def compute_partial_fluid_transfer(brain_volume, resistance, total_transfer):
    new_resistance = 1.0 / (total_transfer * brain_volume) - resistance
    return 1.0 / (new_resistance * brain_volume)


def compute_connected_fluid_transfer(brain_volume, flow_rate, pressure_drop):
    return flow_rate / (pressure_drop * brain_volume)


def get_osmotic_pressure(params):
    pi_B = params["osmotic_pressure"]["blood"]
    csf_factor = params["osmotic_pressure_fraction"]["csf"]
    pi = {}
    for x in SHARED_PARAMETERS["all"]:
        if x in SHARED_PARAMETERS["blood"]:
            pi[x] = pi_B
        elif x in SHARED_PARAMETERS["csf"]:
            pi[x] = csf_factor * pi_B

    return pi


def get_osmotic_reflection(params, solute):
    sigma = distribute_subset_parameters(params["osmotic_reflection"], SHARED_PARAMETERS)[solute]
    for interface in SHARED_PARAMETERS["disconnected"]:
        sigma[interface] = 0.0
    return sigma


def get_convective_solute_transfer(params, solute):
    sigma = get_osmotic_reflection(params, solute)
    G = get_convective_fluid_transfer(params)
    return {ij: G[ij] * (1 - sigma[ij]) for ij in G}

"""
def diffusive_permeabilities(params,solute):
    P = {}
    # Permeability over membranes.
    for vi in SHARED_PARAMETERS["blood"]:
        d_vi = params["diameter"][vi]
        R_aef_vi = diffusive_resistance_aef(params, vi, solute)
        P[(pvs(vi), vi)] = 0.0 * mm / s
        P[("e", pvs(vi))] = 1.0 / (pi * d_vi * R_aef_vi)

    # Assume purely convection-driven transport between connected compartments.
    for i, j in SHARED_PARAMETERS["connected"]:
        P[(i, j)] = 0.0 * mm / s
    return {key: val.to("mm / s") for key, val in P.items()}
"""

def diffusive_resistance_aef(params, vessel, solute):
    D_free = params["diffusion_coefficient_free"][solute]
    membranes = params["membranes"]
    thickness = membranes["thickness"]["aef"]
    B_aef = membranes["elementary_radius"]["aef"][vessel]
    solute_radius = params["solute_radius"][solute]
    D_eff = diffusion_porous(D_free, solute_radius, B_aef)
    return resistance_aef(thickness, B_aef, D_eff)


def diffusion_porous(D_free: Quantity, solute_radius: Quantity, pore_radius: Quantity) -> Quantity:
    beta = solute_radius / pore_radius
    return D_free * (
        1.0 - 2.10444 * beta + 2.08877 * beta**3 - 0.094813 * beta**5 - 1.372 * beta**6
    )


def resistance_aef(layer_thickness, pore_radius, effective_diffusion):
    return layer_thickness / (2.0 * pore_radius * effective_diffusion)


def get_diffusive_solute_transfer(params,solute):
    P = diffusive_permeabilities(params,solute)
    surf_volume_ratio = params["surface_volume_ratio"]
    L = {}
    for vi in SHARED_PARAMETERS["blood"]:
        L[("e", pvs(vi))] = P[("e", pvs(vi))] * surf_volume_ratio[("e", vi)]
        L[(pvs(vi), vi)] = P[(pvs(vi), vi)] * surf_volume_ratio[("e", vi)]
    for i, j in [*SHARED_PARAMETERS["connected"], *SHARED_PARAMETERS["disconnected"]]:
        L[(i, j)] = 0.0 * 1 / s
    return {key: val.to(1 / (s)) for key, val in L.items()}


def get_boundary_hydraulic_permeabilities(p):
    dp = p["pressure_drop"]
    Q = p["flowrate"]["blood"]
    Ra = dp[("a", "c")] / Q

    Rpa = 2 * p["resistance"]["pa"]
    S = p["brain_surface_area"]
    V = p["brain_volume"]
    gamma = get_convective_fluid_transfer(p)

    L_bdry = {}
    L_bdry["e"] = 1.0 / (2 * Rpa * S)
    L_bdry["pa"] = gamma[("e", "pa")] * V / S
    L_bdry["a"] = 1.0 / (Ra * S)
    return {key: value.to("mm / (Pa * s)") for key, value in L_bdry.items()}


def get_arterial_inflow(params):
    B = params["flowrate"]["blood"]
    S = params["ratbrain_surface_area"]
    return B / S


def compute_parameters(params, solute):
    return {
        # "csf_volume": params["csf_volume_fraction"] * params["brain_volume"],
        # "csf_renewal_rate": params["csf_renewal_rate"],
        "permeability": get_permeabilities(params),
        "viscosity": params["viscosity"],
        "porosity": get_porosities(params),
        "hydraulic_conductivity": get_hydraulic_conductivity(params),
        "convective_fluid_transfer": get_convective_fluid_transfer(params),
        "osmotic_pressure": get_osmotic_pressure(params),
        "osmotic_reflection": get_osmotic_reflection(params, solute),
        "effective_diffusion": get_effective_diffusion(params, solute),
        "free_diffusion": get_free_diffusion(params, solute),
        "diffusive_solute_transfer": get_diffusive_solute_transfer(params,solute),
        "convective_solute_transfer": get_convective_solute_transfer(params, solute),
        "hydraulic_conductivity_bdry": get_boundary_hydraulic_permeabilities(params),
        "pressure_boundaries": params["pressure_boundary"],
    }


######################################
# GENERALIZED DIFFUSIVE PERMEABILITIES
######################################
def diffusive_permeabilities(params, solute):
    bbb_layers = [
        "glycocalyx",
        "inner_endothelial_cleft",
        "endothelial_junction",
        "outer_endothelial_cleft",
        "basement_membrane",
    ]
    P = {}
    for vi in SHARED_PARAMETERS["blood"]:
        dvi = params["diameter"][vi]
        Rvi = diffusive_resistance_membrane_layer(params, solute, vi)
        R_bbb = sum([Rvi[layer] for layer in bbb_layers])
        R_aef = Rvi["aef"]
        if solute != "inulin" and solute != "gadobutrol":
            P[(pvs(vi), vi)] = 1.0 / (pi * dvi) / R_bbb  
        else: 
            P[(pvs(vi), vi)] =  0.0 * mm / s
        P[("e", pvs(vi))] = 1.0 / (pi * dvi) / R_aef

    # Assume purely convection-driven transport between connected compartments.
    for i, j in SHARED_PARAMETERS["connected"]:
        P[(i, j)] = 0.0 * mm / s
    return {key: val.to("mm / s") for key, val in P.items()}


def diffusive_resistance_membrane_layer(params, solute, vessel):
    membranes = distribute_membrane_params(params["membranes"])
    D_free = params["diffusion_coefficient_free"][solute]
    solute_radius = params["solute_radius"][solute]
    R = {}
    for layer, layertype in membranes["layertype"].items():
        if layertype == "fiber":
            Vf = membranes["fiber_volume_fraction"][layer][vessel]
            r = membranes["elementary_radius"][layer][vessel]
            D_eff = diffusion_fibrous(D_free, solute_radius, r, Vf)
        elif layertype == "pore":
            r = membranes["elementary_radius"][layer][vessel]
            D_eff = diffusion_porous(D_free, solute_radius, r)
        else:
            raise ValueError(f"layertype should be 'pore' or 'fiber', got {layertype}")

        thickness = membranes["thickness"][layer][vessel]

        R[layer] = solute_resistance_layer(thickness, r, D_eff)
    return R


def distribute_membrane_params(membranes):
    """Take the membrane-parameter dictionary, and create a new dictionary with
    keys for each of the various vascular compartments, e.g.
    membranes[thickness][aef] -> membranes[thickness][aef][vi] for vi in blood."""
    unpacked = {}
    for param_name, param_values in membranes.items():
        # Do not separate layertype between different vessels
        if param_name == "layertype":
            unpacked[param_name] = param_values
            continue

        unpacked[param_name] = {}
        for layer, layer_value in param_values.items():
            if not isinstance(layer_value, dict):
                unpacked[param_name][layer] = {vi: layer_value for vi in SHARED_PARAMETERS["blood"]}
            else:
                unpacked[param_name][layer] = {**layer_value}
    return unpacked


def diffusion_fibrous(D_free, solute_radius, fiber_radius, fiber_volume_fraction):
    return D_free * exp(-sqrt(fiber_volume_fraction) * (1.0 + solute_radius / fiber_radius))


def solute_resistance_layer(layer_thickness, elementary_radius, effective_diffusion):
    return layer_thickness / (2.0 * elementary_radius * effective_diffusion)

"""
def get_diffusive_solute_transfer(params, solute):
    P = diffusive_permeabilities(params, solute)
    surf_volume_ratio = params["surface_volume_ratio"]
    L = {}
    for vi in SHARED_PARAMETERS["blood"]:
        L[("e", f"pvs_{vi}")] = P[("e", f"pvs_{vi}")] * surf_volume_ratio[("e", vi)]
        L[(f"pvs_{vi}", vi)] = P[(f"pvs_{vi}", vi)] * surf_volume_ratio[("e", vi)]
    return {key: val.to(1 / (s)) for key, val in L.items()}
"""
    
def dict_reference():
    dictio_ref = {
    "brain_volume": ""
    
    } 
    
    
def save_table_parameters(p, offset, depth = 0):
    """This function saves the table of the parameters. 
    For each parameter, we will have the LaTeX symbol, the meaning, the units, the value, 
    the reference to an article or the mention that it was estimated from data or assumed 
    to be physically relevant. 
    
    """
    
    format_size = offset - depth * 2
    for key, value in p.items():
        print(key)

        if isinstance(value, dict):
            print(f"{depth*'  '}{str(key)}")
            print_quantities(value, offset, depth=depth + 1)
        else:
            if isquantity(value):
                print(f"{depth*'  '}{str(key):<{format_size+1}}: {value:.3e}")
            else:
                print(f"{depth*'  '}{str(key):<{format_size+1}}: {value}")



if __name__ == "__main__":
    solute = "gadobutrol"
    base = get_base_parameters()
    
    extended = distribute_subset_parameters(base, SHARED_PARAMETERS)
    params = compute_parameters(extended,solute)
    #save_table_parameters(base, 40)
    """
    print_quantities(base, 40)
    print("=" * 60)
    print_quantities(params, 40)
    print("=" * 60)
    """
    print_quantities(convert_to_units(params, PARAMETER_UNITS), 40)
