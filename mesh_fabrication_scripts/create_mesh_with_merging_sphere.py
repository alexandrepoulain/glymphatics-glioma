#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import SVMTK as svmtk

from convert_to_dolfin_mesh import write_mesh_to_xdmf,write_xdmf_to_h5

def create_volume_mesh(stlfile_white, stlfile_pial, ventricles_stl, output, mixed, resolution = 16, remove_ventricle = True):
    
    # load the surface
    surface_wh = svmtk.Surface(stlfile_white)
    surface_pial = svmtk.Surface(stlfile_pial)
    ventricles = svmtk.Surface(ventricles_stl)
    # create surface for sphere inside white matter
    # Center coordinates    
    #x, y, z = -25.,7,13.5
    #x, y, z = -23,-51,8 # surface tumor
    #x,y,z = -22,-41,20
    x,y,z = -30, 31, 20 # front up
    #x,y,z = -30,-10,23 # center
    
    # Sphere radiuses 
    r1, r2 = 25,10
    # Edge length 
    edge_length = 1.
    # Create spheres 
    sphere1 = svmtk.Surface()
    sphere1.make_sphere(x,y,z,r1,0.5)
    
    sphere2 = svmtk.Surface()
    sphere2.make_sphere(x,y,z,r2,1.)
    
    
    
    
    # create map with tags
    # inside first, ouside second
    smap = svmtk.SubdomainMap()
    if mixed:
        smap.add("100000",1) # inside surface 1 and ouside surface 2 marked as 1
        smap.add("110000",2)
        smap.add("111100",3) # edema
        smap.add("101100",6)
        smap.add("111110",4) # tumor
        smap.add("101110",4) # tumor
        smap.add("111111",5) # ventricle
        # generate volume mesh
        domain = svmtk.Domain([surface_pial,surface_wh, sphere1,sphere1,sphere2, ventricles], smap)
    else:
        smap.add("10000",1) # inside surface 1 and ouside surface 2 marked as 1
        smap.add("11000",2)
        smap.add("11100",3) # edema
        smap.add("10100",3)
        smap.add("11110",4) # tumor
        smap.add("10110",4) # tumor
        smap.add("11111",5) # ventricle
        # generate volume mesh
        domain = svmtk.Domain([surface_pial,surface_wh,sphere1,sphere2, ventricles], smap)
    
    
    
    
    
    domain.create_mesh(resolution)
    
    if remove_ventricle:
        domain.remove_subdomain(5)
        
    # Write the mesh to the output file
    domain.save(output)
    
if __name__ == "__main__":

    stlfile_wh = "white.stl"
    stlfile_pial = "pial.stl"
    stl_ventricle = "ventricles.stl"
    
    mixed = True
    
    outfile = "left_hemi.mesh"
    res = 32
    
    
    create_volume_mesh(stlfile_wh, stlfile_pial, stl_ventricle, outfile, mixed, res)
    
    meshfile = outfile
    hdf5file = "../data/mesh/synthetic_mesh_" +str(res)+".h5"
    xdmfdir = "tmp"

    write_mesh_to_xdmf(meshfile, xdmfdir) 
    write_xdmf_to_h5(xdmfdir, hdf5file) 






