"""This file contains the code to generate the pressure fields images

"""
import pyvista as pv
import numpy as np

def save_slice_pressure(reader, path_article, path_folder, time_step, orientation, comp, origin, png_yes_or_no =False):
    """This function creates a slice of the data then plot it and save it 
    """
    
    if orientation == "horizontal":
        normal=[0,0,1]
        indica = "hor" 
    else:
        normal = [1,0.25,0]
        indica = "ver"
        
    data = reader.read()

    if orientation == "horizontal":
        p = pv.Plotter(window_size=[850, 900])
    else:
        p = pv.Plotter(window_size=[900, 800])
    sargs = dict(height=0.25, vertical=True, position_x=0.05, position_y=0.05)

    # see in the correct direction
    if orientation == "horizontal":
        rot = data.rotate_z(-10, inplace=False)
        clipped = rot.clip(normal= np.array(normal), origin= origin)
        
        p.add_mesh(clipped, show_scalar_bar = False, cmap="jet",clim = [1334, 1336])
        p.view_xy()
        p.camera.zoom(1.)
        p.add_scalar_bar(vertical=True, fmt= "%4.2f",label_font_size=30, bold = True, title = "Pressure (in Pa)", title_font_size = 30, position_x = 0.77, width = 0.11, height = 0.6)
        disk_mesh = pv.Disc(center = [origin[0],origin[1], origin[2]+1], inner=10-0.2, outer=10+0.2,  c_res=50, normal = normal)
        p.add_mesh(disk_mesh,show_edges=True, line_width=7, edge_color="red")
        disk_mesh = pv.Disc(center = [origin[0],origin[1], origin[2]+1], inner=20-0.2, outer=20+0.2,  c_res=50, normal = normal)
        p.add_mesh(disk_mesh,show_edges=True, line_width=10, edge_color ="yellow")
    else:
        normal = -1*np.array(normal)
        clipped = data.clip(normal= normal, origin= origin, invert = True)
        if comp == "ecs":
            p.add_mesh(clipped, show_scalar_bar = False, cmap="jet", clim = [1334, 1337])
        elif comp == "PVS arteries":
            p.add_mesh(clipped, show_scalar_bar = False, cmap="jet", clim = [1336, 1338])
        elif comp == "PVS veins":
            p.add_mesh(clipped, show_scalar_bar = False, cmap="jet", clim = [1333, 1335])
        p.view_vector(normal)
        p.camera.zoom(1.2)
        p.add_scalar_bar(vertical=False, fmt= "%4.2f",label_font_size=30, bold = True, title = "Pressure (in Pa)", title_font_size = 30, width = 0.6, height = 0.11,position_x = 0.3)
        disk_mesh = pv.Disc(center = [origin[0],origin[1], origin[2]], inner=10-0.2, outer=10+0.2,  c_res=50, normal = normal)
        p.add_mesh(disk_mesh,show_edges=True, line_width=7,edge_color="red")
        disk_mesh = pv.Disc(center = [origin[0] + normal[0],origin[1]+ normal[1], origin[2]+ normal[2]] , inner=20-0.2, outer=20+0.2,  c_res=50, normal = normal)
        p.add_mesh(disk_mesh,show_edges=True, line_width=10,edge_color ="yellow")
    
    filename = path_article + path_folder + str(comp) + "_pressure_" + str(indica) + ".eps"
    p.save_graphic(filename)
    
    filename = path_article + path_folder + str(comp) + "_pressure_" + str(indica) + ".png"
    p.screenshot(filename)



if __name__ == "__main__":
    
    exp_name = "mixed_edema" # "mixed_edema", "vasogenic", "nonvasogenic" or "reference"
    mixed_edema = True
    
    path_article = "../article-glymphatics-tumor/images/"

    path_folder = exp_name+"/"
    
    origin = [-30, 31, 20] # tumor center
    
    png_yes_or_no = True # Do you want to save in png
    
    comp = "ecs"
    xdmf_file = "../results/" + exp_name+"/pressure "+comp+"/visual.xdmf"
    reader = pv.get_reader(xdmf_file)
    save_slice_pressure(reader, path_article, path_folder, 0, "horizontal", comp, origin,png_yes_or_no)
    save_slice_pressure(reader, path_article, path_folder, 0, "vertical", comp, origin,png_yes_or_no)
    
    comp = "PVS arteries"
    xdmf_file = "../results/" + exp_name+"/pressure "+comp+"/visual.xdmf"
    reader = pv.get_reader(xdmf_file)
    save_slice_pressure(reader, path_article, path_folder, 0, "horizontal", comp, origin,png_yes_or_no)
    save_slice_pressure(reader, path_article, path_folder, 0, "vertical", comp, origin,png_yes_or_no)
    
    comp = "PVS veins"
    xdmf_file = "../results/" + exp_name+"/pressure "+comp+"/visual.xdmf"
    reader = pv.get_reader(xdmf_file)
    save_slice_pressure(reader, path_article, path_folder, 0, "horizontal", comp, origin,png_yes_or_no)
    save_slice_pressure(reader, path_article, path_folder, 0, "vertical", comp, origin,png_yes_or_no)
    
    
    
