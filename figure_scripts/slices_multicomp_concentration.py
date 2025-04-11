import numpy as np
import pyvista as pv

def save_slice(reader, path_article, path_folder, time_step, orientation, origin, comp, png_yes_or_no =False):
    """ This function creates a slice of the data then plot it and save it 
    """
    if orientation == "horizontal":
        normal=[0,0,1]
        indica = "hor" 
    else:
        normal = [-0.9662364049700265,-0.23938059122944402,0.09531076672255342]
        indica = "ver"
        
    reader.set_active_time_value(time_step*3600)
    data = reader.read()
    #print(data.active_scalars_info)
    #exit(0)
    #data.set_active_scalars("f_123-1")


    if orientation == "horizontal":
        p = pv.Plotter(window_size=[1000, 900])
        print(p.camera.position)
        
    else:
        p = pv.Plotter(window_size=[900, 800])
    
    # see in the correct direction
    if orientation == "horizontal":
        rot = data.rotate_z(-10, inplace=False)
        clipped = rot.clip(normal= normal, origin= origin, invert = True)
        if time_step < 162:
            p.add_mesh(clipped, show_scalar_bar = False, clim = [0, 0.1])
        else:
            p.add_mesh(clipped, show_scalar_bar = False, clim = [0, 0.01])
        p.view_xy()
        p.camera.zoom(.85)
        
        p.add_scalar_bar(vertical=True, fmt= "%4.4f",label_font_size=35, bold = True, title = "Concentration (in nmol/mm³)", title_font_size = 35, position_x = 0.7, position_y = 0.3, width = 0.11, height = 0.6)
    else:
        normal = 1*np.array(normal)
        clipped = data.clip(normal= normal, origin= origin, invert = True)
        if time_step < 162:
            p.add_mesh(clipped, show_scalar_bar = False,clim = [0, 0.1])
        else:
            p.add_mesh(clipped, show_scalar_bar = False,clim = [0, 0.01])
        p.view_vector(normal)
        p.camera.zoom(1.2)
        p.add_scalar_bar(vertical=False, fmt= "%4.4f",label_font_size=35, bold = True, title = "Concentration (in nmol/mm³)", title_font_size = 35, width = 0.6, height = 0.11,position_x = 0.3)
        
    filename = path_article + path_folder + comp + "_" + str(time_step) + "_H_" + str(indica) + ".eps"
    p.save_graphic(filename)
    
    filename = path_article + path_folder + comp + "_" + str(time_step) + "_H_" + str(indica) + ".png"
    p.screenshot(filename)


if __name__ == "__main__":

    exp_name= "reference"
    mixed_edema = True
    path_article = "../article-glymphatics-tumor/images/"+str(exp_name) + "/"

    macro_or_micro = "macro"
    
    origin = [-40, 33, 23] # center of tumor (other choice #origin = [-50, 33, 23])
    origin = [-30, 31, 20]
    
    png_yes_or_no = True # Do you want to save in png

    ### ECS
    if macro_or_micro == "macro": 
        path_folder = "ecs_macro/"
    elif macro_or_micro == "micro":
        path_folder = "ecs_micro/"
        
    comp = "ecs"
    xdmf_file = "../results/" + exp_name+"/ecs " + macro_or_micro + "/visual.xdmf"
    reader = pv.get_reader(xdmf_file)
    
    save_slice(reader, path_article, path_folder, 0, "horizontal", origin,comp, png_yes_or_no)
    save_slice(reader, path_article, path_folder, 0, "vertical", origin,comp,png_yes_or_no)

    save_slice(reader, path_article, path_folder, 4, "horizontal", origin,comp,png_yes_or_no)
    save_slice(reader, path_article, path_folder, 4, "vertical", origin,comp,png_yes_or_no)

    save_slice(reader, path_article, path_folder, 12, "horizontal", origin,comp,png_yes_or_no)
    save_slice(reader, path_article, path_folder, 12, "vertical", origin,comp,png_yes_or_no)

    save_slice(reader, path_article, path_folder, 24, "horizontal", origin,comp,png_yes_or_no)
    save_slice(reader, path_article, path_folder, 24, "vertical", origin,comp,png_yes_or_no)
    
    save_slice(reader, path_article, path_folder, 48, "horizontal", origin,comp,png_yes_or_no)
    save_slice(reader, path_article, path_folder, 48, "vertical", origin,comp,png_yes_or_no)
    
    
    save_slice(reader, path_article, path_folder, 168, "horizontal", origin,comp,png_yes_or_no)
    save_slice(reader, path_article, path_folder, 168, "vertical", origin,comp,png_yes_or_no)

    save_slice(reader, path_article, path_folder, 336, "horizontal", origin,comp,png_yes_or_no)
    save_slice(reader, path_article, path_folder, 336, "vertical", origin,comp,png_yes_or_no)

    ### PVS arteries
    path_folder = "pa " + macro_or_micro + "/"
    comp = "pa"
    xdmf_file = "../results/"+ exp_name+"/arteries " + macro_or_micro + "/visual.xdmf"
    reader = pv.get_reader(xdmf_file)

    save_slice(reader, path_article, path_folder, 0, "horizontal", origin,comp,png_yes_or_no)
    save_slice(reader, path_article, path_folder, 0, "vertical", origin,comp,png_yes_or_no)

    save_slice(reader, path_article, path_folder, 12, "horizontal", origin,comp,png_yes_or_no)
    save_slice(reader, path_article, path_folder, 12, "vertical", origin,comp,png_yes_or_no)

    save_slice(reader, path_article, path_folder, 24, "horizontal", origin,comp,png_yes_or_no)
    save_slice(reader, path_article, path_folder, 24, "vertical", origin,comp,png_yes_or_no)

    save_slice(reader, path_article, path_folder, 47, "horizontal", origin,comp,png_yes_or_no)
    save_slice(reader, path_article, path_folder, 47, "vertical", origin,comp,png_yes_or_no)

    ### PVS veins
    path_folder = "pv " + macro_or_micro + "/"
    comp = "pv"
    xdmf_file =  "../results/" +exp_name+"/veins " + macro_or_micro + "/visual.xdmf"
    reader = pv.get_reader(xdmf_file)
    save_slice(reader, path_article, path_folder, 0, "horizontal", origin,comp,png_yes_or_no)
    save_slice(reader, path_article, path_folder, 0, "vertical", origin,comp,png_yes_or_no)

    save_slice(reader, path_article, path_folder, 12, "horizontal", origin,comp,png_yes_or_no)
    save_slice(reader, path_article, path_folder, 12, "vertical", origin,comp,png_yes_or_no)

    save_slice(reader, path_article, path_folder, 24, "horizontal", origin,comp,png_yes_or_no)
    save_slice(reader, path_article, path_folder, 24, "vertical", origin,comp,png_yes_or_no)

    save_slice(reader, path_article, path_folder, 47, "horizontal", origin,comp,png_yes_or_no)
    save_slice(reader, path_article, path_folder, 47, "vertical", origin,comp,png_yes_or_no)
        
