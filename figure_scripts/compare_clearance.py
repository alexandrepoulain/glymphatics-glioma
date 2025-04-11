#!/usr/bin/env python
# coding: utf-8

# Script to compare clearance of solute
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fig_paper = True # if false it does not save the images and add the titles

plt.rcParams['font.size'] = '18'

# import data
dfref = pd.read_csv("../results/reference/clearance_solute.csv")
dfvaso = pd.read_csv("../results/vasogenic/clearance_solute.csv")
dfnonvaso = pd.read_csv("../results/nonvasogenic/clearance_solute.csv")
dfmixed = pd.read_csv("../results/mixed_edema/clearance_solute.csv")

### Compute SAS concentration
nb_days = 14
T = (3600.)*24*nb_days
tt = np.linspace(0,T,1000)

y = 10.*(-np.exp(-tt / (0.05 * 3600.*(24.*2))) + np.exp(-tt / (0.1 *3600.*(24.*2))))
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(tt/3600,y, linewidth = 2)
ax.set_yscale('log')
plt.ylabel(r"Concentration in nmol/mm$^3$")
plt.xlabel("Time in hours")
plt.tight_layout()
if fig_paper:
    plt.savefig('../article-glymphatics-tumor/images/SAS_conc_14days_log.png')
    plt.close()
else:
    plt.title("Concentration in SAS")
    plt.show()


nb_days = 14
T = (3600.)*24*nb_days
tt = np.linspace(0,T,round(T/3600))

y = 10.*(-np.exp(-tt / (0.05 * 3600.*(24.*2))) + np.exp(-tt / (0.1 *3600.*(24.*2))))

plt.plot(tt/3600,y, linewidth = 2)
plt.ylabel(r"Concentration in nmol/mm$^3$")
plt.xlabel("Time in hours")
plt.tight_layout()
if fig_paper:
    plt.savefig('../article-glymphatics-tumor/images/SAS_conc_14days.png')
    plt.close()
else:
    plt.title("Concentration in SAS")
    plt.show()


### check total mass clearance
ref_tot = dfref["massin_healthy"].to_numpy() + dfref["massin_edema"].to_numpy() + dfref["massin_tumor"].to_numpy()
vaso_tot = dfvaso["massin_healthy"].to_numpy() + dfvaso["massin_edema"].to_numpy() + dfvaso["massin_tumor"].to_numpy()
nonvaso_tot = dfnonvaso["massin_healthy"].to_numpy() + dfnonvaso["massin_edema"].to_numpy() + dfnonvaso["massin_tumor"].to_numpy()
mixed_tot = dfmixed["massin_healthy"].to_numpy() + dfmixed["massin_edema"].to_numpy() + dfmixed["massin_tumor"].to_numpy()

plt.plot(ref_tot[:24], linewidth = 2)
plt.plot(vaso_tot[:24], linewidth = 2)
plt.plot(nonvaso_tot[:24], linewidth = 2)
plt.plot(mixed_tot[:24], linewidth = 2)
plt.legend(["reference", "vasogenic", "nonvasogenic", "mixed"])
plt.ylabel("Mass in nmol")
plt.xlabel("Time in hours")
plt.tight_layout()
if fig_paper:
    plt.savefig('../article-glymphatics-tumor/images/total_clearance_mixed.png')
    plt.close()
else:
    plt.title("Clearance of solute in brain")
    plt.show()

plt.plot(tt[47:]/3600,ref_tot[47:], linewidth = 2)
plt.plot(tt[47:]/3600,vaso_tot[47:], linewidth = 2)
plt.plot(tt[47:]/3600,nonvaso_tot[47:], linewidth = 2)
plt.plot(tt[47:]/3600,mixed_tot[47:], linewidth = 2)
plt.legend(["reference", "vasogenic", "nonvasogenic", "mixed"])
plt.ylabel("Mass in nmol")
plt.xlabel("Time in hours")
plt.gca().set_ylim(bottom=0)
plt.tight_layout()
if fig_paper:
    plt.savefig('../article-glymphatics-tumor/images/total_clearance_all_days_mixed.png')
    plt.close()
else:
    plt.title("Clearance of solute in brain")
    plt.show()


### Check clearance in tumor
dfref["massin_tumor"].plot.line(color = "blue", label = "reference")
dfvaso["massin_tumor"].plot.line(color = "orange", label = "vasogenic")
dfnonvaso["massin_tumor"].plot.line(color = "green",   label = "nonvasogenic")
dfmixed["massin_tumor"].plot.line(color = "red",   label = "mixed")

plt.legend()
plt.ylabel("Mass in nmol")
plt.xlabel("Time in H")
plt.tight_layout()
if fig_paper:
    plt.savefig('../article-glymphatics-tumor/images/tumor_clearance_mixed.png')
    plt.close()
else:
    plt.title("Clearance in tumor")
    plt.show()

### Check clearance in edema
dfref["massin_edema"].plot.line(color = "blue", linestyle="-.", label = "reference")
dfvaso["massin_edema"].plot.line(color = "orange", linestyle="-.", label = "vasogenic")
dfnonvaso["massin_edema"].plot.line(color = "green", linestyle="-.",label = "nonvasogenic")
dfmixed["massin_edema"].plot.line(color = "red", linestyle="-.",label = "mixed")

plt.legend()
plt.ylabel("Mass in nmol")
plt.xlabel("Time in H")
plt.tight_layout()
if fig_paper:
    plt.savefig('../article-glymphatics-tumor/images/edema_clearance_mixed.png')
    plt.close()
else:
    plt.title("Clearance in edema")
    plt.show()


### Check EPR ratios
# in tumor
EPR_vaso = dfvaso["massin_tumor"]/dfref["massin_tumor"]
EPR_nonvaso = dfnonvaso["massin_tumor"]/dfref["massin_tumor"]
EPR_mixed = dfmixed["massin_tumor"]/dfref["massin_tumor"]

EPR_vaso.plot.line(color = "orange", label = "vasogenic")
EPR_nonvaso.plot.line(color = "green",   label = "nonvasogenic")
EPR_mixed.plot.line(color = "red",   label = "mixed")

plt.legend()
plt.ylabel("Ratio")
plt.xlabel("Time in H")
plt.tight_layout()
if fig_paper:
    plt.savefig('../article-glymphatics-tumor/images/EPR_tumor.png')
    plt.close()
else:
    plt.title("Clearance in tumor")
    plt.show()

# in edema
EPR_vaso = dfvaso["massin_edema"]/dfref["massin_edema"] 
EPR_nonvaso = dfnonvaso["massin_edema"]/dfref["massin_edema"]
EPR_mixed = dfmixed["massin_edema"]/dfref["massin_edema"]

EPR_vaso.plot.line(color = "orange", label = "vasogenic")
EPR_nonvaso.plot.line(color = "green",   label = "nonvasogenic")
EPR_mixed.plot.line(color = "red",   label = "mixed")

plt.legend()
plt.ylabel("Ratio")
plt.xlabel("Time in H")
plt.tight_layout()
if fig_paper:
    plt.savefig('../article-glymphatics-tumor/images/EPR_edema.png')
    plt.close()
else:
    plt.title("Clearance in tumor")
    plt.show()


# in the healthy part of the brain
EPR_vaso = dfvaso["massin_healthy"]/dfref["massin_healthy"] 
EPR_nonvaso = dfnonvaso["massin_healthy"]/dfref["massin_healthy"]
EPR_mixed = dfmixed["massin_healthy"]/dfref["massin_healthy"]

EPR_vaso.plot.line(color = "orange", label = "vasogenic")
EPR_nonvaso.plot.line(color = "green",   label = "nonvasogenic")
EPR_mixed.plot.line(color = "red",   label = "mixed")

plt.legend()
plt.ylabel("Ratio")
plt.xlabel("Time in H")
plt.tight_layout()
if fig_paper:
    plt.savefig('../article-glymphatics-tumor/images/EPR_healthy.png')
    plt.close()
else:
    plt.title("Clearance in tumor")
    plt.show()

# total brain 
EPR_vaso = dfvaso.sum(axis=1)/dfref.sum(axis=1)
EPR_nonvaso = dfnonvaso.sum(axis=1)/dfref.sum(axis=1)
EPR_mixed = dfmixed.sum(axis=1)/dfref.sum(axis=1)

EPR_vaso.plot.line(color = "orange", label = "vasogenic")
EPR_nonvaso.plot.line(color = "green",   label = "nonvasogenic")
EPR_mixed.plot.line(color = "red",   label = "mixed")

plt.legend()
plt.ylabel("Ratio")
plt.xlabel("Time in H")
plt.tight_layout()
if fig_paper:
    plt.savefig('../article-glymphatics-tumor/images/EPR_total.png')
    plt.close()
else:
    plt.title("Clearance in tumor")
    plt.show()

