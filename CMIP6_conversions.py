import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Data exported from publication Warren et al. 2008 (Fig. 1) but originated in
# Perovich and Govoni 1991.
# Data are interpolated to a fixed wavelength grid that fits with the wavelengths of
# Seferian et al. 2018

infile = "ice-absorption/sea_ice_absorption_perovich_and_govoni.csv"
df = pd.read_csv(infile)
print(df.head())
# Define the grid to interpolate to
wavelengths = np.arange(200, 1000, 10)
k_ice = np.arange(0, 25, 0.01)
# Get values from dataframe
k_ice_pg = df["k_ice_pg"].values
x_k_ice_pg = df["wavelength"].values
# Do the interpolation
interp_k_ice = np.interp(wavelengths, x_k_ice_pg, k_ice_pg)
# Store data to csv file as dataframe
data = {"wavelength": wavelengths, "k_ice_pg": interp_k_ice}
df_out = pd.DataFrame(data, index=wavelengths)
csv_filename = "ice-absorption/sea_ice_absorption_perovich_and_govoni_interpolated.csv"
if os.path.exists(csv_filename): os.remove(csv_filename)
df_out.to_csv(csv_filename, index=False)

# Plot the result
plt.plot(wavelengths, interp_k_ice, c="r", marker="o")
plt.title("Absorption through sea-ice as  function of wavelength (Perovich and Govani 1991)")
plt.show()
