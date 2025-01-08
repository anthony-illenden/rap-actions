import numpy as np
import metpy.calc as mpcalc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from siphon.catalog import TDSCatalog
import xarray as xr
from scipy.ndimage import gaussian_filter
from metpy.units import units
import time
import matplotlib.colors as mcolors

script_start = time.time()

print('---------------------------------------')
print('RAP 250-hPa Script - Script started.')
print('---------------------------------------')

# Helper function for finding proper time variable
def find_time_var(var, time_basename='time'):
    for coord_name in var.coords:
        if coord_name.startswith(time_basename):
            return var.coords[coord_name]
    raise ValueError('No time variable found for ' + var.name)

def find_press_var(var, time_basename='isobaric'):
    for coord_name in var.coords:
        if coord_name.startswith(time_basename):
            return var.coords[coord_name]
    raise ValueError('No time variable found for ' + var.name)

def base_map():
    fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': ccrs.LambertConformal()})
    ax.set_extent([-125, -66.9, 23, 49.4])
    ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='gray', linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    
    return fig, ax

def get_rap_data():
    tds_rap = TDSCatalog('https://thredds.ucar.edu/thredds/catalog/grib/NCEP/RAP/CONUS_13km/latest.html')
    rap_ds = tds_rap.datasets[0]
    ds = xr.open_dataset(rap_ds.access_urls['OPENDAP'])
    ds = ds.metpy.parse_cf()
    ds_latlon = ds.metpy.assign_latitude_longitude()


    time_dim = find_time_var(ds_latlon['Temperature_isobaric'])
    iso_dim = find_press_var(ds_latlon['Temperature_isobaric'])

    init_time = time_dim[0].values

    target_length = len(time_dim) # forecast hours

    # Initialize a variable to store the matching dimension name
    matching_dim = None

    # Loop over the dimensions and check their lengths
    for dim, size in ds_latlon.dims.items():
        if size == target_length:
            matching_dim = dim
            break  # Exit loop once a match is found

    return ds_latlon, matching_dim, init_time

def plot_250(ds_latlon, matching_dim, init_time):
    for i in range(0, 22):
        iteration_start = time.time()
        ds_loop = ds_latlon.isel({matching_dim: i})

        uwnd_250 = ds_loop['u-component_of_wind_isobaric'].sel(isobaric=250*100) * 1.94384 * units.knots 
        vwnd_250 = ds_loop['v-component_of_wind_isobaric'].sel(isobaric=250*100) * 1.94384 * units.knots 
        hght_250 = ds_loop['Geopotential_height_isobaric'].sel(isobaric=250*100) * units.meter

        wind = mpcalc.wind_speed(uwnd_250, vwnd_250) # knots 

        uwnd_smoothed = gaussian_filter(uwnd_250, sigma=1.0)
        vwnd_smoothed = gaussian_filter(vwnd_250, sigma=1.0)
        wind_smoothed = gaussian_filter(wind, sigma=1.0)
        hght_smoothed = gaussian_filter(hght_250, sigma=1.0)

        # Extract latitude and longitude 2D arrays
        latitudes = uwnd_250.coords['latitude'].values
        longitudes = uwnd_250.coords['longitude'].values

        # Select the wind components to match the 2D latitude and longitude
        uwnd_values = uwnd_smoothed
        vwnd_values = vwnd_smoothed

        # Subsample the arrays using the step value (e.g., every 10th point)
        step = 25  # Adjust the step size as needed (larger values = fewer barbs)

        # Extract the latitudes, longitudes, and wind values for the barbs
        latitudes_subsampled = latitudes[::step, ::step]
        longitudes_subsampled = longitudes[::step, ::step]
        uwnd_subsampled = uwnd_values[::step, ::step]
        vwnd_subsampled = vwnd_values[::step, ::step]


        # Define the contour levels in m/s (before conversion)
        levels = np.arange(20, 95, 5)

        # Convert the levels to knots
        levels_knots = levels * 1.94384

        # Define the colors and colormap
        colors = ['#daedfb', '#b7dcf6', '#91bae4', '#7099ce', '#6a999d', '#72ad63', '#77c14a', '#cad955', '#f8cf4f', '#f7953c', '#ef5f28', '#e13e26', '#cd1e28', '#b1181e', '#901617']
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(levels_knots, cmap.N)


        # Now plot the barbs using the subsampled arrays
        fig, ax = base_map()

        # Plot isohypses (contours of geopotential height)
        isohypses = plt.contour(longitudes, latitudes, hght_smoothed, 
                                colors='black', levels=np.arange(8700, 11820, 60), linewidths=1, transform=ccrs.PlateCarree())
        try:
            ax.clabel(isohypses, inline=True, inline_spacing=5, fontsize=10, fmt='%i')
        except IndexError:
            print("No contours to label.")

        cf = plt.contourf(longitudes, latitudes, wind_smoothed, cmap=cmap, norm=norm, levels=levels_knots, extend='max', transform=ccrs.PlateCarree())
        plt.colorbar(cf, ax=ax, orientation='horizontal', label='Isotachs (knots)', pad=0.05, aspect=50)

        # Plot wind barbs using the subsampled coordinates and wind component values
        ax.barbs(longitudes_subsampled, latitudes_subsampled, 
                uwnd_subsampled, vwnd_subsampled, 
                length=6, color='black', transform=ccrs.PlateCarree())

        hour_difference = (ds_latlon[matching_dim][i] - init_time) / np.timedelta64(1, 'h')

        plt.title(f"{ds_latlon[matching_dim][0].dt.strftime('%H00 UTC').item()} RAP 250-hPa Isotachs and Geopotential Heights | {ds_latlon[matching_dim][i].dt.strftime('%Y-%m-%d %H00 UTC').item()} | FH: {hour_difference:.0f}", fontsize=12)
        plt.tight_layout()
        plt.savefig(f'plots/250/_{i}.png', dpi=450)
        iteration_end = time.time()
        print(f'Iteration {i} Processing Time:', round((iteration_end - iteration_start), 2), 'seconds.')

if __name__ == '__main__':
    ds_latlon, matching_dim, init_time = get_rap_data()
    plot_250(ds_latlon, matching_dim, init_time)

