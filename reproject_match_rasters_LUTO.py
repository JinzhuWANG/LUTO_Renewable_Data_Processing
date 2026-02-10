import numpy as np
import rioxarray as rxr
import xarray as xr

from pathlib import Path
from rasterio.enums import Resampling
from scipy.ndimage import distance_transform_edt


# Function to fill values using nearest neighbor interpolation
def fill_with_nearest(data_2d:xr.DataArray, to_fill:float=0) -> np.ndarray:
    mask = data_2d.isnull() | (data_2d == to_fill)
    indices = distance_transform_edt(mask, return_distances=False, return_indices=True)
    data_2d.values = data_2d.values[tuple(indices)]
    return data_2d


# Read LUTO template
luto_template = rxr.open_rasterio('N:/Data-Master/National_Landuse_Map/lumap.tif', masked=True).sel(band=1, drop=True)
luto_mask = luto_template >= -1 


# ---------------------------------------------------
#               capacity rasters            
# ---------------------------------------------------

raw_path = Path("N:/Data-Master/Renewable Energy/20260127/capacity_factor")
raw_solar = rxr.open_rasterio(raw_path / "capacity_factor_solar.tif", masked=True).compute()
raw_wind = rxr.open_rasterio(raw_path / "capacity_factor_wind.tif", masked=True).compute()


# Reproject and match to LUTO template
solar_matched = raw_solar.rio.reproject_match(luto_template, resampling=Resampling.nearest).sel(band=1, drop=True)
wind_matched = raw_wind.rio.reproject_match(luto_template, resampling=Resampling.nearest).sel(band=1, drop=True)

# Fill nan values with nearest neighbor interpolation
raw_solar_filled = fill_with_nearest(solar_matched)
raw_wind_filled = fill_with_nearest(wind_matched)

# Mask to LUTO valid areas
solar_matched_masked = raw_solar_filled.where(luto_mask).expand_dims({'Type': ["Utility Solar PV"]})
wind_matched_masked = raw_wind_filled.where(luto_mask).expand_dims({'Type': ["Onshore Wind"]})



# ---------------------------------------------------
#               capacity expenditure rasters            
# ---------------------------------------------------

year = 2030
raw_path_exp = Path("N:/Data-Master/Renewable Energy/20260127/capex")
raw_solar_exp = rxr.open_rasterio(raw_path_exp / str(year) / f'solar_capex_{year}.tif', masked=True).compute()
raw_wind_exp = rxr.open_rasterio(raw_path_exp / str(year) / f'wind_capex_{year}.tif', masked=True).compute()

# Reproject and match to LUTO template
solar_exp_matched = raw_solar_exp.rio.reproject_match(luto_template, resampling=Resampling.nearest).sel(band=1, drop=True)
wind_exp_matched = raw_wind_exp.rio.reproject_match(luto_template, resampling=Resampling.nearest).sel(band=1, drop=True)

# Fill nan values with nearest neighbor interpolation
raw_solar_exp_filled = fill_with_nearest(solar_exp_matched, to_fill=0)
raw_wind_exp_filled = fill_with_nearest(wind_exp_matched, to_fill=0)

# Mask to LUTO valid areas
solar_exp_matched_masked = raw_solar_exp_filled.where(luto_mask).expand_dims({'year': [year], 'Type': ["Utility Solar PV"]})
wind_exp_matched_masked = raw_wind_exp_filled.where(luto_mask).expand_dims({'year': [year], 'Type': ["Onshore Wind"]})



# ---------------------------------------------------
#            distribution loss factor          
# ---------------------------------------------------

year=2030
raw_dlf_path = Path("N:/Data-Master/Renewable Energy/20260127/distribution_loss_factor")
raw_dlf = rxr.open_rasterio(raw_dlf_path / str(year) / f'transmission_loss_{year}.tif', masked=True).compute()

# Reproject and match to LUTO template
dlf_matched = raw_dlf.rio.reproject_match(luto_template, resampling=Resampling.nearest).sel(band=1, drop=True)
# Fill nan values with nearest neighbor interpolation
raw_dlf_filled = fill_with_nearest(dlf_matched)
# Mask to LUTO valid areas
dlf_matched_masked = raw_dlf_filled.where(luto_mask).expand_dims({'year': [year]})



# ---------------------------------------------------
#            operation expenditure rasters          
# ---------------------------------------------------

year = 2030
raw_opex_path = Path("N:/Data-Master/Renewable Energy/20260127/opex")
raw_solar_opex = rxr.open_rasterio(raw_opex_path / str(year) / f'solar_opex_{year}.tif', masked=True).compute()
raw_wind_opex = rxr.open_rasterio(raw_opex_path / str(year) / f'wind_opex_{year}.tif', masked=True).compute()

# Reproject and match to LUTO template
solar_opex_matched = raw_solar_opex.rio.reproject_match(luto_template, resampling=Resampling.nearest).sel(band=1, drop=True)
wind_opex_matched = raw_wind_opex.rio.reproject_match(luto_template, resampling=Resampling.nearest).sel(band=1, drop=True)

# Fill nan values with nearest neighbor interpolation
raw_solar_opex_filled = fill_with_nearest(solar_opex_matched)
raw_wind_opex_filled = fill_with_nearest(wind_opex_matched)

# Mask to LUTO valid areas
solar_opex_matched_masked = raw_solar_opex_filled.where(luto_mask).expand_dims({'year': [year], 'Type': ["Utility Solar PV"]})
wind_opex_matched_masked = raw_wind_opex_filled.where(luto_mask).expand_dims({'year': [year], 'Type': ["Onshore Wind"]})



# ---------------------------------------------------
#            Combine all rasters into nc          
# ---------------------------------------------------

# Save as 2D first
re_datasets_2D = xr.Dataset({
    'Capacity_percent_of_natural_energy': xr.concat([solar_matched_masked, wind_matched_masked], dim='Type'),
    'Cost_of_install_AUD_ha': xr.concat([solar_exp_matched_masked, wind_exp_matched_masked], dim='Type'),
    'Energy_remain_percent_after_distribution': dlf_matched_masked,
    'Cost_of_operation': xr.concat([solar_opex_matched_masked, wind_opex_matched_masked], dim='Type'),
})

re_datasets_2D.to_netcdf(
    '../processed/renewable_energy_layers_2D.nc',
    encoding={ var: {'zlib': True, 'complevel': 5} for var in re_datasets_2D.data_vars}
    )



# Save to 1D (LUTO long format)
re_stacked = re_datasets_2D.stack(cell=('y', 'x'))
mask_stacked = luto_mask.stack(cell=('y', 'x'))

re_datasets_1D = (
    re_stacked.sel(cell=mask_stacked.values)
    .drop_vars(['cell', 'x', 'y'])
    .assign_coords(cell=np.arange(mask_stacked.sum()))
)

re_datasets_1D.to_netcdf(
    '../processed/renewable_energy_layers_1D.nc', 
    encoding={ var: {'zlib': True, 'complevel': 5} for var in re_datasets_1D.data_vars}
)






