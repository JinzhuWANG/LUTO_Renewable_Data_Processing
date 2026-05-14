import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr

from pathlib import Path
from rasterio.enums import Resampling
from scipy.ndimage import distance_transform_edt
from tqdm.auto import tqdm


# Read LUTO template
luto_template = rxr.open_rasterio('N:/Data-Master/National_Landuse_Map/lumap.tif', masked=True).sel(band=1, drop=True)
luto_mask = luto_template >= -1 
mask_stacked = luto_mask.stack(cell=('y', 'x'))

renewable_path = Path("N:/Data-Master/Renewable Energy")

# Function to fill values using nearest neighbor interpolation
def fill_with_nearest(data_2d, to_fill=0):
    mask = data_2d.isnull() | (data_2d == to_fill)
    indices = distance_transform_edt(mask, return_distances=False, return_indices=True)
    data_2d.values = data_2d.values[tuple(indices)]
    return data_2d

def reproject_and_fill(raw_raster, template, mask, resampling=Resampling.nearest):
    # Reproject and match to LUTO template
    matched = raw_raster.rio.reproject_match(template, resampling=resampling).sel(band=1, drop=True)
    # Fill nan values with nearest neighbor interpolation
    filled = fill_with_nearest(matched)
    # Mask to LUTO valid areas
    masked = filled.where(mask)
    return masked


def load_tifs_as_xr(tifs, tech_name):
    # Load year-stamped tifs into a DataArray with year and tech_name dims
    results = []
    for tif in tifs:
        raw = rxr.open_rasterio(tif, masked=True).compute()
        matched = (
            reproject_and_fill(raw, luto_template, luto_mask)
            .expand_dims({'year': [int(tif.stem[-4:])], 'tech_name': [tech_name]})
        )
        results.append(matched)
    return xr.concat(results, dim='year')


# ---------------------------------------------------
#               capacity rasters            
# ---------------------------------------------------

raw_path = renewable_path / "20260127/capacity_factor"

raw_solar = rxr.open_rasterio(raw_path / "capacity_factor_solar.tif", masked=True).compute()
raw_wind = rxr.open_rasterio(raw_path / "capacity_factor_wind.tif", masked=True).compute()

raw_solar_matched = (
    reproject_and_fill(raw_solar, luto_template, luto_mask)
    .expand_dims({'tech_name': ["Utility Solar PV"]})
)
raw_wind_matched = (
    reproject_and_fill(raw_wind, luto_template, luto_mask)
    .expand_dims({'tech_name': ["Onshore Wind"]})
)

xr_capacity = xr.concat([raw_solar_matched, raw_wind_matched], dim='tech_name')

# Save to geoTIFF, which can be used in getting the existing renewable plants
arr_solar = raw_solar_matched.sel(tech_name="Utility Solar PV", drop=True)
arr_solar.rio.to_raster(renewable_path / "processed/capacity_factor_solar_reproject_match.tif", dtype=np.float32, compress='lzw')
arr_wind = raw_wind_matched.sel(tech_name="Onshore Wind", drop=True)
arr_wind.rio.to_raster(renewable_path / "processed/capacity_factor_wind_reproject_match.tif", dtype=np.float32, compress='lzw')

# ---------------------------------------------------
#     Scenario-based rasters (capex, dlf, opex)
# ---------------------------------------------------

base_path = renewable_path / "20260225"

scenarios = [
    'step_change', 
    'accelerated_transition', 
    'ANU_transmission_T3', 
    'ANU_transmission_T5', 
    'ANU_transmission_T10'
]

anu_path_map = {
    'ANU_transmission_T3':  base_path / 'ANU_transmission/Top3_2050',
    'ANU_transmission_T5':  base_path / 'ANU_transmission/Top5_2050',
    'ANU_transmission_T10': base_path / 'ANU_transmission/Top10_2050',
}


capex_by_scenario = {}
dlf_by_scenario = {}
opex_by_scenario = {}

for scenario in tqdm(scenarios, desc='Scenarios'):
    
    if scenario in ('step_change', 'accelerated_transition'):
        data_path = base_path / scenario
    else:
        data_path = anu_path_map[scenario]

    wind_capex_tifs  = sorted(data_path.rglob("wind_capex*.tif"))
    solar_capex_tifs = sorted(data_path.rglob("solar_capex*.tif"))
    dlf_tifs         = sorted(data_path.rglob("transmission_loss*.tif"))
    wind_opex_tifs   = sorted(data_path.rglob("wind_opex*.tif"))
    solar_opex_tifs  = sorted(data_path.rglob("solar_opex*.tif"))
    
    # Distribution loss does not have `tech_name` dimension
    dlf_list = []
    for tif in dlf_tifs:
        raw = rxr.open_rasterio(tif, masked=True).compute()
        dlf_list.append(
            reproject_and_fill(raw, luto_template, luto_mask)
            .expand_dims({'year': [int(tif.stem[-4:])]})
        )
    dlf_by_scenario[scenario] = xr.concat(dlf_list, dim='year')

    # Capex and Opex have `tech_name` dimension, so we can use the helper function
    capex_by_scenario[scenario] = xr.concat(
        [load_tifs_as_xr(wind_capex_tifs, "Onshore Wind"),
         load_tifs_as_xr(solar_capex_tifs, "Utility Solar PV")],
        dim='tech_name'
    )

    opex_by_scenario[scenario] = xr.concat(
        [load_tifs_as_xr(wind_opex_tifs, "Onshore Wind"),
         load_tifs_as_xr(solar_opex_tifs, "Utility Solar PV")],
        dim='tech_name'
    )



# ANU scenarios only provide 2050 data; broadcast constant values to all years
full_years = dlf_by_scenario['step_change']['year'].values
for s in [sc for sc in scenarios if sc.startswith('ANU')]:
    dlf_by_scenario[s]   = dlf_by_scenario[s].reindex(year=full_years, method='nearest')
    capex_by_scenario[s] = capex_by_scenario[s].reindex(year=full_years, method='nearest')
    opex_by_scenario[s]  = opex_by_scenario[s].reindex(year=full_years, method='nearest')


# ---------------------------------------------------
#            Combine all rasters into nc
# ---------------------------------------------------

# Combine all scenarios into single DataArrays with a new 'scenario' dimension
xr_capex = xr.concat(
    [capex_by_scenario[s].expand_dims({'scenario': [s]}) for s in scenarios],
    dim='scenario'
)
xr_dlf = xr.concat(
    [dlf_by_scenario[s].expand_dims({'scenario': [s]}) for s in scenarios],
    dim='scenario'
)
xr_opex = xr.concat(
    [opex_by_scenario[s].expand_dims({'scenario': [s]}) for s in scenarios],
    dim='scenario'
)

# Get 2D first
re_datasets_2D = xr.Dataset({
    'capacity_factor_multiplier': xr_capacity,
    'Cost_of_install_AUD_kw': xr_capex,
    'distribution_loss_factor_multiplier': xr_dlf,
    'Cost_of_operation_AUD_kw': xr_opex,
})
re_datasets_2D.to_netcdf(
    renewable_path / 'processed/renewable_energy_layers_2D.nc',
    encoding={ var: {'zlib': True, 'complevel': 5} for var in re_datasets_2D.data_vars}
)



# Save to 1D (LUTO long format)
re_stacked = re_datasets_2D.stack(cell=('y', 'x'))

re_datasets_1D = (
    re_stacked.sel(cell=mask_stacked.values)
    .drop_vars(['cell', 'x', 'y'])
    .assign_coords(cell=np.arange(mask_stacked.sum()))
)

re_datasets_1D.to_netcdf(
    renewable_path / 'processed/renewable_energy_layers_1D.nc',
    encoding={ var: {'zlib': True, 'complevel': 5} for var in re_datasets_1D.data_vars}
)




# ---------------------------------------------------
#          Renewable exclusion rasters        
# ---------------------------------------------------

exclusion_path = renewable_path / "20260317/QLD_EPBC_MNES_prioritization.tif"

re_exclusion_raw = rxr.open_rasterio(exclusion_path, masked=True).compute()
re_exclusion_reprojected = re_exclusion_raw.rio.reproject_match(luto_template, resampling=Resampling.nearest).sel(band=1, drop=True) * luto_mask
re_exclusion_reprojected.data = np.nan_to_num(re_exclusion_reprojected.data, nan=0)

# Save to 1D (LUTO long format)
excl_stacked = re_exclusion_reprojected.stack(cell=('y', 'x'))

re_exclusion_1D = (
    excl_stacked.sel(cell=mask_stacked.values)
    .drop_vars(['cell', 'x', 'y'])
    .assign_coords(cell=np.arange(mask_stacked.sum()))
)

re_exclusion_1D.name = 'data'

re_exclusion_1D.to_netcdf(
    renewable_path / 'processed/renewable_QLD_EPBC_MNES_prioritization.nc',
    encoding={'data': {'zlib': True, 'complevel': 5}}
)


# Load real cell area (ha) and stack to 1D to match re_exclusion_1D
cell_area_1D = (
    rxr.open_rasterio('N:/Data-Master/National_Landuse_Map/NLUM_2010-11_cell_ha.tif', masked=True)
    .sel(band=1, drop=True)
    .stack(cell=('y', 'x'))
    .sel(cell=mask_stacked.values)
    .drop_vars(['cell', 'x', 'y'])
    .assign_coords(cell=np.arange(mask_stacked.sum()))
)


# Build cumulative area vs cumulative priority contribution performance curve
excl_df = re_exclusion_1D.to_dataframe(name='PRIORITY_RANK').reset_index()
excl_df['area'] = cell_area_1D.values
excl_df = excl_df[excl_df['PRIORITY_RANK'] > 0].sort_values('PRIORITY_RANK', ascending=False)

excl_df['AREA_COVERAGE_PERCENT'] = excl_df['area'].astype(float).cumsum() / excl_df['area'].sum() * 100
excl_df['PRIORITY_RANK_CUMSUM_CONTRIBUTION'] = (
    (excl_df['PRIORITY_RANK'].astype(float) * excl_df['area']).cumsum() 
    / (excl_df['PRIORITY_RANK'] * excl_df['area']).sum() 
    * 100
)

# Downsample to 101 rows at integer percentiles (0-100)
idx = abs(np.arange(101).reshape(-1, 1) - excl_df['AREA_COVERAGE_PERCENT'].values).argmin(axis=1)
excl_stats = excl_df.iloc[idx, :].drop(columns=['area', 'cell']).copy().drop(columns=['spatial_ref'])
excl_stats['AREA_COVERAGE_PERCENT'] = np.arange(101)


excl_stats.to_csv(renewable_path / 'processed' / 'renewable_QLD_EPBC_MNES_prioritization_performance.csv', index=False)

