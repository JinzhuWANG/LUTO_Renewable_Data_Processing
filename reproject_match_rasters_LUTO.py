import numpy as np
import rioxarray as rxr
import xarray as xr

from pathlib import Path
from rasterio.enums import Resampling
from scipy.ndimage import distance_transform_edt
from tqdm.auto import tqdm


# Read LUTO template
luto_template = rxr.open_rasterio('N:/Data-Master/National_Landuse_Map/lumap.tif', masked=True).sel(band=1, drop=True)
luto_mask = luto_template >= -1 


# Function to fill values using nearest neighbor interpolation
def fill_with_nearest(data_2d:xr.DataArray, to_fill:float=0) -> np.ndarray:
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
    """Load year-stamped tifs into a DataArray with year and tech_name dims."""
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

raw_path = Path("N:/Data-Master/Renewable Energy/20260127/capacity_factor")

raw_solar = rxr.open_rasterio(raw_path / "capacity_factor_solar.tif", masked=True).compute()
raw_wind = rxr.open_rasterio(raw_path / "capacity_factor_wind.tif", masked=True).compute()

raw_solar_matched = (
    reproject_and_fill(raw_solar, luto_template, luto_mask)
    .expand_dims({'year': list(range(2030,2051)), 'tech_name': ["Utility Solar PV"]})
)
raw_wind_matched = (
    reproject_and_fill(raw_wind, luto_template, luto_mask)
    .expand_dims({'year': list(range(2030,2051)), 'tech_name': ["Onshore Wind"]})
)

xr_capacity = xr.concat([raw_solar_matched, raw_wind_matched], dim='tech_name')


# ---------------------------------------------------
#     Scenario-based rasters (capex, dlf, opex)
# ---------------------------------------------------

base_path = Path("N:/Data-Master/Renewable Energy/20260225")

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






