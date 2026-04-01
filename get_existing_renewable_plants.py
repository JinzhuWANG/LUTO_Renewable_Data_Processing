import os
import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray as rxr
import rasterio
import rasterio.transform

from pathlib import Path
from shapely.geometry import box
from shapely.ops import unary_union
from joblib import Parallel, delayed
from tqdm.auto import tqdm


# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------

DATA_DIR = Path('N:/Data-Master')
RE_DIR   = DATA_DIR / 'Renewable Energy'


# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

CF_SOLAR    = rxr.open_rasterio(RE_DIR / 'processed/capacity_factor_solar_reproject_match.tif').sel(band=1, drop=True).compute()
CF_WIND     = rxr.open_rasterio(RE_DIR / 'processed/capacity_factor_wind_reproject_match.tif').sel(band=1, drop=True).compute()
HOURS       = 365 * 24      # Hours per year

POWER_SOLAR = 0.45          # Average MWh per ha
POWER_WIND  = 0.04          # Average MWh per ha

# Capacity field priority (highest to lowest) — Solar
# Priority 1: Agg Nameplate Capacity (MW AC)       - Aggregate-level + AC side; closest to grid-metered generation (37 rows)
# Priority 2: Aggregated Upper Nameplate Capacity  - Aggregate-level but AC/DC unspecified; fallback (89 rows)
# Priority 3: Max Site Capacity (AC)               - Site-level AC capacity ceiling; secondary fallback (0 rows in current data)
# Priority 4: Nameplate Capacity (MW)              - Unit-level nameplate; limited coverage (8 rows)
# Priority 5: Upper Nameplate Capacity (MW)        - Unit-level upper bound; last resort (0 rows in current data)
CAPACITY_PRIORITY_SOLAR = [
    'Agg Nameplate Capacity (MW AC)',
    'Aggregated Upper Nameplate Capacity (MW)',
    'Max Site Capacity (AC)',
    'Nameplate Capacity (MW)',
    'Upper Nameplate Capacity (MW)',
]

# Capacity field priority (highest to lowest) — Wind
# Priority 1: Agg Nameplate Capacity (MW AC)       - Aggregate-level + AC side; closest to grid-metered generation (160 rows)
# Priority 2: Max Site Capacity (AC)               - Site-level AC capacity ceiling (0 new rows beyond P1)
# Priority 3: Nameplate Capacity (MW)              - Unit-level nameplate; covers remaining 93 rows
# Priority 4: Aggregated Upper Nameplate Capacity  - Aggregate-level upper bound; safety net (0 new rows beyond P1-P3)
# Priority 5: Upper Nameplate Capacity (MW)        - Unit-level upper bound; last resort (0 new rows beyond P1-P4)
CAPACITY_PRIORITY_WIND = [
    'Agg Nameplate Capacity (MW AC)',
    'Max Site Capacity (AC)',
    'Nameplate Capacity (MW)',
    'Aggregated Upper Nameplate Capacity (MW)',
    'Upper Nameplate Capacity (MW)',
]


# -----------------------------------------------------------------------
# LUTO reference grid
# -----------------------------------------------------------------------

luto_template  = rasterio.open(DATA_DIR / 'National_Landuse_Map/lumap.tif')
ref_transform  = luto_template.transform
ref_crs        = luto_template.crs
ref_mask       = luto_template.read_masks(1).astype(bool)
ref_shape      = (luto_template.height, luto_template.width)
ref_meta_float = {**luto_template.meta, 'dtype': 'float32', 'nodata': np.nan}


lumap = pd.read_hdf(
    'N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_LU_mapping.h5', 
    key = 'cell_LU_mapping', 
    columns=['LU_DESC','LU_ID_LUTO']
)
idx_out_LUTO = np.isin(lumap['LU_DESC'], ['Non-agricultural land'])                 # shape=6956407, sum=2737674


states = pd.read_hdf(
    'N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_zones_df.h5', 
    key='cell_zones_df', 
    columns=['STE_NAME11']
)




# -----------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------

def build_luto_cells():
    """Return a GeoDataFrame of all valid LUTO cells (where ref_mask is True)."""

    def _make_cell(r, c):
        x0, y0 = rasterio.transform.xy(ref_transform, r, c, offset='ul')
        return r, c, box(x0, y0 + ref_transform.e, x0 + ref_transform.a, y0)

    rc_pairs = np.argwhere(ref_mask)
    tasks = [delayed(_make_cell)(r, c) for r, c in rc_pairs]
    rows, cols, geoms = zip(*tqdm(
        Parallel(n_jobs=-1, return_as='generator')(tasks), total=len(tasks), desc='Building LUTO cells'
    ))
    gdf = gpd.GeoDataFrame({'row': rows, 'col': cols, 'geometry': geoms}, crs=ref_crs)
    return gdf


def pick_capacity(row, priority_cols):
    for col in priority_cols:
        if pd.notna(row[col]):
            return row[col], col
    return np.nan, 'NO DATA'


# -----------------------------------------------------------------------
# Build LUTO cells and save cell area
# -----------------------------------------------------------------------

luto_cells = build_luto_cells()
luto_cells['area_ha_cell'] = luto_cells.geometry.to_crs('EPSG:3577').area / 10_000   # ha

cell_area = np.where(ref_mask, 0.0, np.nan).astype(np.float32)
for (r, c), prop in luto_cells.groupby(['row', 'col'])['area_ha_cell'].sum().items():
    cell_area[r, c] = prop

cell_area_xr      = xr.DataArray(cell_area, dims=('y', 'x'), coords={'y': CF_SOLAR.y, 'x': CF_SOLAR.x})
cell_area_xr.name = 'data'
cell_area_xr.to_netcdf(RE_DIR / 'processed/luto_cell_area_ha.nc', encoding={'data': {'dtype': 'float32', 'zlib': True, 'complevel': 5}})


# -----------------------------------------------------------------------
# Load input data
# -----------------------------------------------------------------------

# 1) Load raw points and polygons (no filtering)
in_gpkg_points = gpd.read_file(RE_DIR / '20260305/Network Map Renewables January 2026_AUS points.gpkg').to_crs(luto_template.crs)
in_gpkg_polys  = gpd.read_file(RE_DIR / '20260305/Network Map Renewables January 2026_AUS polygons.gpkg').to_crs(luto_template.crs)

# 2) Split by technology type
solar_points_raw = in_gpkg_points[in_gpkg_points['Technology Type'].str.contains("Solar", na=False)]
wind_points_raw  = in_gpkg_points[in_gpkg_points['Technology Type'].str.contains("Wind",  na=False)]
solar_polys_raw  = in_gpkg_polys[in_gpkg_polys['Technology Type'].str.contains("Solar",  na=False)]
wind_polys_raw   = in_gpkg_polys[in_gpkg_polys['Technology Type'].str.contains("Wind",   na=False)]

# 3) Build index dictionaries — status removals
solar_idx = {'Raw': solar_points_raw.index}
wind_idx  = {'Raw': wind_points_raw.index}

for tech, raw, idx in [('Solar', solar_points_raw, solar_idx), ('Wind', wind_points_raw, wind_idx)]:
    idx['Removing Anticipated']        = raw.index[raw['Commitment Status'] == 'Anticipated']
    idx['Removing Publicly Announced'] = raw.index[raw['Commitment Status'] == 'Publicly Announced']
    idx['Status kept'] = raw.index.difference(idx['Removing Anticipated']).difference(idx['Removing Publicly Announced'])

# 4) Build index dictionaries — capacity priority picks (solar)
re_solar_points = solar_points_raw.loc[solar_idx['Status kept']].copy()
for col in CAPACITY_PRIORITY_SOLAR:
    re_solar_points[col] = pd.to_numeric(re_solar_points[col], errors='coerce')

_unpicked = re_solar_points.index
for col in CAPACITY_PRIORITY_SOLAR:
    solar_idx[f'Keeping {col}'] = _unpicked[re_solar_points.loc[_unpicked, col].notna()]
    _unpicked = _unpicked.difference(solar_idx[f'Keeping {col}'])
solar_idx['Removing No capacity'] = _unpicked
solar_idx['Final'] = re_solar_points.index.difference(_unpicked)

# 5) Build index dictionaries — capacity priority picks (wind)
re_wind_points = wind_points_raw.loc[wind_idx['Status kept']].copy()
for col in CAPACITY_PRIORITY_WIND:
    re_wind_points[col] = pd.to_numeric(re_wind_points[col], errors='coerce')

_unpicked = re_wind_points.index
for col in CAPACITY_PRIORITY_WIND:
    wind_idx[f'Keeping {col}'] = _unpicked[re_wind_points.loc[_unpicked, col].notna()]
    _unpicked = _unpicked.difference(wind_idx[f'Keeping {col}'])
wind_idx['Removing No capacity'] = _unpicked
wind_idx['Final'] = re_wind_points.index.difference(_unpicked)

# 6) Filter to final rows using indexes
re_solar_points = re_solar_points.loc[solar_idx['Final']].copy()
re_wind_points  = re_wind_points.loc[wind_idx['Final']].copy()

# 7) Add unified capacity column
re_solar_points['Capacity (MW)'] = re_solar_points.apply(lambda r: pick_capacity(r, CAPACITY_PRIORITY_SOLAR)[0], axis=1)
re_wind_points['Capacity (MW)']  = re_wind_points.apply( lambda r: pick_capacity(r, CAPACITY_PRIORITY_WIND)[0],  axis=1)

re_solar_points.to_file(RE_DIR / '20260305/Network Map Renewables January 2026_AUS points_solar_filtered.gpkg')
re_wind_points.to_file(RE_DIR / '20260305/Network Map Renewables January 2026_AUS points_wind_filtered.gpkg')

# Cascade allocation tables
for tech, idx, priority_cols in [('Solar', solar_idx, CAPACITY_PRIORITY_SOLAR), ('Wind', wind_idx, CAPACITY_PRIORITY_WIND)]:
    print(f"\n{tech} points cascade{'':<37} {'Count':>6}")
    print('-' * 63)
    _steps = (
        [(f'Raw {tech.lower()} points',           +len(idx['Raw']))]
        + [(f'  Remove: Anticipated',              -len(idx['Removing Anticipated']))]
        + [(f'  Remove: Publicly Announced',       -len(idx['Removing Publicly Announced']))]
        + [(f'Remaining after status filter',        len(idx['Status kept']))]
        + [(f'  Keep: {col}',                       len(idx[f'Keeping {col}'])) for col in priority_cols]
        + [(f'  Remove: No capacity data',         -len(idx['Removing No capacity']))]
        + [(f'Final {tech.lower()} points used',    len(idx['Final']))]
    )
    for _label, _count in _steps:
        _sign = '+' if _count > 0 else ''
        print(f"{_label:<55} {_sign}{_count:>5}")





# Dissolve polygons by technology type to get unified areas.
#   Otherwise, overlapping polygons (e.g. multiple projects in the same area)
#   would lead to double-counting of generation potential.
re_solar_polys = solar_polys_raw[solar_polys_raw['Site Name'].isin(re_solar_points['Site Name'])].copy()
re_solar_polys['geometry'] = [unary_union(shp.buffer(0)) for shp in re_solar_polys['geometry']]

re_wind_polys  = wind_polys_raw[wind_polys_raw['Site Name'].isin(re_wind_points['Site Name'])].copy()
re_wind_polys['geometry'] = [unary_union(shp.buffer(0)) for shp in re_wind_polys['geometry']]


# Calculate capacity/ha
capacity_per_ha_solar = gpd.GeoDataFrame(
    re_solar_points[['Site Name', 'Capacity (MW)']].merge(re_solar_polys[['Site Name', 'area_ha', 'geometry']], on='Site Name')
)
capacity_per_ha_wind  = gpd.GeoDataFrame(
    re_wind_points[['Site Name', 'Capacity (MW)']].merge(re_wind_polys[['Site Name', 'area_ha', 'geometry']], on='Site Name')
)

capacity_per_ha_solar['Capacity per ha (MW/ha)'] = capacity_per_ha_solar['Capacity (MW)'] / capacity_per_ha_solar['area_ha']
capacity_per_ha_wind['Capacity per ha (MW/ha)']   = capacity_per_ha_wind['Capacity (MW)'] / capacity_per_ha_wind['area_ha']



# -----------------------------------------------------------------------
# Solar — polygon-based generation estimates
# -----------------------------------------------------------------------

# Intersect → fragment area within each LUTO cell
isect = gpd.overlay(capacity_per_ha_solar, luto_cells, how='intersection')
isect['area_insec'] = isect.geometry.to_crs('EPSG:3577').area / 10_000   # ha
isect['capacity_MW_insec'] = isect['area_insec'] * isect['Capacity per ha (MW/ha)']

# Write into output array
solar_MW_xr = xr.DataArray(np.where(ref_mask, 0.0, np.nan).astype(np.float32), dims=('y', 'x'), coords={'y': CF_SOLAR.y, 'x': CF_SOLAR.x})
for (r, c), prop in isect.groupby(['row', 'col'])['capacity_MW_insec'].sum().items():
    solar_MW_xr[r, c] = prop


# -----------------------------------------------------------------------
# Wind — polygon-based generation estimates
# -----------------------------------------------------------------------

# Intersect → fragment area within each LUTO cell
isect_wind = gpd.overlay(capacity_per_ha_wind, luto_cells, how='intersection')
isect_wind['area_insec'] = isect_wind.geometry.to_crs('EPSG:3577').area / 10_000   # ha
isect_wind['capacity_MW_insec'] = isect_wind['area_insec'] * isect_wind['Capacity per ha (MW/ha)']

# Write into output array
wind_MW_xr = xr.DataArray(np.where(ref_mask, 0.0, np.nan).astype(np.float32), dims=('y', 'x'), coords={'y': CF_WIND.y, 'x': CF_WIND.x})
for (r, c), prop in isect_wind.groupby(['row', 'col'])['capacity_MW_insec'].sum().items():
    wind_MW_xr[r, c] = prop




# -----------------------------------------------------------------------
# Save xarray outputs
# -----------------------------------------------------------------------

# Save 2D
existing_solar_capacity = xr.concat(
    [
        wind_MW_xr.expand_dims({'tech_name': ['Onshore Wind']}), 
        solar_MW_xr.expand_dims({'tech_name': ['Utility Solar PV']})
    ],
    dim='tech_name'
)

existing_solar_capacity.name = 'capacity_MW'
existing_solar_capacity.to_netcdf(RE_DIR / 'processed/renewable_existing_capacity_MW_2D.nc', encoding={'capacity_MW': {'dtype': 'float32', 'zlib': True, 'complevel': 5}})

# Save 1D 
existing_solar_capacity_1D = (
    existing_solar_capacity.stack(z=('y', 'x'))
    .sel(z=~solar_MW_xr.stack(z=('y', 'x')).isnull())
    .drop_vars('z')
    .rename({'z': 'cell'})
    .assign_coords(cell=range(len(luto_cells)))
)
existing_solar_capacity_1D.to_netcdf(RE_DIR / 'processed/renewable_existing_capacity_MW_1D.nc', encoding={'capacity_MW': {'dtype': 'float32', 'zlib': True, 'complevel': 5}})




# -----------------------------------------------------------------------
# Calculate the outside-LUTO power generation
# -----------------------------------------------------------------------

out_power = {}
for state in states['STE_NAME11'].unique():
    outside_state_mask = ((states['STE_NAME11'] == state) & idx_out_LUTO).values
    if state not in out_power.keys():
        out_power[state] = {}
        
    out_power[state]['Utility Solar PV'] = (existing_solar_capacity_1D.sel(tech_name='Utility Solar PV') * outside_state_mask).sum().item()
    out_power[state]['Onshore Wind'] = (existing_solar_capacity_1D.sel(tech_name='Onshore Wind') * outside_state_mask).sum().item()
        

out_power_df = pd.DataFrame(out_power).T.reset_index().rename(columns={'index': 'State'})
out_power_df.to_csv(RE_DIR / 'processed/renewable_existing_capacity_outside_LUTO_by_state.csv', index=False)




