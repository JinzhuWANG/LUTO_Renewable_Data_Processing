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
# LUTO reference grid
# -----------------------------------------------------------------------

luto_template  = rasterio.open(DATA_DIR / 'National_Landuse_Map/lumap.tif')
ref_transform  = luto_template.transform
ref_crs        = luto_template.crs
ref_mask       = luto_template.read_masks(1).astype(bool)
ref_shape      = (luto_template.height, luto_template.width)
ref_meta_float = {**luto_template.meta, 'dtype': 'float32', 'nodata': np.nan}

luto_template_xr = rxr.open_rasterio(DATA_DIR / 'National_Landuse_Map/lumap.tif', masked=True).sel(band=1, drop=True)


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
# Capacity factor rasters
# -----------------------------------------------------------------------
capacity_factor = xr.open_dataset(RE_DIR / 'processed/renewable_energy_layers_2D.nc')['capacity_factor_multiplier'].sel(year=2030, drop=True).compute()
CF_SOLAR    = capacity_factor.sel(tech_name='Utility Solar PV')
CF_WIND     = capacity_factor.sel(tech_name='Onshore Wind')


# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

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

# 8) Remove points without a valid commission year
#    CSVs were manually populated (via Google Earth) with a 'Commision Year' column.
#    Rows with a blank year (Cancelled, not visible from map, etc.) are excluded.

re_solar_df = re_solar_points[['Site Name','geometry']].copy()
re_solar_df['coords'] = list(zip(re_solar_df.geometry.y.round(2), re_solar_df.geometry.x.round(2)))
if (RE_DIR / 'processed/renewable_existing_capacity_commision_year_solar_points.csv').exists():
    commision_year_solar = pd.read_csv(RE_DIR / 'processed/renewable_existing_capacity_commision_year_solar_points.csv')
    commision_year_solar['Commision Year'] = commision_year_solar['Commision Year'].astype('Int16')   # Convert to nullable integer type
    commision_year_solar = commision_year_solar.dropna(subset=['Commision Year'])
    valid_solar_sites = commision_year_solar['Site Name']
    solar_idx['Removing No commission year'] = re_solar_points.index[~re_solar_points['Site Name'].isin(valid_solar_sites)]
    solar_idx['Final'] = re_solar_points.index[re_solar_points['Site Name'].isin(valid_solar_sites)]
    re_solar_points = re_solar_points.loc[solar_idx['Final']].copy()
else:
    re_solar_df.drop('geometry', axis=1).to_csv(RE_DIR / 'processed/renewable_existing_capacity_commision_year_solar_points.csv', index=False)
    solar_idx['Removing No commission year'] = re_solar_points.index[[]]

re_wind_df = re_wind_points[['Site Name','geometry']].copy()
re_wind_df['coords'] = list(zip(re_wind_df.geometry.y.round(2), re_wind_df.geometry.x.round(2)))
if (RE_DIR / 'processed/renewable_existing_capacity_commision_year_wind_points.csv').exists():
    commision_year_wind = pd.read_csv(RE_DIR / 'processed/renewable_existing_capacity_commision_year_wind_points.csv')
    commision_year_wind['Commision Year'] = commision_year_wind['Commision Year'].astype('Int16')   # Convert to nullable integer type
    commision_year_wind = commision_year_wind.dropna(subset=['Commision Year'])
    valid_wind_sites = commision_year_wind['Site Name']
    wind_idx['Removing No commission year'] = re_wind_points.index[~re_wind_points['Site Name'].isin(valid_wind_sites)]
    wind_idx['Final'] = re_wind_points.index[re_wind_points['Site Name'].isin(valid_wind_sites)]
    re_wind_points = re_wind_points.loc[wind_idx['Final']].copy()
else:
    re_wind_df.drop('geometry', axis=1).to_csv(RE_DIR / 'processed/renewable_existing_capacity_commision_year_wind_points.csv', index=False)
    wind_idx['Removing No commission year'] = re_wind_points.index[[]]
    
commision_years = range(
    min(commision_year_solar['Commision Year'].min(), commision_year_wind['Commision Year'].min()), 
    max(commision_year_solar['Commision Year'].max(), commision_year_wind['Commision Year'].max()) + 1
)


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
        + [(f'  Remove: No commission year',        -len(idx['Removing No commission year']))]
        + [(f'Final {tech.lower()} points used',    len(idx['Final']))]
    )
    for _label, _count in _steps:
        _sign = '+' if _count > 0 else ''
        print(f"{_label:<55} {_sign}{_count:>5}")



# -----------------------------------------------------------------------
# Calculate effective capacity, save filtered points for reference
# -----------------------------------------------------------------------
coords_x_solar = xr.DataArray(re_solar_points.geometry.x.values, dims='cell')
coords_y_solar = xr.DataArray(re_solar_points.geometry.y.values, dims='cell')
coords_x_wind  = xr.DataArray(re_wind_points.geometry.x.values, dims='cell')
coords_y_wind  = xr.DataArray(re_wind_points.geometry.y.values, dims='cell')

re_solar_points['Capacity (MW)'] = (
    re_solar_points['Capacity (MW)']
    * CF_SOLAR.interp(x=coords_x_solar, y=coords_y_solar, method='nearest').values
)
re_wind_points['Capacity (MW)']  = (
    re_wind_points['Capacity (MW)']
    * CF_WIND.interp(x=coords_x_wind, y=coords_y_wind, method='nearest').values
)

re_solar_points.to_file(RE_DIR / '20260305/Network Map Renewables January 2026_AUS points_solar_filtered.gpkg')
re_wind_points.to_file(RE_DIR / '20260305/Network Map Renewables January 2026_AUS points_wind_filtered.gpkg')



# -----------------------------------------------------------------------
# Get the capacity MW
# -----------------------------------------------------------------------

# Dissolve overlapping/touching polygons into unified spatial groups to avoid
#   double-counting when multiple projects (e.g. Stage 1 + Stage 2) share the same land.
#   Capacity is summed across all sites in each group; commission year is taken as the
#   earliest year in the group (first stage operational = when the cell becomes active).

capacity_per_ha_solar_rows = []
capacity_per_ha_wind_rows  = []

for polys_raw, points_filtered, commision_year_df, out_rows in [
    (solar_polys_raw, re_solar_points, commision_year_solar, capacity_per_ha_solar_rows),
    (wind_polys_raw,  re_wind_points,  commision_year_wind,  capacity_per_ha_wind_rows),
]:
    # 1. Filter polys to sites with known capacity, fix self-intersections
    polys = (
        polys_raw[polys_raw['Site Name']
        .isin(points_filtered['Site Name'])][['Site Name', 'geometry']]
        .merge(points_filtered[['Site Name', 'Capacity (MW)']], on='Site Name')
        .merge(commision_year_df[['Site Name', 'Commision Year']], on='Site Name')
        .reset_index(drop=True)
    )
    polys['geometry'] = [unary_union(shp.buffer(0)) for shp in polys['geometry']]

    # 2. Find and dissolve intersecting polygons into single groups
    remaining = list(polys.index)
    while remaining:
        i = remaining.pop(0)
        geom      = polys.at[i, 'geometry']
        capacity  = polys.at[i, 'Capacity (MW)']
        comm_year = polys.at[i, 'Commision Year']

        merged = True
        while merged:
            merged   = False
            absorbed = []
            for j in remaining:
                if geom.intersects(polys.at[j, 'geometry']):
                    geom      = unary_union([geom, polys.at[j, 'geometry']])
                    capacity += polys.at[j, 'Capacity (MW)']
                    comm_year = min(comm_year, polys.at[j, 'Commision Year'])
                    absorbed.append(j)
                    merged = True
            for j in absorbed:
                remaining.remove(j)

        out_rows.append({'geometry': geom, 'Capacity (MW)': capacity, 'Commision Year': comm_year})

capacity_per_ha_solar = gpd.GeoDataFrame(capacity_per_ha_solar_rows, crs=solar_polys_raw.crs)
capacity_per_ha_solar['area_ha'] = capacity_per_ha_solar.geometry.to_crs('EPSG:3577').area / 10_000
capacity_per_ha_solar['Capacity per ha (MW/ha)'] = capacity_per_ha_solar['Capacity (MW)'] / capacity_per_ha_solar['area_ha']

capacity_per_ha_wind = gpd.GeoDataFrame(capacity_per_ha_wind_rows, crs=wind_polys_raw.crs)
capacity_per_ha_wind['area_ha'] = capacity_per_ha_wind.geometry.to_crs('EPSG:3577').area / 10_000
capacity_per_ha_wind['Capacity per ha (MW/ha)'] = capacity_per_ha_wind['Capacity (MW)'] / capacity_per_ha_wind['area_ha']


# -----------------------------------------------------------------------
# Solar — polygon-based generation estimates
# -----------------------------------------------------------------------

# Intersect → fragment area within each LUTO cell
isect = gpd.overlay(capacity_per_ha_solar, luto_cells, how='intersection')
isect['area_insec'] = isect.geometry.to_crs('EPSG:3577').area / 10_000   # ha
isect['capacity_MW_insec'] = isect['area_insec'] * isect['Capacity per ha (MW/ha)']


# Get capacity in each cell
solar_MW_xr = xr.DataArray(
    np.where(ref_mask, 0.0, np.nan).astype(np.float32),
    dims=('y', 'x'),
    coords={
        'y': luto_template_xr.y,
        'x': luto_template_xr.x
    }
).expand_dims({'year': commision_years}, axis=0).copy()


for (yr, r, c), prop in isect.groupby(['Commision Year', 'row', 'col'])['capacity_MW_insec'].sum().items():
    solar_MW_xr.loc[dict(year=yr, y=luto_template_xr.y[r], x=luto_template_xr.x[c])] = prop

# Get area fraction of solar within each cell
isect['area_insec_frac'] = isect['area_insec'] / isect['area_ha_cell']   # sanity check — should be <=1 for all rows
cell_frac_solar = isect.groupby(['row', 'col'])['area_insec_frac'].sum().reset_index()

solar_frac_xr = xr.DataArray(
    np.where(ref_mask, 0.0, np.nan).astype(np.float32),
    dims=('y', 'x'),
    coords={
        'y': luto_template_xr.y,
        'x': luto_template_xr.x
    }
)

for _, row in cell_frac_solar.iterrows():
    solar_frac_xr[int(row['row']), int(row['col'])] = row['area_insec_frac']



# -----------------------------------------------------------------------
# Wind — polygon-based generation estimates
# -----------------------------------------------------------------------

# Intersect → fragment area within each LUTO cell
isect_wind = gpd.overlay(capacity_per_ha_wind, luto_cells, how='intersection')
isect_wind['area_insec'] = isect_wind.geometry.to_crs('EPSG:3577').area / 10_000   # ha
isect_wind['capacity_MW_insec'] = isect_wind['area_insec'] * isect_wind['Capacity per ha (MW/ha)']

# Get capacity in each cell
wind_MW_xr = xr.DataArray(
    np.where(ref_mask, 0.0, np.nan).astype(np.float32),
    dims=('y', 'x'),
    coords={
        'y': luto_template_xr.y,
        'x': luto_template_xr.x
    }
).expand_dims({'year': commision_years}, axis=0).copy()

for (yr, r, c), prop in isect_wind.groupby(['Commision Year', 'row', 'col'])['capacity_MW_insec'].sum().items():
    wind_MW_xr.loc[dict(year=yr, y=luto_template_xr.y[r], x=luto_template_xr.x[c])] = prop

# Get area fraction of wind within each cell
isect_wind['area_insec_frac'] = isect_wind['area_insec'] / isect_wind['area_ha_cell']   # sanity check — should be <=1 for all rows
cell_frac_wind = isect_wind.groupby(['row', 'col'])['area_insec_frac'].sum().reset_index()

wind_frac_xr = xr.DataArray(
    np.where(ref_mask, 0.0, np.nan).astype(np.float32),
    dims=('y', 'x'),
    coords={
        'y': luto_template_xr.y,
        'x': luto_template_xr.x
    }
)

for _, row in cell_frac_wind.iterrows():
    wind_frac_xr[int(row['row']), int(row['col'])] = row['area_insec_frac']



# -----------------------------------------------------------------------
# Save xarray outputs
# -----------------------------------------------------------------------

# Save 2D
existing_re_capacity = xr.concat(
    [
        wind_MW_xr.expand_dims({'tech_name': ['Onshore Wind']}),
        solar_MW_xr.expand_dims({'tech_name': ['Utility Solar PV']})
    ],
    dim='tech_name'
)
existing_re_capacity.name = 'capacity_MW'
existing_re_capacity.to_netcdf(
    RE_DIR / 'processed/renewable_existing_capacity_MW_2D.nc',
    encoding={'capacity_MW': {'dtype': 'float32', 'zlib': True, 'complevel': 5}}
)


existing_re_frac = xr.concat(
    [
        wind_frac_xr.expand_dims({'tech_name': ['Onshore Wind']}),
        solar_frac_xr.expand_dims({'tech_name': ['Utility Solar PV']})
    ],
    dim='tech_name'
)
existing_re_frac.name = 'capacity_area_fraction'
existing_re_frac.to_netcdf(
    RE_DIR / 'processed/renewable_existing_capacity_area_fraction_2D.nc',
    encoding={'capacity_area_fraction': {'dtype': 'float32', 'zlib': True, 'complevel': 5}}
)


# Save 1D
existing_re_capacity_1D = (
    existing_re_capacity.stack(z=('y', 'x'))
    .sel(z=~luto_template_xr.stack(z=('y', 'x')).isnull())
    .drop_vars('z')
    .rename({'z': 'cell'})
    .assign_coords(cell=range(len(luto_cells)))
)
existing_re_capacity_1D.to_netcdf(
    RE_DIR / 'processed/renewable_existing_capacity_MW_1D.nc',
    encoding={'capacity_MW': {'dtype': 'float32', 'zlib': True, 'complevel': 5}}
)


existing_re_frac_1D = (
    existing_re_frac.stack(z=('y', 'x'))
    .sel(z=~luto_template_xr.stack(z=('y', 'x')).isnull())
    .drop_vars('z')
    .rename({'z': 'cell'})
    .assign_coords(cell=range(len(luto_cells)))
)
existing_re_frac_1D.to_netcdf(
    RE_DIR / 'processed/renewable_existing_capacity_area_fraction_1D.nc',
    encoding={'capacity_area_fraction': {'dtype': 'float32', 'zlib': True, 'complevel': 5}}
)







