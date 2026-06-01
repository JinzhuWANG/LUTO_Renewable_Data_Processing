import geopandas as gpd
import json


# read file 
shp_aemo = "N:/Data-Master/Renewable Energy/20260520_REM_shp_overlay/aemo-indicative-rez-boundaries-2025/AEMO 2025 Indicative REZ boundaries-POLYGON.shp"
gpd_aemo = gpd.read_file(shp_aemo).query('descriptio == "REZ"').reset_index(drop=True)


# Save to shp
gpd_aemo_simplified = gpd_aemo.copy()
gpd_aemo_simplified['geometry'] = gpd_aemo_simplified.simplify(0.1)
gpd_aemo_simplified.to_file("N:/Data-Master/Renewable Energy/processed/REZ_boundary/aemo_rez_boundaries_2025.shp", driver='ESRI Shapefile')

