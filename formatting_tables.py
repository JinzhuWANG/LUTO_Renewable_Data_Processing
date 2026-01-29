import pandas as pd
import shutil


# Define data root path
data_root = 'N:/Data-Master/Renewable Energy'



# ---------------------------------------------------
#               Electricity Price Table              
# ---------------------------------------------------
# TODO: rename state to full names to match other tables

df = pd.read_csv(f'{data_root}/20260127/elec_price_forecast.csv')

# Filter to Electricity sector if present, index by State
if 'Sector' in df.columns:
    df = df[df['Sector'] == 'Electricity']
df = df.set_index('State')

# Keep only year columns, convert c/kWh -> $/MWh (x10), transpose
year_cols = [c for c in df.columns if str(c).isdigit()]
df = df[year_cols].astype(float).mul(10).T
df.index = df.index.astype(int)
df.index.name = 'Year'

# Save
df.to_csv(f'{data_root}/processed/renewable_elec_price_AUD_MWh.csv')



# ---------------------------------------------------
#                       Targets     
# ---------------------------------------------------
'''
Just need to rename the abbreviated state names to full names to match other tables.
'''
# TODO: rename states to full names to match other tables
state_rename = {
    'NSW': 'New South Wales',
    'VIC': 'Victoria',
    'QLD': 'Queensland',
    'SA': 'South Australia',
    'WA': 'Western Australia',
    'TAS': 'Tasmania',
    'NT': 'Northern Territory',
    'ACT': 'Australian Capital Territory'
}

re_targets = pd.read_csv(f'{data_root}/20260127/renewable_targets.csv')
re_targets['STATE'] = re_targets['STATE'].map(state_rename)
re_targets.to_csv(f'{data_root}/processed/renewable_targets.csv', index=False)




# ---------------------------------------------------
#        Bundle Onshore-Wind / Solar Data 
# ---------------------------------------------------

REQUIRED_COLUMNS = [
    'Productivity', 
    'Establishment_Cost_Multiplier', 
    'OM_Cost_Multiplier', 
    'Revenue',
    'Annual Cost Per Ha (A$2010/yr)', 
    'Biodiversity_compatability',
    'IMPACTS_water_retention', 
    'INPUT-wrt_water-required'
]


# Load and map onshore wind data
onshore_wind_file = f'{data_root}/20260127/20260105_Bundle_Wind.xlsx'
wind_sheet_names = ['cropping', 'horticulture', 'livestock', 'unallocated']

onshore_wind_df = pd.DataFrame()
for category in wind_sheet_names:
    data = pd.read_excel(onshore_wind_file, sheet_name=f'Onshore Wind ({category})', index_col='Year')

    missing_cols = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing_cols:
        print(f"Warning: Missing columns in {category} wind data: {missing_cols}")

    onshore_wind_df = pd.concat([onshore_wind_df, data])



# Load and map solar PV data
solar_pv_file = f'{data_root}/20260127/20260105_Bundle_SPV.xlsx'
wind_sheet_names = ['cropping',  'livestock', 'unallocated']  # No 'horticulture' sheet for solar

solar_df = pd.DataFrame()
for category in wind_sheet_names:
    data = pd.read_excel(solar_pv_file, sheet_name=f'Solar PV ({category})', index_col='Year')

    missing_cols = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing_cols:
        print(f"Warning: Missing columns in {category} solar PV data: {missing_cols}")

    solar_df = pd.concat([solar_df, data])


# Save bundled data
renewable_bundle_df = pd.concat([onshore_wind_df, solar_df], axis=0).reset_index()
renewable_bundle_df.to_csv(f'{data_root}/processed/renewable_energy_bundle.csv', index=False)













