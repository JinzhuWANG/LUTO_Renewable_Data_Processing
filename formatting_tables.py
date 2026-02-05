import pandas as pd
from itertools import product


# Define data root path
data_root = 'N:/Data-Master/Renewable Energy'

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


# ---------------------------------------------------
#               Electricity Price Table              
# ---------------------------------------------------

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

# Rename states to full names
df = df.rename(columns=state_rename).reset_index()
df = df.melt(id_vars='Year', var_name='State', value_name='Price_AUD_per_MWh').set_index(['Year', 'State']).reset_index()


# Fill price between 2010 and 2021 with 2021 prices
for state in df['State'].unique():
    price_2021 = df[(df['Year'] == 2021) & (df['State'] == state)]['Price_AUD_per_MWh'].values[0]
    for year in range(2010, 2021):
        df = pd.concat([df, pd.DataFrame({'Year': [year], 'State': [state], 'Price_AUD_per_MWh': [price_2021]})], ignore_index=True)


# Save
df = df.sort_values(['State', 'Year'])
df.to_csv(f'{data_root}/processed/renewable_elec_price_AUD_MWh.csv', index=False)



# ---------------------------------------------------
#                       Targets     
# ---------------------------------------------------
'''
Just need to rename the abbreviated state names to full names to match other tables.
'''

re_targets = pd.read_csv(f'{data_root}/20260127/renewable_targets.csv')
re_targets['STATE'] = re_targets['STATE'].map(state_rename)
re_targets = re_targets.melt(
    id_vars=['SCENARIO', 'STATE', 'PRODUCT'], 
    value_vars=[str(i) for i in range(2020, 2051)], 
    var_name='Year', 
    value_name='Renewable_Target_TWh'
)
re_targets['Year'] = re_targets['Year'].astype(int)


# Linearly interpolate year 2010-2020 from 0 in 2010 to 2020 target in 2020
rows = []
for scneario, state, prod in product(re_targets['SCENARIO'].unique(), re_targets['STATE'].unique(), re_targets['PRODUCT'].unique()):
    t = re_targets.query("SCENARIO == @scneario & STATE == @state & PRODUCT == @prod & Year == 2020")['Renewable_Target_TWh'].values[0]
    rows += [{
        'SCENARIO': scneario,
        'STATE': state,
        'PRODUCT': prod,
        'Year': y,
        'Renewable_Target_TWh': t * (y - 2010) / 10
        } for y in range(2010, 2020) ]
    
re_targets = (
    pd.concat([re_targets, pd.DataFrame(rows)], ignore_index=True)
    .sort_values(['SCENARIO', 'STATE', 'PRODUCT', 'Year'])
    .reset_index(drop=True)
)

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


# Save bundled data; Capitalise the 'Commodity' column for consistency
onshore_wind_df['Commodity'] = onshore_wind_df['Commodity'].str.capitalize()
solar_df['Commodity'] = solar_df['Commodity'].str.capitalize()
renewable_bundle_df = pd.concat([onshore_wind_df, solar_df], axis=0).reset_index()
renewable_bundle_df.to_csv(f'{data_root}/processed/renewable_energy_bundle.csv', index=False)













