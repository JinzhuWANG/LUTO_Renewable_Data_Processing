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


# Solar
df = pd.read_csv(f'{data_root}/20260210/solar_elec_price_AUD_MWh.csv')
df = df.rename(columns=state_rename).reset_index(drop=True)
df = df.melt(id_vars='Year', var_name='State', value_name='Price_AUD_per_MWh').set_index(['Year', 'State']).reset_index()

for state in df['State'].unique():
    price_2021 = df[(df['Year'] == 2021) & (df['State'] == state)]['Price_AUD_per_MWh'].values[0]
    for year in range(2010, 2021):
        df = pd.concat([df, pd.DataFrame({'Year': [year], 'State': [state], 'Price_AUD_per_MWh': [price_2021]})], ignore_index=True)

# Remove ACT from solar price table (ACT is within NSW for LUTO purposes, and we will just use the NSW price for ACT)
df = df[df['State'] != 'Australian Capital Territory'].reset_index(drop=True).sort_values(['State', 'Year'])
df.to_csv(f'{data_root}/processed/renewable_price_AUD_MWh_solar.csv', index=False)



# Wind
df = pd.read_csv(f'{data_root}/20260210/wind_elec_price_AUD_MWh.csv')

df = df.rename(columns=state_rename).reset_index(drop=True)
df = df.melt(id_vars='Year', var_name='State', value_name='Price_AUD_per_MWh').set_index(['Year', 'State']).reset_index()

for state in df['State'].unique():
    price_2021 = df[(df['Year'] == 2021) & (df['State'] == state)]['Price_AUD_per_MWh'].values[0]
    for year in range(2010, 2021):
        df = pd.concat([df, pd.DataFrame({'Year': [year], 'State': [state], 'Price_AUD_per_MWh': [price_2021]})], ignore_index=True)

# Remove ACT from wind price table (ACT is within NSW for LUTO purposes, and we will just use the NSW price for ACT)
df = df[df['State'] != 'Australian Capital Territory'].reset_index(drop=True).sort_values(['State', 'Year'])
df.to_csv(f'{data_root}/processed/renewable_price_AUD_MWh_wind.csv', index=False)



# ---------------------------------------------------
#                       Targets     
# ---------------------------------------------------
'''
Just need to rename the abbreviated state names to full names to match other tables.
'''
re_targets = pd.read_csv(f'{data_root}/20260305/renewable_targets_20260305.csv')
re_targets['state'] = re_targets['state'].map(state_rename)
year_cols = [c for c in re_targets.columns if c.isdigit()]

re_targets = re_targets.melt(
    id_vars=['scen', 'state', 'tech'], 
    value_vars=year_cols,
    var_name='Year', 
    value_name='Renewable_Target_TWh'
)
re_targets['Year'] = re_targets['Year'].astype(int)


# Add ACT targets into NSW (ACT is within NSW for LUTO purposes)
act_targets = re_targets.query("state == 'Australian Capital Territory'").copy()
act_targets['state'] = 'New South Wales'
re_targets = (
    pd.concat([re_targets, act_targets], ignore_index=True)
    .groupby(['scen', 'state', 'tech', 'Year'], as_index=False)['Renewable_Target_TWh']
    .sum()
    .query("state != 'Australian Capital Territory'")
    .reset_index(drop=True)
)

# Linearly interpolate to fill annual years between 5-yr gaps (post-2030)
all_years = list(range(re_targets['Year'].min(), re_targets['Year'].max() + 1))
re_targets = (
    re_targets
    .groupby(['scen', 'state', 'tech'], group_keys=False)
    [['scen', 'state', 'tech', 'Year', 'Renewable_Target_TWh']]
    .apply(lambda df:
        df.set_index('Year')['Renewable_Target_TWh']
        .reindex(all_years)
        .interpolate(method='index')
        .rename_axis('Year')
        .reset_index()
        .assign(scen=df['scen'].iloc[0], state=df['state'].iloc[0], tech=df['tech'].iloc[0])
    )
    .reset_index(drop=True)
    [['scen', 'state', 'tech', 'Year', 'Renewable_Target_TWh']]
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













