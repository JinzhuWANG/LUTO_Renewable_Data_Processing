# Renewable Energy Data Processing

Python scripts for processing and formatting Australian renewable energy data.

## Overview

This repository contains data processing utilities that:

- **Electricity Price Processing**: Converts electricity price forecasts from c/kWh to $/MWh and reformats data by state
- **Renewable Targets Processing**: Standardizes Australian state name abbreviations to full names
- **Renewable Energy Bundling**: Combines onshore wind and solar PV data from multiple Excel sheets into unified datasets

## Data Sources

Input data is expected in `N:/Data-Master/Renewable Energy/20260127/`:
- `elec_price_forecast.csv` - Electricity price forecasts by state
- `renewable_targets.csv` - State renewable energy targets
- `20260105_Bundle_Wind.xlsx` - Onshore wind data by land use category
- `20260105_Bundle_SPV.xlsx` - Solar PV data by land use category

## Output

Processed files are saved to `N:/Data-Master/Renewable Energy/processed/`:
- `renewable_elec_price_AUD_MWh.csv` - Electricity prices in AUD/MWh
- `renewable_targets.csv` - Targets with full state names
- `renewable_energy_bundle.csv` - Combined wind and solar data

## Requirements

- Python 3.x
- pandas
- openpyxl (for Excel file reading)

## Usage

```bash
python formatting_tables.py
```
