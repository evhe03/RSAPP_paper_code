# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 15:05:48 2026

@author: Niklas
"""

import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd


era5_accu = xr.open_dataset(r"D:\master\rsap\paper\correlation\correlation_catchment_data\ERA5_data\data_stream-oper_stepType-accum.nc")

print("Variables in dataset:", list(era5_accu.data_vars))

era5_inst = xr.open_dataset(r"D:\master\rsap\paper\correlation\correlation_catchment_data\ERA5_data\data_stream-oper_stepType-instant.nc")

print("Variables in dataset:", list(era5_inst.data_vars))

gfm = xr.open_dataset(
    r"D:\master\rsap\paper\correlation\correlation_catchment_data\gfm_data\data.grib",
    engine="cfgrib"
)

#temperature prossecing

temp = era5_inst['t2m']

#Data is in Kelvin --> calculating into Celsius and changing units
daily_temp = temp.resample(valid_time='1D').mean() - 273.15
daily_temp.attrs['units'] = '°C'

print(daily_temp)

# soil moisture prossecing

#Definieren der beiden Layer ( Habe nur die oberen 2 genommen, Erklärung ist in meinem Workflow Dokument. Können aber auch die anderen beiden hinzufügen)
sm_layer1 = era5_inst["swvl1"]
sm_layer2 = era5_inst["swvl2"]

#processing data (calculating the mean from all 4 variables)
daily_sm1 = sm_layer1.resample(valid_time='1D').mean()
daily_sm2 = sm_layer2.resample(valid_time='1D').mean()

#percipitation processing


precipitation = era5_accu["tp"]

#taking the max from each day (which is the total amount of precipitation since its accumulated data and converting meter into millimeter
daily_precipitation = precipitation.resample(valid_time= "1 D").sum()*1000
daily_precipitation.attrs["units"] = "mm"

#creating the different "windows"/time slots 
tp_rolling_ds = xr.Dataset()

windows = [3, 5, 7, 14, 30]
for w in windows:
    # Calculate the rolling sum
    data = daily_precipitation.rolling(valid_time=w, min_periods=w).sum()
    
    # Add it to the dataset with a dynamic name
    var_name = f"tp_{w}d"
    tp_rolling_ds[var_name] = data
    
#evapo proccessing
evapo = era5_accu['e']

daily_evapo = evapo.resample(valid_time='1D').sum()

print(daily_evapo)

#runoff processing
runoff = era5_accu['ro']

daily_runoff = runoff.resample(valid_time='1D').sum()

print(daily_runoff)



#ERA5 Data merge: 

era5_data_merge =xr.merge([daily_precipitation, daily_evapo, daily_runoff, daily_sm1,
                           daily_sm2, daily_temp, tp_rolling_ds],join="inner", compat="override")

print(list(era5_data_merge.data_vars))


era5_data_merge.to_netcdf(r"D:\master\rsap\paper\correlation\correlation_catchment_data\ERA5_data/era5_all_data_merge2.nc")


#merge era5 and gfm data

era5_data_merge2=era5_data_merge.rename({'valid_time':'time'})
era5_data_merge2=era5_data_merge2.interp_like(gfm, method="linear")

all_data_gfm_era5_merge=xr.merge([gfm, era5_data_merge2], join="inner", compat="override")

print(list(all_data_gfm_era5_merge.data_vars))
    
all_data_gfm_era5_merge.to_netcdf(r"D:\master\rsap\paper\correlation\correlation_catchment_data\correlation/all_data_merge2.nc")



