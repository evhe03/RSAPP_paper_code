# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 13:06:57 2026

@author: Niklas
"""

import cdsapi
import os

# creating folder for variable
os.makedirs("precipitation", exist_ok=True)

dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "total_precipitation",
        "volumetric_soil_water_layer_1",
        "volumetric_soil_water_layer_2",
        "evaporation",
        "runoff",
        "snowmelt"
    ],
    "year": ["2024"],
    "month": ["05", "06", "07"],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "time": [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
    ],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [ 31, 82,25, 98]
}

client = cdsapi.Client(url=, key=)
client.retrieve(dataset, request).download()
