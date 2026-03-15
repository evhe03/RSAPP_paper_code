# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 17:59:38 2026

@author: Niklas
"""


import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np
import geopandas as gpd
import rioxarray  # für rio-Accessor
from shapely.geometry import mapping


ds=xr.open_dataset(r"D:\master\rsap\paper\results\data\era5_data_merge_2.nc")
ds=ds.rename({'valid_time':'time'})

print(list(ds.data_vars))

ds_river=xr.open_dataset(r"D:\master\rsap\paper\results\data\raw_river_discharge.nc")
ds_river=ds_river.rename({'valid_time':'time'})
aoi= gpd.read_file(r"D:\master\rsap\paper\results\data\Shapefile\Catchment.shp")

#clipping
aoi = aoi.to_crs("EPSG:4326")
ds = ds.rio.write_crs("EPSG:4326")

ds_clipped = ds.rio.clip(aoi.geometry.apply(mapping), aoi.crs, drop=True)

print(ds_clipped)


#color shema for diffrent variables 
VAR_COLORS = {
    "dis24": "GnBu",
    "tp": "YlGnBu",
    "t2m": "RdYlBu_r",
    "swvl1": "BrBG",
    "rowe": "viridis",
    "swir": "YlGnBu", 
    "e": "YlGn",   
    "ro": "Blues"
}
name_var={
    "tp": "Total precipitation",
    "e":"Evapotranspiration",
    "smlt" : "Snow Melt",
    "ro": "Runoff",
    "swvl1": "Soil Water Level (0-7cm)",
    "swvl2": "Soil Water Level (7-28cm)",
    "tp_3d": "Total precipitation last 3 Days",
    "tp_5d": "Total precipitation last 5 Days",
    "tp_7d": "Total precipitation last 7 Days",
    "tp_14d": "Total precipitation last 14 Days",
    "tp_30d": "Total precipitation last 30 Days"
    }

#making dis24 nan where 0, because its describing the river discharge and 0 are no rivers 
ds_river["dis24"]=ds_river["dis24"].where(ds_river["dis24"] != 0)


#masking so the dis24 only has values where actually rivers are 
river_mask=ds_river["dis24"].mean(dim='time')>10

#target location for the river discharge, taken from google maps 
target_lat = 26   # change to your river location
target_lon = 90   # change to your river location

#river_ts2 = ds['dis24'].sel(latitude=target_lat, longitude=target_lon, method='nearest')

# 1. Define the bounding box for the 3x3 window
# Note: Adjust the 0.25 offset based on your dataset's actual resolution
lat_slice = slice(target_lat + 0.15, target_lat - 0.15) 
lon_slice = slice(target_lon - 0.15, target_lon + 0.15)


# 2. Select the region
region = ds_river['dis24'].sel(latitude=lat_slice, longitude=lon_slice)

# 3. Mask zeros and calculate the mean across the spatial dimensions
# .where(region > 0) replaces 0s with NaN, so they aren't counted in the mean
river_ts = region.where(region > 1000).mean(dim=['latitude', 'longitude'])


#plotting of the river discharge at 26, 90 as a representive point of flow

plt.figure(figsize=(10, 5))
plt.plot(river_ts.time, river_ts.values, color='red', linewidth=1.5)

plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel("Date")
plt.ylabel("Discharge")
plt.title("Time Series Analysis")
plt.show()

    
def plot_corr_r2_thresholds(field_var, target_ts, lags=(0, 1, 2, 3), var_name=None,
                             dim="time", save_path=None):
    """
    Plots 2x2 grid:
      [0,0] Full correlation map
      [0,1] Correlation masked R² >= 0.25 (|r| >= 0.50)
      [1,0] Correlation masked R² >= 0.50 (|r| >= 0.71)
      [1,1] R² map at best lag
    """
    # --- Resolve display name ---
    raw_name     = field_var.name
    display_name = name_var.get(raw_name, raw_name)

    # --- Find best lag ---
    r2_list = []
    for lag in lags:
        lagged = field_var.shift({dim: lag})
        r  = xr.corr(lagged, target_ts, dim=dim)
        r2 = (r ** 2).assign_coords(lag=lag).expand_dims("lag")
        r2_list.append(r2)

    r2_all       = xr.concat(r2_list, dim="lag")
    mean_r2      = r2_all.mean(dim=["latitude", "longitude"])
    best_lag     = int(mean_r2.idxmax(dim="lag").values)
    best_mean_r2 = float(mean_r2.sel(lag=best_lag).values)

    # --- Compute maps at best lag ---
    lagged_best = field_var.shift({dim: best_lag})
    corr_map    = xr.corr(lagged_best, target_ts, dim=dim)
    r2_map      = corr_map ** 2

    # --- Setup 2x2 grid ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    def add_markers(ax):
        ax.scatter(target_lon, target_lat, color='deepskyblue', edgecolors='black',
                   marker='v', s=100, zorder=7, linewidth=1.2)
        aoi.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5, zorder=5)

    # --- [0,0] Full correlation ---
    im_corr = corr_map.plot(ax=axes[0,0], cmap="RdBu_r", vmin=-1, vmax=1, add_colorbar=False)
    add_markers(axes[0,0])
    axes[0,0].set_title(f"{display_name}  |  lag={best_lag}  |  Full correlation")

    # --- [0,1] Masked R² >= 0.25 ---
    corr_map.where(r2_map >= 0.25).plot(ax=axes[0,1], cmap="RdBu_r", vmin=-1, vmax=1, add_colorbar=False)
    add_markers(axes[0,1])
    axes[0,1].set_title(f"{display_name}  |  lag={best_lag}  |  r² ≥ 0.25")

    # --- [1,0] Masked R² >= 0.50 ---
    corr_map.where(r2_map >= 0.50).plot(ax=axes[1,0], cmap="RdBu_r", vmin=-1, vmax=1, add_colorbar=False)
    add_markers(axes[1,0])
    axes[1,0].set_title(f"{display_name}  |  lag={best_lag}  |  r² ≥ 0.50")

    # --- [1,1] R² map ---
    im_r2 = r2_map.plot(ax=axes[1,1], cmap="YlOrRd", vmin=0, vmax=1, add_colorbar=False)
    add_markers(axes[1,1])
    axes[1,1].set_title(f" r² |  {display_name}  |  lag={best_lag}  |  mean={best_mean_r2:.3f}")
    
    for i, ax in enumerate(axes.flat):
        # only left column gets y-label
        if i % 2 != 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("Latitude [°N]")
        # only bottom row gets x-label
        if i < 2:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Longitude [°E]")
    '''
    # --- Colorbars ---
    # shared correlation colorbar — bottom left
    cbar_ax1 = fig.add_axes([0.05, 0.02, 0.4, 0.02])  # [left, bottom, width, height]
    fig.colorbar(im_corr, cax=cbar_ax1, orientation='horizontal', label='Correlation (r)')
    
    fig.colorbar(im_corr, ax=axes[0,0], orientation='horizontal', label='Correlation (r)')    # R² colorbar — bottom right
    fig.colorbar(im_corr, ax=axes[1,0], orientation='horizontal', label='Correlation (r)')    # R² colorbar — bottom right
    fig.colorbar(im_corr, ax=axes[0,1], orientation='horizontal', label='Correlation (r)')    # R² colorbar — bottom right
    fig.colorbar(im_r2, ax=axes[1,1], orientation='horizontal', label='R²')    # R² colorbar — bottom right
    
    cbar_ax2 = fig.add_axes([0.55, 0.02, 0.4, 0.02])
    fig.colorbar(im_r2, cax=cbar_ax2, orientation='horizontal', label='R²')
    '''
    # keep only these four:
    fig.colorbar(im_corr, ax=axes[0,0], orientation='horizontal', shrink=0.8, pad=0.05,aspect=30, label='Correlation (r)')
    fig.colorbar(im_corr, ax=axes[0,1], orientation='horizontal', shrink=0.8, pad=0.05,aspect=30, label='Correlation (r)')
    fig.colorbar(im_corr, ax=axes[1,0], orientation='horizontal', shrink=0.8, pad=0.05,aspect=30, label='Correlation (r)')
    fig.colorbar(im_r2,   ax=axes[1,1], orientation='horizontal', shrink=0.8, pad=0.05,aspect=30, label='R²')
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


for var_name in ds_clipped.data_vars:
    plot_corr_r2_thresholds(
        ds_clipped[var_name],
        river_ts,
        save_path=rf"D:\master\rsap\paper\results\plots\finals{var_name}_corr_r2_combi.png"
        )

def plot_corr_masked_by_r2(field_var, target_ts, lags=(0, 1, 2, 3), var_name=None, 
                            dim="time", r2_threshold=0.25, save_path=None):
    """
    Plots correlation map at best lag, masked to only show pixels where R² >= r2_threshold.
    
    Parameters:
        field_var     : xarray DataArray (the climate variable)
        target_ts     : xarray DataArray (river discharge time series)
        lags          : tuple of lags to search over
        var_name      : display name for titles
        dim           : time dimension name
        r2_threshold  : minimum R² value to display (default=0.25 → r>0.5)
        save_path     : optional file path to save the figure
    """
    #var_name 
    raw_name = var_name or (field_var.name if field_var.name else "Variable")
    display_name = name_var.get(raw_name, raw_name) if name_var else raw_name
    
    # --- Find best lag (same logic as before) ---
    r2_list = []
    for lag in lags:
        lagged = field_var.shift({dim: lag})
        r = xr.corr(lagged, target_ts, dim=dim)
        r2 = (r ** 2).assign_coords(lag=lag).expand_dims("lag")
        r2_list.append(r2)

    r2_all = xr.concat(r2_list, dim="lag")
    mean_r2 = r2_all.mean(dim=["latitude", "longitude"])
    best_lag = int(mean_r2.idxmax(dim="lag").values)
    best_mean_r2 = float(mean_r2.sel(lag=best_lag).values)

    # --- Compute maps at best lag ---
    lagged_best = field_var.shift({dim: best_lag})
    corr_map = xr.corr(lagged_best, target_ts, dim=dim)
    r2_map   = corr_map ** 2

    # --- Mask: only show pixels where R² >= threshold ---
    corr_masked = corr_map.where(r2_map >= r2_threshold)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    im = corr_masked.plot(
        ax=ax, cmap="RdBu_r", vmin=-1, vmax=1, add_colorbar=False
    )

    # grey background for masked-out (low R²) pixels
    ax.set_facecolor("#d3d3d3")

    ax.scatter(target_lon, target_lat, color='deepskyblue', edgecolors='black',
               marker='v', s=100, zorder=7, linewidth=1.2)
    aoi.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5, zorder=5)

    ax.set_title(
        f"{display_name}  |  lag={best_lag}  | R²≥{r2_threshold:.2f}"
    )

    fig.colorbar(im, ax=ax, orientation='horizontal', shrink=0.6, 
                 aspect=25, pad=0.08, label='Correlation (r)')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()

for var_name in ds_clipped.data_vars:
    plot_corr_masked_by_r2(
        ds_clipped[var_name],
        river_ts,
        r2_threshold=0.5
        #save_path=rf"D:\master\rsap\paper\results\plots\correlation_plots\correlation_{var_name}.png"
        )

def plot_corr_masked_by_r2_all_lags(field_var, target_ts, lags=(0, 1, 2, 3), var_name=None, 
                            dim="time", r2_threshold=0.25, save_path=None):
    raw_name = var_name or (field_var.name if field_var.name else "Variable")
    display_name = name_var.get(raw_name, raw_name) if name_var else raw_name
    
    # --- Compute correlation and R² for ALL lags ---
    corr_maps = {}
    r2_maps = {}
    for lag in lags:
        lagged = field_var.shift({dim: lag})
        r = xr.corr(lagged, target_ts, dim=dim)
        corr_maps[lag] = r
        r2_maps[lag] = r ** 2

    # --- Find best lag by mean R² ---
    mean_r2_per_lag = {lag: float(r2_maps[lag].mean().values) for lag in lags}
    best_lag = max(mean_r2_per_lag, key=mean_r2_per_lag.get)

    # --- Plot: 2x2 subplots, one per lag ---
    ncols = 2
    nrows = math.ceil(len(lags) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8), constrained_layout=True)
    axes = axes.flatten()

    for i, lag in enumerate(lags):
        ax = axes[i]
        corr_masked = corr_maps[lag].where(r2_maps[lag] >= r2_threshold)

        im = corr_masked.plot(
            ax=ax, cmap="RdBu_r", vmin=-1, vmax=1, add_colorbar=False
        )
        ax.set_facecolor("white")
        ax.scatter(target_lon, target_lat, color='deepskyblue', edgecolors='black',
                   marker='v', s=100, zorder=7, linewidth=1.2)
        aoi.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5, zorder=5)

        # Highlight the best lag subplot
        title_suffix = "  ★ best" if lag == best_lag else ""
        ax.set_title(
            f"{display_name}  |  lag={lag}  |  r²≥{r2_threshold:.2f}{title_suffix}",
            fontweight='bold' if lag == best_lag else 'normal'
        )

    # Hide any unused axes (if lags count is odd)
    for j in range(len(lags), len(axes)):
        axes[j].set_visible(False)

    # Shared colorbar
    fig.colorbar(im, ax=axes, orientation='vertical', shrink=0.6,
                 aspect=30, pad=0.02, label='Correlation (r)')

    fig.suptitle(f"{display_name}", 
                 fontsize=13, fontweight='bold')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    
for var_name in ds_clipped.data_vars:
    plot_corr_masked_by_r2_all_lags(
        ds_clipped[var_name],
        river_ts,
        r2_threshold=0.5,
        save_path=rf"D:\master\rsap\paper\results\plots\finals\lags\correlation_{var_name}.png"
        )

def final_plot(field_var, target_ts, lags=(0, 1, 2, 3), var_name=None, 
                            dim="time", r2_threshold=0.25, save_path=None):
    raw_name = var_name or (field_var.name if field_var.name else "Variable")
    display_name = name_var.get(raw_name, raw_name) if name_var else raw_name
    
    # --- Compute correlation and R² for ALL lags ---
    corr_maps = {}
    r2_maps = {}
    for lag in lags:
        lagged = field_var.shift({dim: lag})
        r = xr.corr(lagged, target_ts, dim=dim)
        corr_maps[lag] = r
        r2_maps[lag] = r ** 2

    # --- Find best lag by mean R² ---
    mean_r2_per_lag = {lag: float(r2_maps[lag].mean().values) for lag in lags}
    best_lag = max(mean_r2_per_lag, key=mean_r2_per_lag.get)

    # --- Full (unmasked) maps: best lag ---
    corr_full = corr_maps[best_lag]
    r2_full   = r2_maps[best_lag]

    # --- Layout: 3 rows x 2 cols ---
    fig = plt.figure(figsize=(13, 14), constrained_layout=True)
    gs  = fig.add_gridspec(3, 2)

    ax_corr_full = fig.add_subplot(gs[0, 0])  # top-left:  full correlation
    ax_r2_full   = fig.add_subplot(gs[0, 1])  # top-right: full R²
    lag_axes     = [fig.add_subplot(gs[1 + i // 2, i % 2]) for i in range(len(lags))]

    # ── helper: scatter + AOI overlay ────────────────────────────────────────
    def _overlay(ax):
        ax.scatter(target_lon, target_lat, color='deepskyblue', edgecolors='black',
                   marker='v', s=100, zorder=7, linewidth=1.2)
        aoi.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5, zorder=5)

    # ── Row 0: full unmasked maps ─────────────────────────────────────────────
    im_corr = corr_full.plot(ax=ax_corr_full, cmap="RdBu_r", vmin=-1, vmax=1,
                              add_colorbar=False)
    ax_corr_full.set_facecolor("#d3d3d3")
    _overlay(ax_corr_full)
    ax_corr_full.set_title(f"{display_name}  |  Full Correlation  |  lag={best_lag}")

    im_r2 = r2_full.plot(ax=ax_r2_full, cmap="YlOrRd", vmin=0, vmax=1,
                          add_colorbar=False)
    ax_r2_full.set_facecolor("#d3d3d3")
    _overlay(ax_r2_full)
    ax_r2_full.set_title(f"{display_name}  |  Full r²  |  lag={best_lag}")

    # ── Row 1–2: masked lag subplots ──────────────────────────────────────────
    for i, lag in enumerate(lags):
        ax = lag_axes[i]
        corr_masked = corr_maps[lag].where(r2_maps[lag] >= r2_threshold)
        im_lag = corr_masked.plot(ax=ax, cmap="RdBu_r", vmin=-1, vmax=1,
                                   add_colorbar=False)
        ax.set_facecolor("#d3d3d3")
        _overlay(ax)
        title_suffix = "  ★ best" if lag == best_lag else ""
        ax.set_title(
            f"{display_name}  |  lag={lag}  |  r²≥{r2_threshold:.2f}{title_suffix}",
            fontweight='bold' if lag == best_lag else 'normal'
        )

    # ── Colorbars ─────────────────────────────────────────────────────────────
    fig.colorbar(im_corr, ax=ax_corr_full, orientation='horizontal',
                 shrink=0.8, aspect=25, pad=0.05, label='Correlation (r)')
    fig.colorbar(im_r2, ax=ax_r2_full, orientation='horizontal',
                 shrink=0.8, aspect=25, pad=0.05, label='r²')
    fig.colorbar(im_lag, ax=lag_axes, orientation='vertical',
                 shrink=0.6, aspect=30, pad=0.02, label='Correlation (r)')

    fig.suptitle(f"{display_name}  |  r²≥{r2_threshold:.2f}  |  best lag={best_lag}",
                 fontsize=13, fontweight='bold')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


for var_name in ds_clipped.data_vars:
    final_plot(
        ds_clipped[var_name],
        river_ts,
        r2_threshold=0.5,
        save_path=rf"D:\master\rsap\paper\results\plots\corr_masked_all_legs\correlation_{var_name}.png"
        )