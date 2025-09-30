#!/usr/bin/env python3
"""
ECMWF Tropical Cyclone Track Data Visualization

This module provides visualization functions for tropical cyclone track data
extracted from ECMWF BUFR files using the ecmwf_tc_data_extractor module.

Data Requirements:
- Input: CSV file from ecmwf_tc_data_extractor module
- Required columns: storm_id, ensemble_member, step, datetime,
  latitude, longitude, pressure, wlatitude, wlongitude, wind,
  wind_threshold, quadrant, bearing_start, bearing_end, wind_radius

Features:
- Interactive track visualization with ensemble members
- Intensity evolution plots (wind speed and pressure)
- Hurricane category thresholds
- Wind field visualization with wind sectors by threshold
"""

import warnings
from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gigaspatial.processing import convert_to_geodataframe, buffer_geodataframe
import geopandas as gpd
from shapely.geometry import LineString, Polygon
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyproj

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration
DEFAULT_DPI = 300
DEFAULT_FIGSIZE = (16, 12)

# Hurricane category thresholds (m/s converted from knots)
HURRICANE_CATEGORIES = {
    17.5: 'TS',    # ~34kt - Tropical Storm
    33.0: 'Cat1',  # ~64kt - Category 1 Hurricane
    42.5: 'Cat2',  # ~83kt - Category 2 Hurricane
    49.5: 'Cat3',  # ~96kt - Category 3 Hurricane
    58.0: 'Cat4',  # ~113kt - Category 4 Hurricane
    70.0: 'Cat5'   # ~137kt - Category 5 Hurricane
}


def ms_to_knots(ms_speed: float) -> float:
    """
    Convert wind speed from meters per second to knots.
    
    Args:
        ms_speed (float): Wind speed in m/s
        
    Returns:
        float: Wind speed in knots
    """
    return ms_speed * 1.94384


def create_track_visualization(csv_file: str, output_filename: Optional[str] = None):
    """
    Create an interactive map visualization of tropical cyclone tracks.
    
    This function creates a Folium map showing:
    - Storm center positions as buffered points
    - Track lines connecting consecutive positions for each ensemble member
    - Interactive features for exploring ensemble forecasts
    
    Args:
        csv_file (str): Path to CSV file from ecmwf_tc_data_extractor
        output_filename (str, optional): Output filename for HTML map (without .html extension)
        
    Returns:
        folium.Map: Interactive map object
    """
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    # Check for required columns
    required_cols = ['ensemble_member', 'latitude', 'longitude']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    # Convert to GeoDataFrame
    gdf = convert_to_geodataframe(df)
    
    # Create buffered points for storm centers (200km buffer)
    gdf_buffer = buffer_geodataframe(gdf, 200)

    # Filter out points near the dateline
    gdf_buffer_filtered = gdf_buffer.copy()

    # Check if the track crosses dateline
    lons = gdf['longitude'].values
    if (lons.max() - lons.min()) > 180:
        # Track crosses dateline - only show points on one side
        # Keep points either in Eastern or Western hemisphere, whichever has more
        eastern = gdf_buffer[gdf_buffer.geometry.centroid.x > 0]
        western = gdf_buffer[gdf_buffer.geometry.centroid.x < 0]

        if len(eastern) > len(western):
            gdf_buffer_filtered = eastern
        else:
            gdf_buffer_filtered = western

        print(f"Dateline crossing detected - showing {len(gdf_buffer_filtered)} of {len(gdf_buffer)} points")


    # Create track lines for each ensemble member
    lines = []
    from_idx = []
    to_idx = []
    member = []

    for m in gdf.ensemble_member.unique():
        gdf_m = gdf[gdf.ensemble_member == m].sort_values('step')
        pts = gdf_m.geometry.to_list()

        # Create line segments between consecutive points
        if len(pts) > 1:
            for i in range(len(pts) - 1):
                lon1 = pts[i].x
                lon2 = pts[i + 1].x

                # Check if crossing dateline (longitude difference > 180)
                if abs(lon2 - lon1) > 180:
                    # Skip this line segment (dateline crossing)
                    continue

                lines.append(LineString([pts[i], pts[i + 1]]))
                from_idx.append(gdf_m.index[i])
                to_idx.append(gdf_m.index[i + 1])
                member.append(m)

    # Create GeoDataFrame for track lines
    gdf_lines = gpd.GeoDataFrame(
        {"from_idx": from_idx, "to_idx": to_idx, 'member': member},
        geometry=lines,
        crs=gdf.crs
    )

    # Separate styling for control members 51 & 52
    gdf_lines['is_control'] = gdf_lines['member'] > 50

    # Create interactive map
    m = gdf_buffer_filtered.explore()

    # Plot regular ensemble members first
    ensemble_lines = gdf_lines[gdf_lines['is_control'] == False]
    if not ensemble_lines.empty:
        ensemble_lines.explore(m=m)

    # Plot control members with red, thick lines
    control_lines = gdf_lines[gdf_lines['is_control'] == True]
    if not control_lines.empty:
        m2 = control_lines.explore(m=m, color='red', style_kwds={'weight': 4, 'opacity': 0.8})
    else:
        m2 = m

    # Save map if filename provided
    if output_filename:
        m2.save(f'{output_filename}.html')
        print(f"Track visualization saved as: {output_filename}.html")
    
    return m2


def create_intensity_plot(csv_file: str, output_filename: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Create intensity evolution plots for tropical cyclone data.
    
    This function creates a two-panel plot showing:
    - Top panel: Wind speed evolution over time with hurricane category thresholds
    - Bottom panel: Pressure evolution over time (inverted scale)
    
    Args:
        csv_file (str): Path to CSV file from ecmwf_tc_data_extractor
        output_filename (str, optional): Output filename for saved plot
        
    Returns:
        matplotlib.figure.Figure or None: Figure object if successful, None if no data
    """
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    # Check for required columns
    required_cols = ['ensemble_member', 'step', 'wind', 'pressure', 'storm_id']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    # Get intensity data (use first quadrant to avoid duplicates if quadrant column exists)
    if 'quadrant' in df.columns:
        intensity_data = df[df['quadrant'] == 1].copy()
    else:
        # If no quadrant column, use all data but remove duplicates
        intensity_data = df.drop_duplicates(subset=['ensemble_member', 'step']).copy()

    if intensity_data.empty:
        print("No intensity data found")
        return None

    # Convert wind speed to knots for plotting
    intensity_data['wind_knots'] = intensity_data['wind'].apply(ms_to_knots)
    
    # Convert pressure to hPa if it's in Pa
    if intensity_data['pressure'].max() > 2000:  # Likely in Pa
        intensity_data['pressure_hpa'] = intensity_data['pressure'] / 100.0
    else:
        intensity_data['pressure_hpa'] = intensity_data['pressure']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=DEFAULT_FIGSIZE, sharex=True, dpi=DEFAULT_DPI)

    # Get ensemble members to plot
    all_members = sorted(intensity_data['ensemble_member'].unique())

    # Plot wind speed evolution
    for member in all_members:
        member_data = intensity_data[intensity_data['ensemble_member'] == member].sort_values('step')
        if len(member_data) > 1:
            # Check if it's a control member
            if member > 50:  # Control members
                ax1.plot(member_data['step'], member_data['wind_knots'],
                         '-', linewidth=3, color='red', alpha=0.9)
            else:  # Regular ensemble members
                ax1.plot(member_data['step'], member_data['wind_knots'],
                         '-', alpha=0.6, linewidth=1)

    # Add hurricane category thresholds
    for threshold_ms, category in HURRICANE_CATEGORIES.items():
        threshold_knots = ms_to_knots(threshold_ms)
        ax1.axhline(threshold_knots, color='red', linestyle='--', alpha=0.5)
        ax1.text(ax1.get_xlim()[1], threshold_knots, f' {category}',
                verticalalignment='center', fontsize=8)

    ax1.set_ylabel('Wind Speed (knots)')
    storm_name = intensity_data["storm_id"].iloc[0] if 'storm_id' in intensity_data.columns else 'Unknown'
    ax1.set_title(f'Tropical Cyclone {storm_name} - Intensity Evolution')
    ax1.grid(True, alpha=0.3)

    # Plot pressure evolution
    for member in all_members:
        member_data = intensity_data[intensity_data['ensemble_member'] == member].sort_values('step')
        if len(member_data) > 1:
            # Check if it's a control member
            if member > 50:  # Control members
                ax2.plot(member_data['step'], member_data['pressure_hpa'],
                         '-', linewidth=3, color='red', alpha=0.9)
            else:  # Regular ensemble members
                ax2.plot(member_data['step'], member_data['pressure_hpa'],
                         '-', alpha=0.6, linewidth=1)

    ax2.set_xlabel('Forecast Hour')
    ax2.set_ylabel('Pressure (hPa)')
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()  # Lower pressure at top

    plt.tight_layout()

    if output_filename:
        plt.savefig(output_filename, dpi=DEFAULT_DPI, bbox_inches='tight')
        print(f"Intensity plot saved as: {output_filename}")

    return fig


def create_wind_field_visualization(csv_file: str, output_filename: Optional[str] = None,
                                    members: Optional[List[int]] = None,
                                    n_members: Optional[int] = None) -> plt.Figure:
    """
    Create wind field visualization showing tropical cyclone wind sectors.
    
    This function creates a map showing:
    - Storm tracks as faint gray lines
    - Wind sectors colored by wind speed threshold (18, 26, 33 m/s)
    
    Args:
        csv_file (str): Path to CSV file from ecmwf_tc_data_extractor
        output_filename (str, optional): Output filename for saved plot
        members (List[int], optional): Specific member numbers to plot
        n_members (int, optional): Number of members to plot (ignored if members is specified)
        
    Returns:
        matplotlib.figure.Figure: Figure object with wind field visualization
    """
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    # Check for required columns
    required_cols = ['ensemble_member', 'latitude', 'longitude', 'wind_threshold', 
                     'wind_radius', 'bearing_start', 'bearing_end']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    if members is not None:
        df = df[df['ensemble_member'].isin(members)].copy()
        if df.empty:
            raise ValueError(f"No data found for specified members: {members}")
    elif n_members is not None:
        available_members = sorted(df['ensemble_member'].unique())
        selected_members = available_members[:n_members]
        df = df[df['ensemble_member'].isin(selected_members)].copy()
        print(f"Selected members: {selected_members}")

    # Clean missing values
    df["wind_radius"] = df["wind_radius"].fillna(0.0)
    df["bearing_start"] = df["bearing_start"].fillna(0.0)
    df["bearing_end"] = df["bearing_end"].fillna(0.0)

    # Color map by threshold
    colors = {18: "tab:green", 26: "tab:orange", 33: "tab:red"}

    # Map extent from data with padding
    min_lon, max_lon = df["longitude"].min(), df["longitude"].max()
    min_lat, max_lat = df["latitude"].min(), df["latitude"].max()
    pad_lon = max(3, (max_lon - min_lon) * 0.25)
    pad_lat = max(3, (max_lat - min_lat) * 0.25)
    extent = [min_lon - pad_lon, max_lon + pad_lon, min_lat - pad_lat, max_lat + pad_lat]

    # Figure + base map
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=1)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5, zorder=1)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    # Tracks (background, faint gray)
    track_pts = df[["ensemble_member", "datetime", "latitude", "longitude"]].drop_duplicates()
    for _, g in track_pts.groupby("ensemble_member"):
        ax.plot(g["longitude"], g["latitude"], "-", color="gray", alpha=0.3,
                linewidth=0.6, markersize=0, transform=ccrs.PlateCarree())

    # Geodesic helper (distances in METERS)
    geod = pyproj.Geod(ellps="WGS84")

    def sector_polygon(lat, lon, radius_m, start_deg, end_deg, n=90):
        """Create a polygon representing a wind sector."""
        if radius_m <= 0:
            return None
        sa = start_deg % 360
        ea = end_deg % 360
        if ea < sa:
            ea += 360
        az = np.linspace(sa, ea, n, dtype=float)
        lons, lats, _ = geod.fwd(np.full_like(az, lon),
                                 np.full_like(az, lat),
                                 az,
                                 np.full_like(az, radius_m))
        # Close polygon back to center
        lons = np.concatenate(([lon], lons, [lon]))
        lats = np.concatenate(([lat], lats, [lat]))
        return Polygon(np.column_stack([lons, lats]))

    # Draw wind sectors, colored by threshold
    for _, r in df.iterrows():
        thr = int(r["wind_threshold"])
        poly = sector_polygon(r["latitude"], r["longitude"],
                              r["wind_radius"],
                              r["bearing_start"], r["bearing_end"], n=60)
        if poly is None or not poly.is_valid:
            continue
        ax.add_geometries([poly], crs=ccrs.PlateCarree(),
                          facecolor=colors.get(thr, "blue"),
                          edgecolor=colors.get(thr, "blue"),
                          alpha=0.25, linewidth=0.6, zorder=3)

    # Legend
    for thr, col in colors.items():
        ax.plot([], [], color=col, label=f"{thr} m/s")
    ax.legend(title="Wind thresholds")

    # Title
    storm_name = df["storm_id"].iloc[0] if 'storm_id' in df.columns else 'Unknown'
    plt.title(f"Tropical Cyclone {storm_name} - Wind Field Visualization")

    plt.tight_layout()

    if output_filename:
        plt.savefig(output_filename, dpi=DEFAULT_DPI, bbox_inches='tight')
        print(f"Wind field visualization saved as: {output_filename}")

    return fig