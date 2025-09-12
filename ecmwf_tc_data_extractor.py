#!/usr/bin/env python3
"""
ECMWF Tropical Cyclone Track Data Extractor

This module provides functions to extract tropical cyclone track data from ECMWF BUFR files.
It focuses purely on data extraction without complex post-processing or mapping.

The extractor handles the complex BUFR format used by ECMWF for tropical cyclone
ensemble forecasts, which contains multiple ensemble members, forecast time steps,
and wind radii data for different wind speed thresholds.

BUFR File Structure:
┌─────────────────┐
│   SECTION 0     │ ← File header - "BUFR" signature + metadata
├─────────────────┤
│   SECTION 1     │ ← Identification - ECMWF center, data category
├─────────────────┤
│   SECTION 2     │ ← Local info - ECMWF-specific metadata
├─────────────────┤
│   SECTION 3     │ ← Data description - Template 316082, subsets
├─────────────────┤
│   SECTION 4     │ ← THE DATA - All meteorological data
├─────────────────┤
│   SECTION 5     │ ← End marker - "7777"
└─────────────────┘

BUFR Template 316082 - "Tropical Cyclone Track and Wind Radii":
https://vocabulary-manager.eumetsat.int/vocabularies/BUFR/WMO/35/TABLE_D/316082
- Storm identification and metadata
- Position sequences (latitude, longitude)
- Meteorological parameters (pressure, wind speed)
- Wind radii for thresholds (34kt, 50kt, 64kt)
- Ensemble member information
- Forecast time periods (0-240 hours)

Significance Codes (WMO Standard):
- Code 1: Storm Center - forecast center position of hurricane circulation
- Code 3: Maximum Wind - location of strongest winds (often offset from center)
- Code 4: Analysis Position - current observed position (not forecast)
- Code 5: Wind Radii - points defining wind speed thresholds and storm extent

Processing Stages:
1. BUFR File Reading - Open and validate BUFR file using eccodes
2. Metadata Extraction - Extract storm ID, forecast time, ensemble info
3. Array Extraction - Extract all data arrays (lat, lon, pressure, wind, etc.)
4. Data Structuring - Create DataFrame with all extracted parameters
5. Data Export - Save structured data as CSV file

References:
- ECMWF BUFR Format: https://confluence.ecmwf.int/display/ECC/BUFR+examples
- eccodes: https://sites.ecmwf.int/docs/eccodes/
- Template 316082: https://vocabulary-manager.eumetsat.int/vocabularies/BUFR/WMO/35/TABLE_D/316082
- Tropical Cyclone Data: https://essential.ecmwf.int
"""

import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import eccodes as ec
import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration
DEFAULT_OUTPUT_DIR = "tc_data"
DEFAULT_CSV_SUFFIX = "_extracted.csv"

def extract_tc_data(filename: str, verbose: bool = True) -> pd.DataFrame:
    """
    Extract tropical cyclone track data from ECMWF BUFR file.
    
    This function processes BUFR Template 316082 files containing tropical cyclone
    ensemble forecasts. It extracts all raw data arrays from the BUFR file without
    complex post-processing or mapping.
    
    Stage 1: BUFR File Reading
    - Opens BUFR file using eccodes library
    - Validates file format and unpacks data
    - Extracts basic metadata (storm ID, forecast time, ensemble info)
    
    Stage 2: Array Extraction  
    - Extracts all data arrays from Section 4 (the main data section)
    - Arrays include: latitude, longitude, pressure, wind speed, wind thresholds,
      bearings, radii, significance codes, time periods, ensemble members
    - Handles missing values using BUFR missing value codes
    
    Stage 3: Data Structuring
    - Creates DataFrame with all extracted parameters
    - Converts units (Pa to hPa, m/s to knots)
    - Maps significance codes to position types
    - Calculates forecast datetime from base time + forecast step
    
    Args:
        filename (str): Path to the BUFR file
        verbose (bool): Whether to print detailed progress information
        
    Returns:
        pd.DataFrame: Raw extracted tropical cyclone data with columns:
            - storm_id: Storm identifier
            - ensemble_member: Ensemble member number (0-51)
            - forecast_step_hours: Forecast time step in hours
            - datetime: Calculated forecast datetime
            - significance_code: BUFR significance code (1,3,4,5)
            - position_type: Mapped position type (storm_center, max_wind, etc.)
            - latitude, longitude: Geographic coordinates
            - pressure_pa, pressure_hpa: Pressure in Pa and hPa
            - wind_speed_ms, wind_speed_knots: Wind speed in m/s and knots
            - wind_threshold_ms, wind_threshold_knots: Wind threshold for radii
            - bearing_degrees: Wind direction for radii (0°, 90°, 180°, 270°)
            - max_radius_km: Maximum radius for given wind threshold
    """
    # ========================================================================
    # STAGE 1: BUFR FILE READING
    # ========================================================================
    # Open and validate BUFR file using eccodes library
    # BUFR files use a specific binary format with 6 sections (0-5)
    # Section 0: File header with "BUFR" signature
    # Section 1: Identification and metadata (ECMWF center, data category)
    # Section 2: Local information (ECMWF-specific metadata)
    # Section 3: Data description (Template 316082, number of subsets)
    # Section 4: THE DATA (all meteorological parameters)
    # Section 5: End marker ("7777")
    
    with open(filename, 'rb') as f:
        bufr = ec.codes_bufr_new_from_file(f)

        if bufr is None:
            if verbose:
                print("Error: Cannot read BUFR file - invalid format or corrupted")
            return pd.DataFrame()

        try:
            if verbose:
                print("Stage 1: Reading BUFR file structure...")
            # Unpack the BUFR data - this processes all sections and makes data accessible
            ec.codes_set(bufr, 'unpack', 1)

            # Extract basic metadata from BUFR Section 1 (Identification)
            # This includes forecast time, storm identifier, and ensemble information
            num_subsets = ec.codes_get(bufr, 'numberOfSubsets')  # Number of ensemble members
            year = ec.codes_get(bufr, 'year')
            month = ec.codes_get(bufr, 'month')
            day = ec.codes_get(bufr, 'day')
            hour = ec.codes_get(bufr, 'hour')
            minute = ec.codes_get(bufr, 'minute')

            # Create base datetime for forecast calculations
            base_datetime = datetime(year, month, day, hour, minute)
            storm_id = ec.codes_get(bufr, 'stormIdentifier')  # Storm name (e.g., "KIKO")
            ensemble_members = ec.codes_get_array(bufr, 'ensembleMemberNumber')  # Array of member numbers (0-51)

            if verbose:
                print(f"Storm: {storm_id}, Ensemble Members: {len(ensemble_members)}")
                print(f"Forecast Base Time: {base_datetime}")
                print(f"Number of Subsets: {num_subsets}")

            # ========================================================================
            # STAGE 2: ARRAY EXTRACTION FROM SECTION 4 (THE DATA)
            # ========================================================================
            # Extract all data arrays from BUFR Section 4 - this contains all the
            # meteorological data compressed using BUFR algorithms
            # Template 316082 defines the structure and order of these arrays
            
            if verbose:
                print("Stage 2: Extracting data arrays from BUFR Section 4...")
            
            arrays = {}
            # Geographic position data
            arrays['latitude'] = ec.codes_get_array(bufr, 'latitude')  # Degrees North
            arrays['longitude'] = ec.codes_get_array(bufr, 'longitude')  # Degrees East
            
            # Significance codes (WMO standard) - defines what each position represents
            arrays['significance'] = ec.codes_get_array(bufr, 'meteorologicalAttributeSignificance')
            # 1=Storm Center, 3=Max Wind, 4=Analysis Position, 5=Wind Radii
            
            # Time information
            arrays['time_period'] = ec.codes_get_array(bufr, 'timePeriod')  # Forecast step in hours
            
            # Meteorological parameters
            arrays['pressure'] = ec.codes_get_array(bufr, 'pressureReducedToMeanSeaLevel')  # Pascals
            arrays['wind_speed'] = ec.codes_get_array(bufr, 'windSpeedAt10M')  # m/s at 10m height
            
            # Wind radii data - defines storm size and extent
            arrays['wind_threshold'] = ec.codes_get_array(bufr, 'windSpeedThreshold')  # Wind speed thresholds (m/s)
            arrays['bearing'] = ec.codes_get_array(bufr, 'bearingOrAzimuth')  # Wind direction (degrees)
            
            # Maximum radius for each wind threshold and direction
            try:
                arrays['max_radius'] = ec.codes_get_array(bufr, 'maximumRadiusForGivenWindThreshold')  # km
            except:
                arrays['max_radius'] = np.array([])  # Some files may not have radius data

            if verbose:
                print(f"Array sizes:")
                for name, arr in arrays.items():
                    if len(arr) > 0:
                        print(f"  {name}: {len(arr)}")
                        if name in ['wind_threshold', 'bearing', 'significance']:
                            unique_vals = np.unique(arr)
                            print(f"    Unique values: {unique_vals}")

            # ========================================================================
            # STAGE 3: DATA STRUCTURING
            # ========================================================================
            # Create DataFrame with all extracted data arrays
            # This is a simple extraction without complex mapping - just get all the data
            
            if verbose:
                print(f"\nStage 3: Structuring extracted data...")

            # Create records for all data points
            all_records = []
            
            # Get the length of the main coordinate arrays
            n_positions = len(arrays['latitude'])
            
            if verbose:
                print(f"Creating records for {n_positions} data points...")

            # Process each data point
            for pos_idx in range(n_positions):
                # Basic position data
                lat = arrays['latitude'][pos_idx] if arrays['latitude'][pos_idx] != ec.CODES_MISSING_DOUBLE else None
                lon = arrays['longitude'][pos_idx] if arrays['longitude'][pos_idx] != ec.CODES_MISSING_DOUBLE else None

                # Significance code
                significance = arrays['significance'][pos_idx] if pos_idx < len(arrays['significance']) and arrays['significance'][pos_idx] != ec.CODES_MISSING_DOUBLE else None

                # Time period
                time_period = int(arrays['time_period'][pos_idx]) if pos_idx < len(arrays['time_period']) and arrays['time_period'][pos_idx] != ec.CODES_MISSING_DOUBLE else 0

                # Ensemble member
                member = ensemble_members[pos_idx % len(ensemble_members)]

                # Meteorological data
                pressure = arrays['pressure'][pos_idx] if pos_idx < len(arrays['pressure']) and arrays['pressure'][pos_idx] != ec.CODES_MISSING_DOUBLE else None
                wind_speed = arrays['wind_speed'][pos_idx] if pos_idx < len(arrays['wind_speed']) and arrays['wind_speed'][pos_idx] != ec.CODES_MISSING_DOUBLE else None

                # Wind radii data (raw arrays)
                wind_threshold = arrays['wind_threshold'][pos_idx] if pos_idx < len(arrays['wind_threshold']) and arrays['wind_threshold'][pos_idx] != ec.CODES_MISSING_DOUBLE else None
                bearing = arrays['bearing'][pos_idx] if pos_idx < len(arrays['bearing']) and arrays['bearing'][pos_idx] != ec.CODES_MISSING_DOUBLE else None
                max_radius = arrays['max_radius'][pos_idx] if pos_idx < len(arrays['max_radius']) and arrays['max_radius'][pos_idx] != ec.CODES_MISSING_DOUBLE else None

                # Create record with all extracted data
                record = {
                    'storm_id': storm_id,
                    'ensemble_member': int(member),
                    'forecast_step_hours': time_period,
                    'datetime': base_datetime + timedelta(hours=time_period),
                    'significance_code': significance,
                    'position_type': map_significance_to_type(significance),
                    'latitude': lat,
                    'longitude': lon,
                    'pressure_pa': pressure,
                    'pressure_hpa': pressure / 100.0 if pressure is not None and pressure > 10000 else pressure,
                    'wind_speed_ms': wind_speed,
                    'wind_speed_knots': wind_speed * 1.94384 if wind_speed is not None else None,
                    'wind_threshold_ms': wind_threshold,
                    'wind_threshold_knots': wind_threshold * 1.94384 if wind_threshold is not None else None,
                    'bearing_degrees': bearing,
                    'max_radius_km': max_radius,
                    'array_index': pos_idx
                }

                all_records.append(record)

            # ========================================================================
            # STAGE 4: DATA VALIDATION AND SUMMARY
            # ========================================================================
            # Validate extracted data and provide summary statistics
            
            if verbose:
                print(f"\nStage 4: Validating extracted data...")
                print(f"Total records extracted: {len(all_records)}")

            # Count records with valid data
            valid_coords = sum(1 for r in all_records if r['latitude'] is not None and r['longitude'] is not None)
            valid_pressure = sum(1 for r in all_records if r['pressure_pa'] is not None)
            valid_wind = sum(1 for r in all_records if r['wind_speed_ms'] is not None)
            wind_radii_data = sum(1 for r in all_records if r['wind_threshold_ms'] is not None and r['bearing_degrees'] is not None)

            if verbose:
                print(f"Records with valid coordinates: {valid_coords}")
                print(f"Records with pressure data: {valid_pressure}")
                print(f"Records with wind speed data: {valid_wind}")
                print(f"Records with wind radii data: {wind_radii_data}")
                print(f"Data quality: {valid_coords/len(all_records)*100:.1f}% valid coordinates")

        finally:
            ec.codes_release(bufr)

    return pd.DataFrame(all_records)

def map_significance_to_type(significance: Optional[int]) -> str:
    """
    Map BUFR significance codes to position types.
    
    Significance codes are defined by WMO (World Meteorological Organization)
    and indicate what each position in the BUFR data represents:
    
    Code 1 - Storm Center:
        The forecast center position of the hurricane circulation
        Used for track forecasting and storm warnings
        Represents where the lowest pressure and circulation center is predicted
        This is the "official" storm position used in advisories
        
    Code 3 - Maximum Wind Location:
        Where the strongest winds are forecast to occur
        Often offset from the storm center (especially in asymmetric storms)
        Critical for wind hazard mapping and damage assessment
        Shows the storm's intensity structure, not just position
        
    Code 4 - Analysis Position:
        Current observed or analyzed storm position (not forecast)
        Based on satellite, radar, or aircraft observations
        Used as the starting point for forecast models
        Represents "ground truth" vs. model predictions
        
    Code 5 - Wind Radii:
        Points defining wind speed thresholds (34kt, 50kt, 64kt winds)
        Shows the storm size and extent of dangerous winds
        Used for warning zones and evacuation planning
        Indicates storm structure beyond just the center
    
    Args:
        significance (int, optional): BUFR significance code (1, 3, 4, or 5)
        
    Returns:
        str: Human-readable position type
    """
    mapping = {
        1: "storm_center",      # Forecast center position of hurricane circulation
        3: "max_wind",          # Location of strongest winds (often offset from center)
        4: "analysis_position", # Current observed position (not forecast)
        5: "wind_radii"         # Points defining wind speed thresholds and storm extent
    }
    return mapping.get(significance, f"unknown_{significance}")


def extract_tc_data_from_file(filename: str, 
                             output_dir: str = DEFAULT_OUTPUT_DIR,
                             verbose: bool = True) -> Dict[str, Union[str, int]]:
    """
    Extract tropical cyclone data from a BUFR file and save results.
    
    This is the main function for processing ECMWF tropical cyclone BUFR files.
    It handles the complete extraction pipeline from BUFR file to structured data.
    
    The function processes BUFR Template 316082 files containing:
    - Multiple ensemble members (typically 52 members)
    - Forecast time steps (0-240 hours)
    - Wind radii data for different thresholds (34kt, 50kt, 64kt)
    - Meteorological parameters (pressure, wind speed)
    - Position data (storm center, max wind, analysis positions)
    
    Processing Pipeline:
    1. Validates BUFR file format and structure
    2. Extracts all data arrays using eccodes library
    3. Maps wind radii data to geographic positions
    4. Creates comprehensive DataFrame with all parameters
    5. Saves data as CSV file
    
    Args:
        filename (str): Path to the BUFR file (must be Template 316082)
        output_dir (str): Output directory for saved files (default: "tc_data")
        verbose (bool): Whether to print detailed progress information (default: True)
        
    Returns:
        dict: Summary dictionary with keys:
            - success (bool): Whether extraction was successful
            - csv_file (str): Path to saved CSV file (None if failed)
            - records (int): Total number of records extracted
            
    Example:
        # Basic extraction
        result = extract_tc_data_from_file('tc_data/storm_bufr4.bin')
        if result['success']:
            print(f"Extracted {result['records']} records")
            print(f"CSV saved to: {result['csv_file']}")
        
        # Extract with custom output directory
        result = extract_tc_data_from_file('tc_data/storm_bufr4.bin',
                                         output_dir='processed_data')
        
        # Silent extraction for batch processing
        result = extract_tc_data_from_file('tc_data/storm_bufr4.bin', verbose=False)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print(f"Extracting tropical cyclone data from: {filename}")
    
    # Extract data
    df = extract_tc_data(filename, verbose=verbose)
    
    if df.empty:
        if verbose:
            print("Error: Failed to extract data from BUFR file")
        return {'success': False, 'csv_file': None, 'records': 0}
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(filename))[0]
    csv_file = os.path.join(output_dir, base_name + DEFAULT_CSV_SUFFIX)
    
    # Save CSV data
    df.to_csv(csv_file, index=False)
    if verbose:
        print(f"Data saved to: {csv_file}")
    
    # Summary
    if verbose:
        print("=" * 50)
        print(f"Summary:")
        print(f"   Successfully extracted: {len(df)} records")
        print(f"   CSV file: {csv_file}")
        print("=" * 50)
    
    return {
        'success': True,
        'csv_file': csv_file,
        'records': len(df)
    }
