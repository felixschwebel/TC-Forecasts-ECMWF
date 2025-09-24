#!/usr/bin/env python3
"""
ECMWF Tropical Cyclone Track Data Extractor

This module provides functions to extract tropical cyclone track data from ECMWF BUFR files.
It handles the complex BUFR Template 316082 format used by ECMWF for tropical cyclone
ensemble forecasts.

The extractor processes:
- Multiple ensemble members (typically 52 members)
- Forecast time steps (0-240 hours)
- Storm center positions and maximum wind locations
- Wind radii data for different thresholds (18, 26, 33 m/s = 34, 50, 64 knots)
- Meteorological parameters (pressure, wind speed)

Data Structure:
- Storm center: Forecast center position of hurricane circulation
- Maximum wind: Location of strongest winds (often offset from center)
- Analysis position: Current observed position (time step 0)
- Wind radii: Storm size and extent for different wind speed thresholds

References:
- ECMWF BUFR Format: https://confluence.ecmwf.int/display/ECC/BUFR+examples
- eccodes: https://sites.ecmwf.int/docs/eccodes/
- Tropical Cyclone Data: https://essential.ecmwf.int
- BUFR Template 316082: https://confluence.ecmwf.int/display/ECC/bufr_read_tropical_cyclone
"""

import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import collections
from eccodes import *
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

    Args:
        filename (str): Path to the BUFR file
        verbose (bool): Whether to print detailed progress information

    Returns:
        pd.DataFrame: Raw extracted tropical cyclone data with columns:
            - Member: Ensemble member number
            - step: Forecast time step (hours from analysis)
            - latitude: Storm center latitude (degrees)
            - longitude: Storm center longitude (degrees)
            - pressure: Sea level pressure at storm center (Pa)
            - wlatitude: Latitude of maximum wind location (degrees)
            - wlongitude: Longitude of maximum wind location (degrees)
            - wind: Maximum 10m wind speed (m/s)
    """
    # Open BUFR file
    f = open(filename, 'rb')
    data = collections.defaultdict(dict)
    cnt = 0

    # Loop for the messages in the file
    while 1:
        # Get handle for message
        bufr = codes_bufr_new_from_file(f)
        if bufr is None:
            break # End of file reached

        # Instruct ecCodes to expand all the descriptors (unpack the data values)
        codes_set(bufr, 'unpack', 1)

        # Extract basic message metadata
        numObs = codes_get(bufr, "numberOfSubsets")
        year = codes_get(bufr, "year")
        month = codes_get(bufr, "month")
        day = codes_get(bufr, "day")
        hour = codes_get(bufr, "hour")
        minute = codes_get(bufr, "minute")
        stormIdentifier = codes_get(bufr, "stormIdentifier")

        if verbose:
            print('**************** MESSAGE: ', cnt + 1, '  *****************')
            print('Date and time: ', day, '.', month, '.', year, '  ', hour, ':', minute)
            print('Storm identifier: ', stormIdentifier)

        # Determine how many forecast time periods are in this message
        # Each period represents a different forecast lead time
        numberOfPeriods = 0
        while True:
            numberOfPeriods = numberOfPeriods + 1
            try:
                # Try to get timePeriod array for this period number
                codes_get_array(bufr, "#%d#timePeriod" % numberOfPeriods)
            except CodesInternalError as err:
                break # No more periods found
        # Note: numberOfPeriods includes the analysis (period=0)

        # Get ensemble member numbers - each member represents a different forecast scenario
        memberNumber = codes_get_array(bufr, "ensembleMemberNumber")
        memberNumberLen = len(memberNumber)
        print('Number of Ensemble Members: ', memberNumberLen)


        # *************************************************************************************************
        # Code table for element 008005 (METEOROLOGICAL ATTRIBUTE SIGNIFICANCE)
        # https://confluence.ecmwf.int/display/ECC/WMO%3D33+code-flag+table#WMO=33codeflagtable-CF_008005

        # Code 1: STORM CENTRE
        # significance = ec.codes_get(bufr, '#1#meteorologicalAttributeSignificance')
        # latitudeCentre = ec.codes_get(bufr, '#1#latitude')
        # longitudeCentre = ec.codes_get(bufr, '#1#longitude')


        # Code 3: LOCATION OF MAXIMUM WIND
        significance = codes_get(bufr, '#3#meteorologicalAttributeSignificance')

        if significance != 3:
            print('ERROR: unexpected #3#meteorologicalAttributeSignificance=', significance)
            raise ValueError(f"Unexpected meteorological significance code: {significance}, expected 3")

        # Get arrays of maximum wind location and speed for all ensemble members
        latitudeMaxWind0 = codes_get_array(bufr, '#3#latitude')      # Lat of max wind
        longitudeMaxWind0 = codes_get_array(bufr, '#3#longitude')    # Lon of max wind
        windMaxWind0 = codes_get_array(bufr, '#1#windSpeedAt10M')    # Max wind speed


        # Extract STORM ANALYSIS LOCATION (Codes 4 & 5)
        # This represents the analyzed storm position (initial conditions)
        significance = codes_get_array(bufr, '#2#meteorologicalAttributeSignificance')

        if not any(sig in (4, 5) for sig in significance):
            print('ERROR: unexpected #2#meteorologicalAttributeSignificance')
            raise ValueError(f"Unexpected meteorological significance codes: {significance}, expected 4 or 5")

        # Get storm center analysis data for all ensemble members
        latitudeAnalysis = codes_get_array(bufr, '#2#latitude')                    # Storm center lat
        longitudeAnalysis = codes_get_array(bufr, '#2#longitude')                  # Storm center lon
        pressureAnalysis = codes_get_array(bufr, '#1#pressureReducedToMeanSeaLevel')  # Sea level pressure



        # Store initial conditions (time step 0) for each ensemble member
        # Handle cases where analysis data may be broadcast across all members
        if len(latitudeAnalysis) == len(memberNumber) and len(latitudeMaxWind0) == len(memberNumber):
             # Each member has its own analysis values
            for k in range(len(memberNumber)):
                data[k][0] = [
                    latitudeAnalysis[k],      # Storm center latitude
                    longitudeAnalysis[k],     # Storm center longitude
                    pressureAnalysis[k],      # Storm center pressure
                    latitudeMaxWind0[k],      # Max wind latitude
                    longitudeMaxWind0[k],     # Max wind longitude
                    windMaxWind0[k]           # Max wind speed
                ]
        else:
            # Analysis position is the same for all members, but pressure/wind may vary
            for k in range(len(memberNumber)):
                data[k][0] = [
                    latitudeAnalysis[0],      # Same storm center lat for all
                    longitudeAnalysis[0],     # Same storm center lon for all
                    pressureAnalysis[k],      # Member-specific pressure
                    latitudeMaxWind0[0],      # Same max wind lat for all
                    longitudeMaxWind0[0],     # Same max wind lon for all
                    windMaxWind0[k]           # Member-specific wind speed
                ]

        # Process forecast data for each time period beyond analysis (t=0)
        timePeriod = [0 for x in range(numberOfPeriods)]  # Initialize time period array


        for i in range(1, numberOfPeriods):
            # Calculate rank indices for accessing data in BUFR structure
            # The BUFR format uses a specific indexing scheme for nested data
            rank1 = i * 2 + 2  # Index for storm center data
            rank3 = i * 2 + 3  # Index for maximum wind data

            # Extract forecast lead time (hours from analysis)
            ivalues = codes_get_array(bufr, "#%d#timePeriod" % i)

            # Handle cases where timePeriod might be an array or single value
            if len(ivalues) == 1:
                timePeriod[i] = ivalues[0]
            else:
                # Find the first non-missing value
                for j in range(len(ivalues)):
                    if ivalues[j] != CODES_MISSING_LONG:
                        timePeriod[i] = ivalues[j]
                        break

            # ===== Extract STORM CENTER location (Code 1) =====
            values = codes_get_array(bufr, "#%d#meteorologicalAttributeSignificance" % rank1)

            # Get significance code, handling array vs single value
            if len(values) == 1:
                significance = values[0]
            else:
                # Find first non-missing significance code
                for j in range(len(values)):
                    if values[j] != CODES_MISSING_LONG:
                        significance = values[j]
                        break

            # Verify we have storm center data (Code 1)
            if significance == 1:
                # Extract storm center position and pressure for this forecast step
                lat = codes_get_array(bufr, "#%d#latitude" % rank1)
                lon = codes_get_array(bufr, "#%d#longitude" % rank1)
                press = codes_get_array(bufr, "#%d#pressureReducedToMeanSeaLevel" % (i + 1))
            else:
                print('ERROR: unexpected meteorologicalAttributeSignificance=', significance)
                raise ValueError(f"Unexpected meteorological significance code: {significance}, expected 1")


            # ===== Extract MAXIMUM WIND location (Code 3) =====
            values = codes_get_array(bufr, "#%d#meteorologicalAttributeSignificance" % rank3)

            # Get significance code for wind data
            if len(values) == 1:
                significanceWind = values[0]
            else:
                # Find first non-missing significance code
                for j in range(len(values)):
                    if values[j] != CODES_MISSING_LONG:
                        significanceWind = values[j]
                        break

            # Verify we have maximum wind location data (Code 3)
            if significanceWind == 3:
                # Extract maximum wind location and speed for this forecast step
                latWind = codes_get_array(bufr, "#%d#latitude" % rank3)
                lonWind = codes_get_array(bufr, "#%d#longitude" % rank3)
                wind10m = codes_get_array(bufr, "#%d#windSpeedAt10M" % (i + 1))
            else:
                print('ERROR: unexpected meteorologicalAttributeSignificance=', significanceWind)
                raise ValueError(f"Unexpected meteorological significance code: {significanceWind}, expected 3")


            # Store forecast data for all ensemble members at this time step
            for k in range(len(memberNumber)):
                data[k][i] = [
                    lat[k],      # Storm center latitude
                    lon[k],      # Storm center longitude
                    press[k],    # Storm center pressure
                    latWind[k],  # Maximum wind latitude
                    lonWind[k],  # Maximum wind longitude
                    wind10m[k]   # Maximum wind speed
                ]


        # *************************************************************************************************
        # EXTRACT WIND RADII DATA
        # Tropical Cyclone Wind Radii product
        # https://confluence.ecmwf.int/display/FCST/New+Tropical+Cyclone+Wind+Radii+product
        # 19003 - windSpeedThreshold [m/s], 3 thresholds: 18, 26 and 33 m/s ( 34, 50 and 64 knots)
        # 5021 - bearingOrAzimuth [deg], two values to define the quadrant limits, e.g. 0 and 90 for quadrant 1
        # 19004 - effectiveRadiusWithRespectToWindSpeedsAboveThreshold [m], maximum radius at which wind speeds exceed the given threshold within the given quadrant

        windSpeedThreshold = codes_get_array(bufr, 'windSpeedThreshold')
        bearingOrAzimuth = codes_get_array(bufr, 'bearingOrAzimuth')
        windRadii = codes_get_array(bufr, 'effectiveRadiusWithRespectToWindSpeedsAboveThreshold')

        # Reshape wind data to match structure: [member][time_step][threshold][quadrant]
        n_thresholds = 3 # 18, 26 and 33 m/s ( 34, 50 and 64 knots)
        n_quadrants = 4
        wind_data = {}

        # Calculate indices for reshaping
        values_per_member_per_timestep = n_thresholds * n_quadrants

        for m in range(memberNumberLen):
            wind_data[m] = {}
            for t in range(numberOfPeriods):
                wind_data[m][t] = {}
                base_idx = (m * numberOfPeriods * values_per_member_per_timestep +
                           t * values_per_member_per_timestep)

                for thresh_idx in range(n_thresholds):
                    threshold_val = windSpeedThreshold[t * n_thresholds + thresh_idx]
                    wind_data[m][t][threshold_val] = []


                    for quad_idx in range(n_quadrants):
                        radius_idx = base_idx + thresh_idx * n_quadrants + quad_idx
                        if radius_idx < len(windRadii):
                            radius = windRadii[radius_idx]

                            # Convert CODES_MISSING_DOUBLE to NaN
                            if radius == CODES_MISSING_DOUBLE or radius == -1e+100:
                                radius = np.nan

                            # Store as (bearing_start, bearing_end, radius)
                            bearing_base = (t * n_thresholds + thresh_idx) * n_quadrants * 2 + quad_idx * 2
                            if bearing_base + 1 < len(bearingOrAzimuth):
                                bearing_pair = (bearingOrAzimuth[bearing_base],
                                              bearingOrAzimuth[bearing_base + 1])
                                wind_data[m][t][threshold_val].append((*bearing_pair, radius))



        # *************************************************************************************************
        # Convert nested dictionary to unpacked format for DataFrame creation
        unpacked_data= []

        # Flatten the data structure and filter out missing values
        for m in range(len(memberNumber)):
            if verbose:
                print("== Member  %d" % memberNumber[m])
                print("step  latitude  longitude   pressure  latitude   longitude    wind")

            # Create base datetime for this message
            base_datetime = datetime(year, month, day, hour, minute)

            for s in range(len(timePeriod)):
                # Only include data points with valid lat/lon coordinates
                if (data[m][s][0] != CODES_MISSING_DOUBLE and
                    data[m][s][1] != CODES_MISSING_DOUBLE):

                    # Calculate the datetime for this forecast step
                    forecast_datetime = base_datetime + timedelta(hours=int(timePeriod[s]))
                    datetime_str = forecast_datetime.strftime("%Y-%m-%d %H:%M:%S")


                    # Print formatted output for verification
                    if verbose:
                        print(" {0:>3d}{1}{2:>6.1f}{3}{4:>6.1f}{5}{6:>8.1f}{7}{8:>6.1f}{9}{10:>6.1f}{11}{12:>6.1f}".format(
                            timePeriod[s], '  ', data[m][s][0], '     ', data[m][s][1], '     ', data[m][s][2], '  ',
                            data[m][s][3], '     ', data[m][s][4], '     ', data[m][s][5]))


                    # Base row data
                    base_row = {
                        'storm_id': stormIdentifier,
                        'ensemble_member': memberNumber[m],
                        'step': timePeriod[s],
                        'datetime': datetime_str,
                        'latitude': data[m][s][0],
                        'longitude': data[m][s][1],
                        'pressure': data[m][s][2],
                        'wlatitude': data[m][s][3],
                        'wlongitude': data[m][s][4],
                        'wind': data[m][s][5]
                    }

                    # Unpack wind radii data for each threshold
                    for threshold in [18, 26, 33]:
                        wind_radii = wind_data[m][s].get(threshold, [])

                        # Ensure we have data for all 4 quadrants
                        for quadrant in range(4):
                            row = base_row.copy()
                            row['wind_threshold'] = threshold
                            row['quadrant'] = quadrant + 1

                            if quadrant < len(wind_radii):
                                # We have data for this quadrant
                                bearing_start, bearing_end, radius = wind_radii[quadrant]
                                row['bearing_start'] = float(bearing_start)
                                row['bearing_end'] = float(bearing_end)
                                row['wind_radius'] = float(radius) if not np.isnan(radius) else np.nan
                            else:
                                # No data for this quadrant
                                row['bearing_start'] = np.nan
                                row['bearing_end'] = np.nan
                                row['wind_radius'] = np.nan

                            unpacked_data.append(row)

        cnt += 1  # Increment message counter

        # Release the BUFR message handle to free memory
        codes_release(bufr)

    # Close the file
    f.close()

    return pd.DataFrame(unpacked_data)


def extract_tc_data_from_file(filename: str, 
                             output_dir: str = DEFAULT_OUTPUT_DIR,
                             verbose: bool = True) -> Dict[str, Union[str, int]]:
    """
    Extract tropical cyclone data from a BUFR file and save results.
    
    This is the main function for processing ECMWF tropical cyclone BUFR files.
    It handles the complete extraction pipeline from BUFR file to structured data.

    Args:
        filename (str): Path to the BUFR file (must be Template 316082)
        output_dir (str): Output directory for saved files (default: "tc_data")
        verbose (bool): Whether to print detailed progress information (default: True)
        
    Returns:
        dict: Summary dictionary with keys:
            - success (bool): Whether extraction was successful
            - csv_file (str): Path to saved CSV file (None if failed)
            - records (int): Total number of records extracted
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
