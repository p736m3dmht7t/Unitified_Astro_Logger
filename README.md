# Unified Astro Logger

A comprehensive Python-based logging and automated image processing system for remote astronomical observatories. The Unified Astro Logger integrates with your existing astrophotography workflow to provide real-time monitoring, environmental data collection, automatic image calibration, and detailed session logging.

## Features

### Real-Time Image Processing
- **Automatic Calibration**: Monitors your LIGHT frame directory and automatically calibrates new images using matching dark and flat master frames
- **ROI Support**: Handles Region of Interest (ROI) images with pixel-accurate calibration, using either configured offsets or centered assumptions
- **Plate-Solving**: Integrates with ASTAP to add World Coordinate System (WCS) data to calibrated images
- **Saturated Pixel Handling**: Identifies and preserves saturated pixels through the calibration process

### Environmental Monitoring
- **Pegasus PowerBox**: Logs temperature, humidity, dewpoint, and optional PPB Advance dew heater status
- **OpenWeatherMap**: Fetches local weather conditions (temperature, pressure, wind, clouds)
- **Boltwood Cloud Sensor**: Reads from Boltwood sky condition files
- **Roof Status**: Monitors observatory roof state with solar failsafe logic

### Focus Tracking
- **N.I.N.A. Integration**: Automatically detects and logs autofocus results from N.I.N.A.'s detection_result.json files
- **Focus Metrics**: Records position, HFR, detected stars, standard deviation, and filter information

### Robust Logging
- **Session Events**: Timestamped CSV logs of all system events (image acquisition, calibration results, errors, etc.)
- **Focus History**: Dedicated CSV for tracking focus performance over time
- **Astronomical Dating**: Uses noon-based date calculation for consistent session organization

## Requirements

### Python and Libraries
- Python 3.7 or higher
- **Required Packages**: dotenv, requests, watchdog, astropy, tenacity, timezonefinder, numpy, astroplan
- **External Tool**: ASTAP (Astrometric Stacking Program) for plate-solving

### Optional Hardware
- Pegasus PowerBox or PowerBox Advance
- Boltwood Cloud Sensor
- Physical observatory infrastructure (for roof monitoring)

### Installation

1. **Clone or download** this repository to your desired location

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

4. **Install required packages**:
   ```bash
   pip install python-dotenv requests watchdog astropy tenacity timezonefinder numpy astroplan
   ```

5. **Download ASTAP** (if not already installed):
   - Visit: https://www.hnsky.org/astap.htm
   - Extract to a known location
   - Note the path to the `astap.exe` executable

6. **Create your `.env` configuration file** (see Configuration section below)

7. **Run the logger**:
   ```bash
   python unified_astro_logger_v5.py
   ```

## Configuration

All configuration is managed through a `.env` file in the same directory as the script. Create this file with the settings appropriate for your observatory.

### Complete Example Configuration

```ini
########################################
# REQUIRED SETTINGS - Must be configured
########################################

# Observatory Location (decimal degrees)
LATITUDE=30.123456
LONGITUDE=-97.654321

# Logging Configuration
SESSION_LOG_DIR=C:\Observatory\Logs\{astro_date}
FOCUS_LOG_FILE=C:\Observatory\Data\Focus_History.csv
PERIODIC_LOG_INTERVAL_SEC=300
FILE_WRITE_DELAY_SEC=2

# Image Monitoring
IMAGE_BASE_DIR=C:\Observatory\Images\{astro_date}

# Temperature Matching
CCD_TEMP_TOLERANCE=2.0

# API Keys
OPENWEATHERMAP_API_KEY=your_openweathermap_api_key_here
LOGGING_LEVEL=INFO

########################################
# OPTIONAL SETTINGS - Configure as needed
########################################

# Pegasus PowerBox (optional)
PEGASUS_API_URL=http://pegasus.local/api/v1/getObservingConditions
PEGASUS_DEVICEMANAGER_URL=http://pegasus.local/api/v1/getDeviceManager

# Pegasus PowerBox Advance Dew Heater (optional)
PEGASUS_PPBA_API_URL=http://pegasus.local/api/v1/getPPBAdvance

# N.I.N.A. Integration (optional)
NINA_LOG_DIR=C:\Users\YourName\Documents\NINA1\AstroImages\Autofocus

# Boltwood Cloud Sensor (optional)
BOLTWOOD_FILE_PATH=C:\Observatory\Sensors\boltwood.txt

# Master Calibration Frames
MASTER_DARK_DIR=C:\Observatory\Calibration\Masters\Darks
MASTER_FLAT_DIR=C:\Observatory\Calibration\Masters\Flats

# ASTAP Plate-Solving (optional)
ASTAP_CLI_PATH=C:\Program Files\ASTAP\astap.exe

# Shutdown Control (optional)
SHUTDOWN_FLAG_FILE=C:\Observatory\shutdown.flag

# Roof Status Monitoring (optional)
ROOF_STATUS_SOURCE_PATH=\\NetworkPath\roof_status.txt
ROOF_STATUS_DEST_PATH=C:\Observatory\roof_status_local.txt

########################################
# ROI OFFSET CONFIGURATION (optional)
# Format: CAMERA_{CAMERA_NAME}_ROI_{WIDTH}x{HEIGHT}_{X|Y}_OFFSET
# Example for ZWO ASI183MM Pro with 1830x1216 ROI
########################################
CAMERA_ZWO_ASI183MM_PRO_ROI_1830x1216_X_OFFSET=125
CAMERA_ZWO_ASI183MM_PRO_ROI_1830x1216_Y_OFFSET=420

# Additional ROI configurations for same camera
CAMERA_ZWO_ASI183MM_PRO_ROI_1830x1050_X_OFFSET=125
CAMERA_ZWO_ASI183MM_PRO_ROI_1830x1050_Y_OFFSET=495

# ROI for different camera
CAMERA_COOLED_ASI120_RROI_320x240_X_OFFSET=150
CAMERA_COOLED_ASI120_RROI_320x240_Y_OFFSET=100
```

### Configuration Guide

#### Required Settings

**LATITUDE / LONGITUDE**: Your observatory's geographic coordinates in decimal degrees. Used for astronomical date calculation, solar positioning, and weather API calls.

**SESSION_LOG_DIR**: Directory where session logs will be stored. The `{astro_date}` placeholder is automatically replaced with the current astronomical date.

**FOCUS_LOG_FILE**: Path to the CSV file for focus history tracking.

**PERIODIC_LOG_INTERVAL_SEC**: How often (in seconds) to log environmental conditions. Default: 300 (5 minutes).

**FILE_WRITE_DELAY_SEC**: Delay after detecting a new file before processing begins, ensuring writes are complete. Default: 2 seconds.

**IMAGE_BASE_DIR**: Root directory to monitor for new LIGHT frames. Must contain a subdirectory named "LIGHT" (case-insensitive). The `{astro_date}` placeholder is automatically replaced.

**CCD_TEMP_TOLERANCE**: Maximum temperature difference (in Celsius) allowed when matching dark frames. Images and masters must match within this tolerance.

**OPENWEATHERMAP_API_KEY**: Your API key from OpenWeatherMap (free tier available at https://openweathermap.org).

**LOGGING_LEVEL**: Logging verbosity. Options: DEBUG, INFO, WARNING, ERROR. Default: INFO.

#### Optional Settings

**Pegasus URLs**: Configure your Pegasus PowerBox device manager URL and observing conditions endpoint. The logger automatically fetches the unique device key.

**NINA_LOG_DIR**: Directory where N.I.N.A. writes autofocus detection results. Monitors for `*final*detection_result.json` files.

**BOLTWOOD_FILE_PATH**: Path to the last line of Boltwood sensor output (typically boltwood.txt).

**MASTER_DARK_DIR / MASTER_FLAT_DIR**: Directories containing master calibration frames. The script recursively searches these directories for matching masters.

**ASTAP_CLI_PATH**: Full path to ASTAP executable for plate-solving calibrated images.

**SHUTDOWN_FLAG_FILE**: Path to a file that, when created, triggers graceful shutdown. Useful for remote control.

**ROOF_STATUS**: Configure both source (network path) and destination (local path) for roof status files. The logger copies from source to destination every periodic interval.

**ROI OFFSETS**: Configured ROI offsets take precedence over centered assumptions. Format is critical:
- Camera name should match your FITS header CAMERAID (spaces, dashes, and underscores are normalized)
- ROI dimensions use format `WIDTHxHEIGHT` (e.g., `1830x1216`)
- Both X_OFFSET and Y_OFFSET must be defined for each ROI size

#### Astronomical Date Logic

The logger uses astronomical dating, where the date changes at **noon local time** (not midnight). This ensures sessions that span midnight are grouped correctly. The system:
- Determines your local timezone from LATITUDE/LONGITUDE
- Calculates the most recent noon in local time
- Uses that date for directory organization and log filenames
- Properly handles DST transitions

## Usage

### Starting the Logger

Once your `.env` file is configured, simply run:

```bash
python unified_astro_logger_v5.py
```

The script will:
1. Validate your configuration
2. Determine the current astronomical date
3. Create necessary directories and log files
4. Start monitoring threads
5. Begin processing images as they appear

### Shutting Down

Three methods are available:

1. **Interactive**: Press `q` followed by Enter
2. **Keyboard Interrupt**: Press Ctrl+C
3. **Flag File**: Create the file specified in `SHUTDOWN_FLAG_FILE` (it will be automatically deleted)

All methods trigger a graceful shutdown that completes in-progress operations and writes a SESSION_END event.

### Monitoring and Logs

The console output shows real-time processing status. You can adjust verbosity using the `LOGGING_LEVEL` setting in your `.env` file.

## Output Files

### Session Log

The session log (`{astro_date}_Unified_Session_Log.csv`) records all system events:

| Column | Description |
|--------|-------------|
| TimestampUTC | ISO 8601 UTC timestamp |
| EventType | Event category (IMAGE_ACQUIRED, CALIBRATION_SUCCESS, PERIODIC_STATUS, etc.) |
| Status | Event status (SUCCESS, INFO, WARNING, ERROR) |
| Message | Human-readable event description |
| DetailsJSON | JSON object with event-specific details |

**Example Event Types**: SESSION_START, IMAGE_ACQUIRED, CALIBRATION_START, CALIBRATION_SUCCESS, CALIBRATION_FAIL, PLATESOLVE_START, PLATESOLVE_SUCCESS, PLATESOLVE_FAIL, AUTOFOCUS_COMPLETE, PERIODIC_STATUS, ROOF_STATUS_UPDATE, SESSION_END

### Focus Log

The focus log (specified by `FOCUS_LOG_FILE`) tracks autofocus performance:

| Column | Description |
|--------|-------------|
| TimestampUTC | ISO 8601 UTC timestamp |
| FocuserPosition | Focuser position in steps |
| AverageHFR | Average Half-Flux Radius (pixels) |
| DetectedStars | Number of stars detected |
| HFRStdDev | Standard deviation of HFR measurements |
| Filter | Filter name used |
| AmbientTempC | Ambient temperature (°C) from Pegasus |
| FocuserTempC | Focuser temperature (°C) when available |
| SourceFile | Path to the N.I.N.A. detection result file |

### Calibrated Images

Calibrated images are saved with the suffix `_calibrated` in the same directory as the original LIGHT frame:
- `M42_300s_R.fits` → `M42_300s_R_calibrated.fits`

The FITS header is updated with:
- `CALSTAT`: Set to 'BDF' (Bias, Dark, Flat calibrated)
- `HISTORY`: Calibration details including master frame names and ROI offsets
- `OBSERVAT`: Observatory code and name

If plate-solving succeeds, WCS keywords are added directly to the header.

## Advanced Features

### ROI Calibration

For Region of Interest images where the light frame dimensions differ from master frames, the logger provides two offset strategies:

**Configured Offsets** (recommended): Measure the ROI position relative to full-frame masters and configure X/Y offsets in `.env`. The logger uses flexible camera name matching (spaces, dashes, underscores are normalized).

**Centered Fallback**: If no configured offsets are found, the logger assumes the ROI is centered and calculates offsets accordingly.

Both methods result in pixel-accurate calibration when masters and lights share the same camera ID.

### Saturated Pixel Preservation

Pixels with values ≥65504 in the raw light frame are flagged as saturated. After calibration, these pixels are set to 65535 (maximum possible value) regardless of calibration mathematics. This preserves true saturation information in the calibrated output.

### Roof Status Failsafe

The roof monitoring system implements a solar-based failsafe:
- If the source roof status file becomes unavailable (network error, etc.), the logger checks the sun's altitude
- If the sun is more than 5 degrees above the horizon, it automatically writes a "CLOSED" status to the destination
- This prevents accidentally leaving the roof open during daylight hours
- During nighttime, the last known good status is preserved

### Calibration Frame Matching

Masters are matched using strict criteria:

**Dark Frames**: Must match within tolerance for exposure time, gain, offset, camera ID, and CCD temperature.

**Flat Frames**: Must match on filter name (case-insensitive) and camera ID.

Within temperature tolerance, the coldest matching dark is selected for optimal signal quality.

## Troubleshooting

### The logger won't start

**Check configuration errors**: Missing required settings will cause immediate exit with a clear error message. Review the console output and verify all required `.env` variables are set.

**Verify Python version**: Use `python --version` to confirm Python 3.7 or higher.

### Images aren't being processed

**Verify IMAGE_BASE_DIR**: Ensure the path is correct and contains a subdirectory named "LIGHT". The logger waits up to 15 seconds for directories to be created if they don't exist.

**Check LIGHT subdirectory**: Only files in a directory containing "LIGHT" (case-insensitive) are processed.

**Review log level**: Set `LOGGING_LEVEL=DEBUG` to see detailed file monitoring activity.

### Calibration fails with "Could not find suitable master frames"

**Missing masters**: Ensure `MASTER_DARK_DIR` and/or `MASTER_FLAT_DIR` are configured and accessible.

**Mismatched parameters**: Verify that your master frames were created with compatible exposure times, gain/offset settings, and filters. Check FITS headers to confirm.

**Temperature mismatch**: If your light frame CCD temperature differs from all available masters by more than `CCD_TEMP_TOLERANCE`, no dark will be selected. Consider increasing tolerance or generating masters at additional temperatures.

**Camera ID mismatch**: Ensure `CAMERAID` in your master frames matches the ID in light frames.

### ROI calibration issues

**Dimension errors**: If the ROI light frame dimensions exceed the master frame dimensions, calibration cannot proceed. Verify your masters are full-frame or larger than your ROI images.

**Unexpected offsets**: Review the log output to see whether configured offsets or centered assumptions were used. Check your `.env` ROI configuration format carefully.

**Camera name matching**: The logger logs which cameras have configured ROI offsets at startup. Verify your FITS header `CAMERAID` matches your `.env` camera name after normalization.

### API/Network issues

**Pegasus connection**: The logger tolerates temporary API outages and logs warnings only once until connection is restored. If permanently unavailable, simply omit PEGASUS URLs from configuration.

**OpenWeatherMap**: Check that your API key is valid and you haven't exceeded rate limits. The free tier provides sufficient quota for typical observatory use.

**Roof status file**: If the source path is a network share, ensure it's accessible. The failsafe will activate if the network is down during daylight hours.

### Excessive log file size

**Reduce periodic interval**: Increase `PERIODIC_LOG_INTERVAL_SEC` if environmental logging frequency is too high.

**Adjust logging level**: Set `LOGGING_LEVEL=WARNING` or `ERROR` to reduce verbosity in normal operation. Use DEBUG only for troubleshooting.

### Plate-solving fails

**Verify ASTAP path**: Ensure `ASTAP_CLI_PATH` points to the correct executable.

**ASTAP command flags**: The script uses `-r 1 -fov 1.44 -z 2 -sip -wcs -update`. These settings work for most telescopes, but you may need to adjust based on your setup.

**Image suitability**: ASTAP requires sufficient stars for solving. Very poor seeing, thin clouds, or wrong exposure settings can prevent successful solving.

## Version History

- **v18**: Enhanced ROI calibration to use camera-specific measured offsets from .env configuration. Falls back to centered ROI assumption if offsets not configured.
- **v17**: Added support for calibrating Region of Interest (ROI) images where NAXIS1/NAXIS2 differ from master frames.
- **v16**: Added robust roof status monitoring, Pegasus PPB Advance dew heater logging, external shutdown via flag file.

## License

MIT License

Copyright (c) 2025 John D. Phillips (john.d.phillips@comcast.net)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Support

For issues, questions, or contributions, please refer to the [project repository](https://github.com/p736m3dmht7t/Unified_Astro_Logger) or contact the development team.

