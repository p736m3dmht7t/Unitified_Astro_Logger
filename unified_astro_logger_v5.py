# unified_astro_logger_v5.py (v18)
#
# Changelog:
# - v18: CRITICAL FIX - Corrected ROI pixel alignment for ASI183MM Pro camera. Uses measured actual ROI positions 
#        instead of assuming perfect mathematical centering. For 50% ROI: offset is (1372,918) not (1376,918).
#        For 25% ROI: offset is (2060,1376) not (2064,1377). This ensures calibration frames are extracted from
#        the correct pixel locations, preventing misaligned hot pixel and flat field correction.
# - v17: Added support for calibrating Region of Interest (ROI) images where NAXIS1/NAXIS2 differ from master frames. 
#        Matches on CAMERAID, assumes centered ROI, crops master dark/flat to match light frame dimensions for pixel-accurate calibration.
# - v16: Added robust roof status monitoring, Pegasus PPB Advance dew heater logging, external shutdown via flag file, and more (see previous changelog).
# - Full-featured, robust logger with API integration and real-time image processing.

import os
import sys
import time
import datetime
import logging
import subprocess
import threading
import csv
import json
import pytz
from pathlib import Path

# Third-Party Libraries
from dotenv import load_dotenv
import requests
import win32com.client
import pythoncom
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from astropy.io import fits
from tenacity import retry, stop_after_attempt, wait_fixed
from timezonefinder import TimezoneFinder
import numpy as np
from astroplan import Observer as AstroplanObserver
from astropy.coordinates import EarthLocation
from astropy.time import Time
import astropy.units as u

load_dotenv()
LOG_LEVELS = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO, 'WARNING': logging.WARNING, 'ERROR': logging.ERROR}
log_level = LOG_LEVELS.get(os.getenv('LOGGING_LEVEL', 'INFO').upper(), logging.INFO)
logging.basicConfig(level=log_level, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

# Camera-specific ROI offset lookup table
# Based on empirical measurements using dark current pattern correlation
CAMERA_ROI_OFFSETS = {
    'ZWO ASI183MM Pro': {
        'full_frame': (5496, 3672),
        'roi_offsets': {
            # Format: (width, height): (x_offset, y_offset)
            # These are the actual measured top-left corner positions in the full frame
            (2744, 1836): (1372, 918),  # 50% ROI - measured via correlation
            (1368, 918): (2060, 1376),   # 25% ROI - measured via correlation
        }
    },
    # Add other camera models here as needed
    # 'ZWO ASI294MC Pro': {
    #     'full_frame': (4144, 2822),
    #     'roi_offsets': {
    #         (2072, 1411): (1036, 705),  # Example 50% ROI - needs measurement
    #     }
    # },
}

def get_roi_offset(camera_model, roi_width, roi_height, full_width, full_height):
    """
    Get the actual ROI offset for a specific camera model and ROI size.
    
    Parameters:
    -----------
    camera_model : str
        Camera model identifier (e.g., 'ZWO ASI183MM Pro')
    roi_width, roi_height : int
        ROI dimensions
    full_width, full_height : int
        Full frame dimensions
    
    Returns:
    --------
    tuple : (x_offset, y_offset)
        Actual top-left corner position of ROI in full frame
    """
    # Try to find camera-specific offsets
    if camera_model in CAMERA_ROI_OFFSETS:
        camera_data = CAMERA_ROI_OFFSETS[camera_model]
        roi_key = (roi_width, roi_height)
        
        if roi_key in camera_data['roi_offsets']:
            x_offset, y_offset = camera_data['roi_offsets'][roi_key]
            logging.info(f"Using measured ROI offset for {camera_model} {roi_width}x{roi_height}: ({x_offset}, {y_offset})")
            return x_offset, y_offset
        else:
            logging.warning(f"No measured offset for {camera_model} ROI {roi_width}x{roi_height}. Falling back to mathematical centering.")
    else:
        logging.warning(f"Camera model '{camera_model}' not in ROI offset table. Using mathematical centering.")
    
    # Fallback to mathematical centering
    x_offset = (full_width - roi_width) // 2
    y_offset = (full_height - roi_height) // 2
    logging.info(f"Using calculated (fallback) ROI offset: ({x_offset}, {y_offset})")
    return x_offset, y_offset

class UnifiedAstroLogger:
    def __init__(self):
        self.pegasus_api_is_down = False
        self.shutdown_event = threading.Event()
        self.session_csv_lock = threading.Lock()
        self.focus_csv_lock = threading.Lock()
        self._load_config()

        # Initialize location-based objects once for efficiency
        self.tf = TimezoneFinder()
        try:
            self.observer_location = EarthLocation.from_geodetic(
                lon=self.config["LONGITUDE"] * u.deg,
                lat=self.config["LATITUDE"] * u.deg
            )
            self.astro_observer = AstroplanObserver(location=self.observer_location)
        except Exception as e:
            raise ValueError(f"Could not initialize Astroplan Observer. Check LATITUDE/LONGITUDE. Error: {e}")

        self.astro_date_str = self._get_astronomical_date_str()
        self._resolve_dynamic_paths()
        self._initialize_csv(self.config["SESSION_LOG_FILE"], ["TimestampUTC", "EventType", "Status", "Message", "DetailsJSON"], self.session_csv_lock)
        self._initialize_csv(self.config["FOCUS_LOG_FILE"], ["TimestampUTC", "FocuserPosition", "AverageHFR", "DetectedStars", "HFRStdDev", "Filter", "AmbientTempC", "FocuserTempC", "SourceFile"], self.focus_csv_lock)
        logging.info(f"Unified Astro Logger v5 initialized for session date: {self.astro_date_str}")
        clean_config = {k: str(v) for k, v in self.config.items() if "API_KEY" not in k}
        self.log_session_event("SESSION_START", "SUCCESS", "Logger started (v5 with ROI alignment fix).", {"config": clean_config})

    def _load_config(self):
        """Loads all configuration from environment variables with validation."""
        self.config = {}
        required_strings = ["SESSION_LOG_DIR", "FOCUS_LOG_FILE", "IMAGE_BASE_DIR", "OPENWEATHERMAP_API_KEY"]
        optional_strings = ["NINA_LOG_DIR", "BOLTWOOD_FILE_PATH", "MASTER_DARK_DIR", "MASTER_FLAT_DIR",
                            "ASTAP_CLI_PATH", "PEGASUS_API_URL", "PEGASUS_DEVICEMANAGER_URL",
                            "PEGASUS_PPBA_API_URL", "SHUTDOWN_FLAG_FILE", "ROOF_STATUS_SOURCE_PATH",
                            "ROOF_STATUS_DEST_PATH"]
        required_floats = ["LATITUDE", "LONGITUDE", "CCD_TEMP_TOLERANCE"]
        required_ints = ["PERIODIC_LOG_INTERVAL_SEC", "FILE_WRITE_DELAY_SEC"]

        for key in required_strings:
            value = os.getenv(key)
            if not value: raise ValueError(f"CRITICAL ERROR: Required setting '{key}' is missing from your .env file.")
            self.config[key] = value
        for key in optional_strings:
            self.config[key] = os.getenv(key)

        for key in required_floats:
            value = os.getenv(key)
            if value is None: raise ValueError(f"CRITICAL ERROR: Required setting '{key}' is missing from your .env file.")
            self.config[key] = float(value)

        for key in required_ints:
            value = os.getenv(key)
            if value is None: raise ValueError(f"CRITICAL ERROR: Required setting '{key}' is missing from your .env file.")
            self.config[key] = int(value)
            
    def _get_astronomical_date_str(self):
        tz_str = self.tf.timezone_at(lat=self.config["LATITUDE"], lng=self.config["LONGITUDE"])
        now_local = datetime.datetime.now(pytz.utc).astimezone(pytz.timezone(tz_str))
        if now_local.hour < 12:
            return (now_local - datetime.timedelta(days=1)).date().strftime('%Y-%m-%d')
        else:
            return now_local.date().strftime('%Y-%m-%d')

    def _resolve_dynamic_paths(self):
        logging.info("Resolving dynamic paths with astro_date: %s", self.astro_date_str)
        for key in ["SESSION_LOG_DIR", "IMAGE_BASE_DIR"]:
            if self.config[key] and "{astro_date}" in self.config[key]:
                self.config[key] = self.config[key].format(astro_date=self.astro_date_str)
        
        self.config["SESSION_LOG_FILE"] = Path(self.config["SESSION_LOG_DIR"]) / f"{self.astro_date_str}_Unified_Session_Log.csv"
        self.config["SESSION_LOG_FILE"].parent.mkdir(parents=True, exist_ok=True)
        
        for key in ["FOCUS_LOG_FILE", "IMAGE_BASE_DIR", "NINA_LOG_DIR", "BOLTWOOD_FILE_PATH", 
                    "MASTER_DARK_DIR", "MASTER_FLAT_DIR", "SHUTDOWN_FLAG_FILE", 
                    "ROOF_STATUS_SOURCE_PATH", "ROOF_STATUS_DEST_PATH"]:
            if self.config.get(key):
                self.config[key] = Path(self.config[key])

    def _initialize_csv(self, file_path, headers, lock):
        with lock:
            if not file_path.exists() or file_path.stat().st_size == 0:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                logging.info(f"Created new log file with headers: {file_path}")

    def _write_to_csv(self, file_path, row, lock):
        with lock:
            try:
                with open(file_path, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow(row)
            except Exception as e:
                logging.error(f"FATAL: Could not write to CSV file {file_path}! {e}")

    def log_session_event(self, event_type, status, message, details_dict=None):
        details_json = json.dumps(details_dict, default=str) if details_dict else "{}"
        row = [datetime.datetime.now(pytz.utc).isoformat(), event_type, status, message, details_json]
        self._write_to_csv(self.config["SESSION_LOG_FILE"], row, self.session_csv_lock)

    def log_focus_event(self, focus_data):
        row = [
            datetime.datetime.now(pytz.utc).isoformat(),
            focus_data.get("focuser_position"),
            focus_data.get("average_hfr"),
            focus_data.get("detected_stars"),
            focus_data.get("hfr_std_dev"),
            focus_data.get("filter"),
            focus_data.get("ambient_temp_c"),
            focus_data.get("focuser_temp_c"),
            focus_data.get("source_file")
        ]
        self._write_to_csv(self.config["FOCUS_LOG_FILE"], row, self.focus_csv_lock)

    def start(self):
        """Main event loop that monitors logs, periodically logs, processes images."""
        threads = []
        nina_log_dir = self.config.get("NINA_LOG_DIR")
        if nina_log_dir and nina_log_dir.exists():
            observer = Observer()
            observer.schedule(NinaLogHandler(self), str(nina_log_dir), recursive=True)
            observer.start()
            threads.append(observer)
            logging.info(f"Watching NINA log directory: {nina_log_dir}")

        image_base_dir = self.config.get("IMAGE_BASE_DIR")
        if image_base_dir and image_base_dir.exists():
            observer = Observer()
            observer.schedule(ImageEventHandler(self), str(image_base_dir), recursive=True)
            observer.start()
            threads.append(observer)
            logging.info(f"Watching Image directory: {image_base_dir}")

        t_periodic = threading.Thread(target=self._periodic_logging_thread, daemon=True)
        t_periodic.start()
        threads.append(t_periodic)

        roof_source = self.config.get("ROOF_STATUS_SOURCE_PATH")
        roof_dest = self.config.get("ROOF_STATUS_DEST_PATH")
        if roof_source and roof_dest:
            t_roof = threading.Thread(target=self._roof_status_monitoring_thread, daemon=True, args=(roof_source, roof_dest))
            t_roof.start()
            threads.append(t_roof)

        shutdown_flag = self.config.get("SHUTDOWN_FLAG_FILE")
        if shutdown_flag:
            t_shutdown_check = threading.Thread(target=self._shutdown_check_thread, daemon=True, args=(shutdown_flag,))
            t_shutdown_check.start()
            threads.append(t_shutdown_check)

        try:
            logging.info("Main thread entering infinite loop. Press CTRL+C to stop.")
            while not self.shutdown_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            logging.warning("KeyboardInterrupt received. Shutting down gracefully...")
        finally:
            self.log_session_event("SESSION_END", "SUCCESS", "Logger stopped normally.", {})
            self.shutdown_event.set()
            for t in threads:
                if isinstance(t, Observer):
                    t.stop()
                    t.join(timeout=2)
            logging.info("UnifiedAstroLogger has shut down.")

    def _periodic_logging_thread(self):
        """Periodically logs weather, mount, and hardware data."""
        interval = self.config["PERIODIC_LOG_INTERVAL_SEC"]
        while not self.shutdown_event.is_set():
            try:
                weather_data = self._get_weather_data()
                boltwood_data = self._get_boltwood_data()
                mount_data = self._get_mount_data()
                pegasus_data = self._get_pegasus_data()
                camera_data = self._get_camera_data()

                details = {
                    "weather": weather_data,
                    "boltwood": boltwood_data,
                    "mount": mount_data,
                    "pegasus": pegasus_data,
                    "camera": camera_data
                }
                self.log_session_event("PERIODIC_LOG", "INFO", "Periodic data snapshot.", details)
            except Exception as e:
                logging.error(f"Error in periodic logging thread: {e}", exc_info=True)
            time.sleep(interval)

    def _shutdown_check_thread(self, flag_file):
        """Monitors a flag file for external shutdown requests."""
        while not self.shutdown_event.is_set():
            if flag_file.exists():
                logging.warning(f"Shutdown flag detected: {flag_file}. Initiating graceful shutdown...")
                self.log_session_event("EXTERNAL_SHUTDOWN", "INFO", "Shutdown flag file detected.", {"file": str(flag_file)})
                self.shutdown_event.set()
                return
            time.sleep(5)

    def _roof_status_monitoring_thread(self, source_path, dest_path):
        """Monitors roof status by reading from source and copying to destination."""
        while not self.shutdown_event.is_set():
            try:
                if source_path.exists():
                    import shutil
                    shutil.copy2(source_path, dest_path)
                    with open(source_path, 'r') as f:
                        status = f.read().strip()
                    self.log_session_event("ROOF_STATUS", "INFO", f"Roof status: {status}", {"source": str(source_path)})
                else:
                    logging.debug(f"Roof status source file not found: {source_path}")
            except Exception as e:
                logging.error(f"Error reading roof status: {e}", exc_info=True)
            time.sleep(30)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _get_weather_data(self):
        """Fetches weather data from OpenWeatherMap."""
        api_key = self.config["OPENWEATHERMAP_API_KEY"]
        lat, lon = self.config["LATITUDE"], self.config["LONGITUDE"]
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        return {
            "temp_c": data["main"]["temp"],
            "humidity_pct": data["main"]["humidity"],
            "pressure_hPa": data["main"]["pressure"],
            "wind_speed_ms": data["wind"]["speed"],
            "clouds_pct": data["clouds"]["all"],
            "description": data["weather"][0]["description"]
        }

    def _get_boltwood_data(self):
        """Reads Boltwood data from a file."""
        boltwood_path = self.config.get("BOLTWOOD_FILE_PATH")
        if not boltwood_path or not boltwood_path.exists():
            return {}
        try:
            with open(boltwood_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            if len(lines) < 2:
                return {}
            parts = lines[1].split()
            return {
                "sky_temp_c": float(parts[4]),
                "ambient_temp_c": float(parts[5]),
                "wind_speed_kmh": float(parts[7]),
                "humidity_pct": int(parts[8]),
                "dewpoint_c": float(parts[9]),
                "cloud_condition": parts[14]
            }
        except Exception as e:
            logging.error(f"Error parsing Boltwood file: {e}", exc_info=True)
            return {}

    def _get_mount_data(self):
        """Retrieves mount information via ASCOM."""
        try:
            pythoncom.CoInitialize()
            try:
                telescope = win32com.client.Dispatch("ASCOM.Software Bisque.Telescope")
                telescope.Connected = True
                return {
                    "ra_hours": telescope.RightAscension,
                    "dec_deg": telescope.Declination,
                    "azimuth_deg": telescope.Azimuth,
                    "altitude_deg": telescope.Altitude,
                    "tracking": telescope.Tracking
                }
            except Exception as e:
                logging.error(f"Error connecting to mount: {e}", exc_info=True)
                return {}
        finally:
            pythoncom.CoUninitialize()

    def _get_pegasus_data(self):
        """Retrieves data from Pegasus Ultimate Powerbox v2 and PPB Advance."""
        data = {}
        
        # Fetch from Pegasus API (Ultimate Powerbox v2)
        pegasus_url = self.config.get("PEGASUS_API_URL")
        if pegasus_url:
            try:
                r = requests.get(pegasus_url, timeout=5)
                r.raise_for_status()
                peg_data = r.json()
                data.update({
                    "pegasus_temp_c": peg_data.get("temperature"),
                    "pegasus_humidity_pct": peg_data.get("humidity"),
                    "pegasus_dewpoint_c": peg_data.get("dewpoint")
                })
                self.pegasus_api_is_down = False
            except Exception as e:
                if not self.pegasus_api_is_down:
                    logging.error(f"Pegasus API error: {e}", exc_info=True)
                    self.pegasus_api_is_down = True

        # Fetch from PPBA API (PPB Advance dew heaters)
        ppba_url = self.config.get("PEGASUS_PPBA_API_URL")
        if ppba_url:
            try:
                r = requests.get(ppba_url, timeout=5)
                r.raise_for_status()
                ppba_data = r.json()
                data.update({
                    "ppba_dew1_power_pct": ppba_data.get("dew1_power"),
                    "ppba_dew2_power_pct": ppba_data.get("dew2_power"),
                    "ppba_dew3_power_pct": ppba_data.get("dew3_power")
                })
            except Exception as e:
                logging.debug(f"PPBA API error (non-critical): {e}")

        # Fetch from Device Manager (focuser temp)
        dm_url = self.config.get("PEGASUS_DEVICEMANAGER_URL")
        if dm_url:
            try:
                r = requests.get(dm_url, timeout=5)
                r.raise_for_status()
                dm_data = r.json()
                data["focuser_temp_c"] = dm_data.get("focuser_temperature")
            except Exception as e:
                logging.debug(f"Device Manager API error (non-critical): {e}")

        return data

    def _get_camera_data(self):
        """Retrieves camera information via ASCOM."""
        try:
            pythoncom.CoInitialize()
            try:
                camera = win32com.client.Dispatch("ASCOM.ASICamera2.Camera")
                camera.Connected = True
                return {
                    "ccd_temp_c": camera.CCDTemperature,
                    "cooler_power_pct": camera.CoolerPower,
                    "cooler_on": camera.CoolerOn
                }
            except Exception as e:
                logging.error(f"Error connecting to camera: {e}", exc_info=True)
                return {}
        finally:
            pythoncom.CoUninitialize()

class ImageEventHandler(FileSystemEventHandler):
    """Handles new FITS image files by calibrating and plate-solving them."""
    def __init__(self, logger):
        self.logger = logger
        self.config = logger.config
        self.processing = set()
        self.lock = threading.Lock()

    def on_created(self, event):
        if event.is_directory or self.logger.shutdown_event.is_set():
            return
        if not event.src_path.lower().endswith(".fits"):
            return
        path = Path(event.src_path)
        
        with self.lock:
            if path in self.processing:
                return
            self.processing.add(path)

        threading.Thread(target=self._process_image, args=(path,), daemon=True).start()

    def _process_image(self, image_path):
        """Process a single FITS image: calibrate and plate-solve."""
        try:
            time.sleep(self.config["FILE_WRITE_DELAY_SEC"])
            
            # Skip already calibrated images
            if "_calibrated" in image_path.stem.lower():
                logging.debug(f"Skipping already calibrated image: {image_path}")
                return

            # Check for LIGHT frame
            try:
                with fits.open(image_path) as hdul:
                    imagetyp = hdul[0].header.get('IMAGETYP', '').strip().upper()
            except Exception as e:
                logging.error(f"Could not read IMAGETYP from {image_path}: {e}")
                return

            if imagetyp != 'LIGHT':
                logging.debug(f"Skipping non-LIGHT frame: {image_path} (IMAGETYP={imagetyp})")
                return

            # Calibrate
            calibrated_path = self.calibrate_image(image_path)
            if calibrated_path:
                # Plate-solve
                self.plate_solve_image(calibrated_path)
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}", exc_info=True)
        finally:
            with self.lock:
                self.processing.discard(image_path)

    def calibrate_image(self, image_path):
        """
        Calibrate a LIGHT frame using master dark and flat.
        Now with CORRECTED ROI pixel alignment for ASI183MM Pro.
        """
        try:
            master_dark_dir = self.config.get("MASTER_DARK_DIR")
            master_flat_dir = self.config.get("MASTER_FLAT_DIR")
            
            if not master_dark_dir or not master_flat_dir:
                logging.warning("Master dark/flat directories not configured, skipping calibration.")
                return None

            with fits.open(image_path) as image_hdul:
                image_header = image_hdul[0].header.copy()
                image_data = image_hdul[0].data.astype(np.float64)
                original_dtype = image_hdul[0].data.dtype
                
                # Extract matching criteria
                image_exposure = image_header.get('EXPOSURE')
                image_ccd_temp = image_header.get('CCD-TEMP')
                image_gain = image_header.get('GAIN')
                image_offset = image_header.get('OFFSET')
                image_xbinning = image_header.get('XBINNING', 1)
                image_ybinning = image_header.get('YBINNING', 1)
                image_filter = image_header.get('FILTER', 'Luminance')
                image_cameraid = image_header.get('CAMERAID', 'Unknown')
                image_naxis1 = image_header.get('NAXIS1')
                image_naxis2 = image_header.get('NAXIS2')

                if not all([image_exposure, image_ccd_temp, image_gain, image_offset, image_naxis1, image_naxis2]):
                    self.logger.log_session_event("CALIBRATION_SKIP", "WARNING", "Missing critical header info.", {"file": str(image_path)})
                    return None

            # Find matching master dark
            dark_path = self._find_master_frame(master_dark_dir, 'dark', image_exposure, image_ccd_temp, 
                                               image_gain, image_offset, image_xbinning, image_ybinning, 
                                               None, image_cameraid)
            if not dark_path:
                self.logger.log_session_event("CALIBRATION_FAIL", "ERROR", "No matching master dark found.", {
                    "file": str(image_path),
                    "exposure": image_exposure,
                    "ccd_temp": image_ccd_temp,
                    "gain": image_gain,
                    "offset": image_offset,
                    "cameraid": image_cameraid
                })
                return None

            # Find matching master flat
            flat_path = self._find_master_frame(master_flat_dir, 'flat', None, None, image_gain, image_offset, 
                                               image_xbinning, image_ybinning, image_filter, image_cameraid)
            if not flat_path:
                self.logger.log_session_event("CALIBRATION_FAIL", "ERROR", "No matching master flat found.", {
                    "file": str(image_path),
                    "filter": image_filter,
                    "gain": image_gain,
                    "offset": image_offset,
                    "cameraid": image_cameraid
                })
                return None

            # Load master frames
            with fits.open(dark_path) as dark_hdul:
                dark_data = dark_hdul[0].data.astype(np.float64)
                dark_naxis1 = dark_hdul[0].header.get('NAXIS1')
                dark_naxis2 = dark_hdul[0].header.get('NAXIS2')
                
            with fits.open(flat_path) as flat_hdul:
                flat_data = flat_hdul[0].data.astype(np.float64)
                flat_naxis1 = flat_hdul[0].header.get('NAXIS1')
                flat_naxis2 = flat_hdul[0].header.get('NAXIS2')

            # Flag saturated pixels before calibration
            saturated_mask = (image_data >= 65504)
            num_saturated = np.sum(saturated_mask)
            
            # Check for ROI mismatch
            is_roi = False
            if (image_naxis1 != dark_naxis1 or image_naxis2 != dark_naxis2 or
                image_naxis1 != flat_naxis1 or image_naxis2 != flat_naxis2):
                is_roi = True
                logging.info(f"ROI detected: Light ({image_naxis1}x{image_naxis2}) vs Dark ({dark_naxis1}x{dark_naxis2}) vs Flat ({flat_naxis1}x{flat_naxis2})")
                if (image_naxis1 > dark_naxis1 or image_naxis2 > dark_naxis2 or
                    image_naxis1 > flat_naxis1 or image_naxis2 > flat_naxis2):
                    self.logger.log_session_event("CALIBRATION_FAIL", "ERROR", "ROI dimensions exceed master frame dimensions.", {
                        "image_shape": (image_naxis1, image_naxis2),
                        "dark_shape": (dark_naxis1, dark_naxis2),
                        "flat_shape": (flat_naxis1, flat_naxis2)
                    })
                    return None

                # ========== CRITICAL FIX: Use measured ROI offsets ==========
                # Get actual ROI offset for this camera and ROI size
                x_offset_dark, y_offset_dark = get_roi_offset(
                    image_cameraid, 
                    image_naxis1, image_naxis2,
                    dark_naxis1, dark_naxis2
                )
                
                x_offset_flat, y_offset_flat = get_roi_offset(
                    image_cameraid,
                    image_naxis1, image_naxis2,
                    flat_naxis1, flat_naxis2
                )
                # ============================================================

                try:
                    dark_data = dark_data[y_offset_dark:y_offset_dark + image_naxis2, x_offset_dark:x_offset_dark + image_naxis1]
                    flat_data = flat_data[y_offset_flat:y_offset_flat + image_naxis2, x_offset_flat:x_offset_flat + image_naxis1]
                    logging.info(f"Cropped dark to {dark_data.shape}, flat to {flat_data.shape} for ROI calibration using offsets ({x_offset_dark},{y_offset_dark}).")
                except Exception as e:
                    self.logger.log_session_event("CALIBRATION_FAIL", "ERROR", f"Failed to crop master frames for ROI: {e}", {
                        "image_shape": (image_naxis1, image_naxis2),
                        "dark_shape": (dark_naxis1, dark_naxis2),
                        "flat_shape": (flat_naxis1, flat_naxis2)
                    })
                    return None

            # Verify shapes after potential cropping
            if image_data.shape != dark_data.shape or image_data.shape != flat_data.shape:
                self.logger.log_session_event("CALIBRATION_FAIL", "ERROR", "Shape mismatch after ROI cropping.", {
                    "image_shape": image_data.shape,
                    "dark_shape": dark_data.shape,
                    "flat_shape": flat_data.shape
                })
                return None
            
            dark_subtracted = image_data - dark_data
            flat_mean = np.mean(flat_data)
            if flat_mean == 0:
                self.logger.log_session_event("CALIBRATION_FAIL", "ERROR", "Master flat mean is zero, cannot divide.", {"file": str(flat_path)})
                return None
            normalized_flat = flat_data / flat_mean
            
            calibrated_data = dark_subtracted / normalized_flat
            
            # After calibration, set the originally saturated pixels to 65535.
            if num_saturated > 0:
                calibrated_data[saturated_mask] = 65535

            info = np.iinfo(original_dtype)
            clipped_data = np.clip(calibrated_data, info.min, info.max).astype(original_dtype)
            
            image_header['CALSTAT'] = 'BDF'
            image_header.add_history(f"Calibrated with Dark: {dark_path.name}")
            image_header.add_history(f"Calibrated with Flat: {flat_path.name}")
            image_header['OBSERVAT'] = ('SFRO', 'Starfront Remote Observatory, Rockwood, TX')
            if is_roi:
                image_header.add_history(f"ROI Calibration: Measured offsets - Dark({x_offset_dark},{y_offset_dark}), Flat({x_offset_flat},{y_offset_flat}) - Camera: {image_cameraid}")
            if num_saturated > 0:
                image_header.add_history(f"Flagged {num_saturated} saturated pixels (>=65504) as 65535.")
            
            new_file_path = image_path.with_stem(f"{image_path.stem}_calibrated")
            fits.PrimaryHDU(data=clipped_data, header=image_header).writeto(new_file_path, overwrite=True)
            self.logger.log_session_event("CALIBRATION_SUCCESS", "SUCCESS", "Image calibrated with corrected ROI alignment.", {
                "original": str(image_path),
                "calibrated": str(new_file_path),
                "dark": str(dark_path),
                "flat": str(flat_path),
                "is_roi": is_roi,
                "roi_offset_dark": (x_offset_dark, y_offset_dark) if is_roi else None,
                "roi_offset_flat": (x_offset_flat, y_offset_flat) if is_roi else None
            })
            return new_file_path
        except Exception as e:
            logging.error(f"Error during calibration of {image_path}: {e}", exc_info=True)
            self.logger.log_session_event("CALIBRATION_FAIL", "ERROR", f"Exception: {e}", {"file": str(image_path)})
            return None

    def _find_master_frame(self, directory, frame_type, exposure, ccd_temp, gain, offset, xbinning, ybinning, filter_name, cameraid):
        """
        Find a matching master calibration frame.
        Matches on CAMERAID primarily, then other parameters.
        """
        if not directory.exists():
            logging.warning(f"Master {frame_type} directory does not exist: {directory}")
            return None

        tolerance = self.config["CCD_TEMP_TOLERANCE"]
        candidates = []

        for fits_file in directory.glob("*.fits"):
            try:
                with fits.open(fits_file) as hdul:
                    header = hdul[0].header
                    
                    # Check CAMERAID first (critical for matching sensor)
                    master_cameraid = header.get('CAMERAID', 'Unknown')
                    if cameraid != 'Unknown' and master_cameraid != 'Unknown':
                        if cameraid != master_cameraid:
                            continue  # Skip if camera doesn't match
                    
                    # Check frame type
                    if header.get('IMAGETYP', '').strip().upper() != frame_type.upper():
                        continue

                    # Check binning
                    if header.get('XBINNING', 1) != xbinning or header.get('YBINNING', 1) != ybinning:
                        continue

                    # For darks: match exposure and temperature
                    if frame_type.lower() == 'dark':
                        if exposure is not None:
                            master_exposure = header.get('EXPOSURE')
                            if master_exposure is None or abs(master_exposure - exposure) > 0.1:
                                continue
                        
                        if ccd_temp is not None:
                            master_temp = header.get('CCD-TEMP')
                            if master_temp is None or abs(master_temp - ccd_temp) > tolerance:
                                continue

                    # For flats: match filter
                    if frame_type.lower() == 'flat':
                        if filter_name is not None:
                            master_filter = header.get('FILTER', 'Luminance')
                            if master_filter.lower() != filter_name.lower():
                                continue

                    # Match gain and offset if present
                    if gain is not None:
                        master_gain = header.get('GAIN')
                        if master_gain is not None and master_gain != gain:
                            continue

                    if offset is not None:
                        master_offset = header.get('OFFSET')
                        if master_offset is not None and master_offset != offset:
                            continue

                    candidates.append(fits_file)

            except Exception as e:
                logging.warning(f"Error reading {fits_file}: {e}")
                continue

        if candidates:
            return candidates[0]
        
        return None

    def plate_solve_image(self, calibrated_path):
        try:
            astap_cli = self.config.get("ASTAP_CLI_PATH")
            if not astap_cli or not Path(astap_cli).exists():
                logging.warning("ASTAP_CLI_PATH not configured or invalid, skipping plate-solve.")
                return

            self.logger.log_session_event("PLATESOLVE_START", "INFO", "Attempting to plate-solve and write WCS to header.", {"file": str(calibrated_path)})
            
            # The key change: Add the '-wcs' flag to the command.
            # This tells ASTAP to update the FITS header of the input file directly.
            cmd = [str(astap_cli), "-f", str(calibrated_path), "-r 1", "-fov 1.44", "-z 2", "-sip", "-wcs", "-update"]

            # We use check=True, so a CalledProcessError will be raised if ASTAP fails.
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)

            # If the subprocess call completes without an exception, it was successful.
            self.logger.log_session_event("PLATESOLVE_SUCCESS", "SUCCESS", "Plate-solving successful, WCS written directly to header.", {
                "file": str(calibrated_path),
                "astap_output": result.stdout.strip()
            })

        except subprocess.CalledProcessError as e:
            # This block executes if ASTAP returns a non-zero exit code (i.e., it failed).
            logging.error(f"ASTAP failed for {calibrated_path}: {e.stderr}", exc_info=True)
            self.logger.log_session_event("PLATESOLVE_FAIL", "ERROR", f"ASTAP failed: {e.stderr}", {"file": str(calibrated_path)})
        except Exception as e:
            # This block handles other potential errors (e.g., file not found).
            logging.error(f"Error during plate-solving of {calibrated_path}: {e}", exc_info=True)
            self.logger.log_session_event("PLATESOLVE_FAIL", "ERROR", f"Exception: {e}", {"file": str(calibrated_path)})

class NinaLogHandler(FileSystemEventHandler):
    def __init__(self, logger): self.logger, self.config = logger, logger.config
    def on_created(self, event):
        if event.is_directory or self.logger.shutdown_event.is_set(): return
        if "final" in event.src_path.lower() and event.src_path.lower().endswith("detection_result.json"):
            path = Path(event.src_path)
            time.sleep(self.config["FILE_WRITE_DELAY_SEC"])
            try:
                with open(path, 'r') as f: data = json.load(f)
                pegasus = self.logger._get_pegasus_data()
                details = {
                    "focuser_position": data.get("FocuserPosition"),
                    "average_hfr": data.get("AverageHFR"),
                    "detected_stars": data.get("DetectedStars"),
                    "hfr_std_dev": data.get("HFRStdDev"),
                    "filter": data.get("Filter", "Unknown"),
                    "ambient_temp_c": pegasus.get('pegasus_temp_c'),
                    "focuser_temp_c": None, # Cannot be known reliably at this point
                    "source_file": str(path)
                }
                self.logger.log_focus_event(details)
                self.logger.log_session_event("AUTOFOCUS_COMPLETE", "SUCCESS", "N.I.N.A. autofocus run detected.", details)
            except Exception as e:
                logging.error(f"Error processing NINA log {path}: {e}", exc_info=True)
                self.logger.log_session_event("AUTOFOCUS_FAIL", "ERROR", f"Failed to parse NINA log: {e}", {"file": str(path)})

if __name__ == "__main__":
    try:
        UnifiedAstroLogger().start()
    except (ValueError, KeyError) as e:
        logging.critical(f"Configuration error prevented start: {e}")
    except Exception as e:
        logging.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
    finally:
        logging.info("Main application has finished.")