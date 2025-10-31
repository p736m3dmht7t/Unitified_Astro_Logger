# unified_astro_logger_v5.py (v18)
#
# Changelog:
# - v18: Enhanced ROI calibration to use camera-specific measured offsets from .env configuration. Falls back to centered ROI assumption if offsets not configured. Supports CAMERA_{CAMERA_NAME}_ROI_{WIDTH}x{HEIGHT}_{X|Y}_OFFSET format.
# - v17: Added support for calibrating Region of Interest (ROI) images where NAXIS1/NAXIS2 differ from master frames. Matches on CAMERAID, assumes centered ROI, crops master dark/flat to match light frame dimensions for pixel-accurate calibration.
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
        logging.info(f"Unified Astro Logger initialized for session date: {self.astro_date_str}")
        clean_config = {k: str(v) for k, v in self.config.items() if "API_KEY" not in k}
        self.log_session_event("SESSION_START", "SUCCESS", "Logger started.", {"config": clean_config})

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
        
        # Load camera ROI offset configurations
        self.config["CAMERA_ROI_OFFSETS"] = {}
        for key, value in os.environ.items():
            if "_ROI_" in key and ("_X_OFFSET" in key or "_Y_OFFSET" in key):
                try:
                    # Parse: CAMERA_{CAMERA_NAME}_ROI_{WIDTH}x{HEIGHT}_{X|Y}_OFFSET
                    parts = key.split("_")
                    if len(parts) >= 5 and parts[0] == "CAMERA":
                        # Find where ROI starts
                        roi_idx = next(i for i, p in enumerate(parts) if p == "ROI")
                        camera_name = "_".join(parts[1:roi_idx])
                        roi_dim_str = parts[roi_idx + 1]  # e.g., "2744x1836"
                        axis = parts[-2]  # "X" or "Y"
                        
                        if camera_name not in self.config["CAMERA_ROI_OFFSETS"]:
                            self.config["CAMERA_ROI_OFFSETS"][camera_name] = {}
                        if roi_dim_str not in self.config["CAMERA_ROI_OFFSETS"][camera_name]:
                            self.config["CAMERA_ROI_OFFSETS"][camera_name][roi_dim_str] = {}
                        
                        self.config["CAMERA_ROI_OFFSETS"][camera_name][roi_dim_str][axis.lower()] = int(value)
                        logging.debug(f"Loaded ROI offset: {camera_name} {roi_dim_str} {axis}={value}")
                except (ValueError, StopIteration, IndexError) as e:
                    logging.warning(f"Could not parse camera ROI offset key '{key}': {e}")
            
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

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _get_pegasus_data(self):
        try:
            # Get unique key - this is needed for all subsequent device calls
            dev_resp = requests.get(self.config["PEGASUS_DEVICEMANAGER_URL"], timeout=3)
            dev_resp.raise_for_status()
            unique_key = dev_resp.json()['data'][0]['uniqueKey']
            
            all_pegasus_data = {}
            
            # Get environmental data from ObservingConditions device
            api_url = f"{self.config['PEGASUS_API_URL']}?DriverUniqueKey={unique_key}"
            resp = requests.get(api_url, timeout=3)
            resp.raise_for_status()
            msg = resp.json()['data']['message']
            all_pegasus_data.update({
                'pegasus_temp_c': msg.get('temperature'),
                'pegasus_humidity_pct': msg.get('humidity'),
                'pegasus_dewpoint_c': msg.get('dewPoint')
            })
            
            # Get dew heater data from PPB Advance if configured
            if self.config.get("PEGASUS_PPBA_API_URL"):
                ppba_url = f"{self.config['PEGASUS_PPBA_API_URL']}?DriverUniqueKey={unique_key}"
                ppba_resp = requests.get(ppba_url, timeout=3)
                ppba_resp.raise_for_status()
                ppba_data = ppba_resp.json()
                
                dew_power = None
                hub_items = ppba_data.get('data', {}).get('message', {}).get('hub', [])
                for item in hub_items:
                    if item.get('messageType') == 'DewPortStatus' and item.get('port', {}).get('number') == 1:
                        dew_power = item.get('port', {}).get('power')
                        break
                if dew_power is not None:
                    all_pegasus_data['pegasus_dewheater_pct'] = dew_power
                else:
                    logging.warning("Could not find Dew Port 1 power in Pegasus PPBA response.")
            
            if self.pegasus_api_is_down:
                logging.info("Successfully re-established connection to Pegasus API.")
                self.pegasus_api_is_down = False
            return all_pegasus_data
            
        except Exception as e:
            if not self.pegasus_api_is_down:
                logging.warning(f"Could not get Pegasus data: {e}. Will not log this error again until connection recovers.")
                self.pegasus_api_is_down = True
            return {}

    def _get_boltwood_data(self):
        try:
            if self.config["BOLTWOOD_FILE_PATH"] and self.config["BOLTWOOD_FILE_PATH"].exists():
                with open(self.config["BOLTWOOD_FILE_PATH"], 'r') as f:
                    lines = [line.strip() for line in f if line.strip()]
                if lines:
                    return {"boltwood_last_line": lines[-1]}
        except Exception as e:
            logging.warning(f"Could not read Boltwood file: {e}")
        return {}

    def _get_time_sync_status(self):
        try:
            result = subprocess.run(['w32tm', '/query', '/status'], capture_output=True, text=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            data = {}
            for line in result.stdout.strip().split('\n'):
                if ':' in line:
                    key, val = [s.strip() for s in line.split(':', 1)]
                    key = key.lower().replace(' ', '_').replace('(', '').replace(')', '')
                    if key == 'referenceid':
                        key, val = 'source_ip', val.split('source IP:')[1].strip().replace(')', '')
                    if key in ['stratum', 'last_successful_sync_time', 'source', 'source_ip', 'root_delay', 'root_dispersion']:
                        data[f"time_{key}"] = val
            return data
        except Exception as e:
            logging.warning(f"Could not check time sync: {e}")
            return {'time_sync_error': str(e)}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def _get_openweathermap_data(self):
        params = {
            'lat': self.config["LATITUDE"],
            'lon': self.config["LONGITUDE"],
            'appid': self.config["OPENWEATHERMAP_API_KEY"],
            'units': 'metric'
        }
        try:
            response = requests.get("https://api.openweathermap.org/data/2.5/weather", params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return {
                'owm_temp_c': data.get('main', {}).get('temp'),
                'owm_feels_like_c': data.get('main', {}).get('feels_like'),
                'owm_pressure_hpa': data.get('main', {}).get('pressure'),
                'owm_humidity_pct': data.get('main', {}).get('humidity'),
                'owm_wind_speed_ms': data.get('wind', {}).get('speed'),
                'owm_wind_deg': data.get('wind', {}).get('deg'),
                'owm_cloud_pct': data.get('clouds', {}).get('all'),
                'owm_description': data.get('weather', [{}])[0].get('description')
            }
        except Exception as e:
            logging.warning(f"Could not get OpenWeatherMap data: {e}")
            return {}

    def _check_roof_status(self):
        """Checks roof status, copies file, and applies failsafe logic if source is unavailable."""
        source_path = self.config.get("ROOF_STATUS_SOURCE_PATH")
        dest_path = self.config.get("ROOF_STATUS_DEST_PATH")

        if not source_path or not dest_path:
            return  # Silently skip if not configured

        try:
            # Attempt to read the source file from the network
            with open(source_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # If successful, write the exact content to the destination
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # Parse the status from the content and log it
            roof_status = "UNKNOWN"
            if "CLOSED" in content.upper():
                roof_status = "CLOSED"
            elif "OPEN" in content.upper():
                roof_status = "OPEN"
            
            self.log_session_event("ROOF_STATUS_UPDATE", roof_status, "Successfully updated roof status from source.", {"source": str(source_path), "content": content.strip()})

        except (IOError, OSError) as e:
            # This block executes if the source file is inaccessible (e.g., network error)
            logging.warning(f"Could not access roof status file at {source_path}: {e}")
            self.log_session_event("ROOF_STATUS_UPDATE", "STALE", f"Could not access source roof status file: {e}", {"source": str(source_path)})

            # Failsafe: Check sun's position
            now = Time.now()
            sun_alt = self.astro_observer.sun_alt(now).to(u.deg)
            
            if sun_alt.value > 5:
                logging.warning(f"Sun is up ({sun_alt:.2f}). Forcing roof status to CLOSED at destination as a failsafe.")
                self.log_session_event("ROOF_STATUS_UPDATE", "SUN_UP", f"Source stale, sun up ({sun_alt:.2f}). Forcing destination to CLOSED.", {"destination": str(dest_path)})
                
                try:
                    # Construct the standard "CLOSED" content with the current local time
                    tz_str = self.tf.timezone_at(lat=self.config["LATITUDE"], lng=self.config["LONGITUDE"])
                    local_tz = pytz.timezone(tz_str)
                    now_local = datetime.datetime.now(pytz.utc).astimezone(local_tz)
                    timestamp_str = now_local.strftime('%Y-%m-%d %I:%M:%S%p') # e.g., 2025-08-08 08:09:44AM
                    new_content = f"???{timestamp_str} Roof Status: CLOSED"

                    # Write the failsafe "CLOSED" status to the destination
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(dest_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)

                except Exception as write_e:
                    logging.error(f"Failed to write SUN_UP forced CLOSED status to {dest_path}: {write_e}")
                    self.log_session_event("ROOF_STATUS_FAIL", "ERROR", f"Failed to write forced CLOSED status: {write_e}", {"destination": str(dest_path)})
            # else: If sun is down, we do nothing and leave the last known good file at the destination.

    def _read_fits_header_data(self, file_path):
        details = {'file': str(file_path)}
        try:
            with fits.open(file_path) as hdul:
                header = hdul[0].header
                def get_keyword(keys):
                    return next((header[key] for key in keys if key in header), None)
                details.update({
                    'exposure': get_keyword(['EXPOSURE', 'EXPTIME']),
                    'ccd_temp': get_keyword(['CCD-TEMP', 'SET-TEMP']),
                    'filter': get_keyword(['FILTER']),
                    'gain': get_keyword(['GAIN']),
                    'offset': get_keyword(['OFFSET']),
                    'focuser_position': get_keyword(['FOCPOS', 'FOCUSPOS']),
                    'focuser_temp': get_keyword(['FOCTEMP', 'FOCUSTEM']),
                    'weather_cloudcover': get_keyword(['CLOUDCVR']),
                    'weather_dewpoint': get_keyword(['DEWPOINT']),
                    'weather_humidity': get_keyword(['HUMIDITY']),
                    'weather_pressure': get_keyword(['PRESSURE']),
                    'weather_ambtemp': get_keyword(['AMBTEMP']),
                    'weather_winddir': get_keyword(['WINDDIR']),
                    'weather_windspeed': get_keyword(['WINDSPD']),
                    'camera_id': get_keyword(['CAMERAID']),
                    'naxis1': header.get('NAXIS1'),
                    'naxis2': header.get('NAXIS2')
                })
        except Exception as e:
            logging.error(f"Could not read FITS header for {file_path}: {e}")
            details['fits_header_error'] = str(e)
        return details

    def _periodic_logger_thread(self):
        threading.current_thread().name = "PeriodicLogger"
        logging.info("Periodic logger thread started.")
        while not self.shutdown_event.wait(self.config["PERIODIC_LOG_INTERVAL_SEC"]):
            status_data = {}
            status_data.update(self._get_pegasus_data())
            status_data.update(self._get_boltwood_data())
            status_data.update(self._get_time_sync_status())
            status_data.update(self._get_openweathermap_data())
            if status_data:
                self.log_session_event("PERIODIC_STATUS", "INFO", "Periodic environment & time check.", status_data)

            # Check roof status on each periodic cycle
            self._check_roof_status()

        logging.info("Periodic logger thread finished.")

    def _file_monitor_thread(self, thread_name, directory_path_str, handler_class, wait_for_creation=False):
        threading.current_thread().name = thread_name
        path = Path(directory_path_str) if directory_path_str else None
        if not path:
            logging.warning(f"{thread_name}: Directory path is not configured. Thread will not start.")
            return

        if wait_for_creation:
            logging.info(f"Waiting for directory to be created: {path}")
            while not path.exists():
                if self.shutdown_event.wait(timeout=15):
                    logging.info(f"Shutdown signal received while waiting for {path}. Exiting thread.")
                    return
            logging.info(f"Directory found: {path}. Initializing monitor.")
        elif not path.exists():
            logging.error(f"Directory does not exist and not waiting: {path}. Exiting thread.")
            return

        path.mkdir(parents=True, exist_ok=True)
        observer = Observer()
        observer.schedule(handler_class(self), str(path), recursive=True)
        observer.start()
        self.shutdown_event.wait()
        observer.stop()
        observer.join()
        logging.info(f"{thread_name} finished.")

    def _input_monitor_thread(self):
        threading.current_thread().name = "InputMonitor"
        while not self.shutdown_event.is_set():
            try:
                text = input()
                if text.lower() == 'q':
                    logging.info("'q' key pressed. Initiating shutdown...")
                    self.stop()
                    break
            except EOFError:
                break

    def _shutdown_monitor_thread(self):
        threading.current_thread().name = "ShutdownMonitor"
        flag_file_path = self.config.get("SHUTDOWN_FLAG_FILE")
        
        if not flag_file_path:
            logging.debug("SHUTDOWN_FLAG_FILE not configured. External shutdown monitor thread will not run.")
            return

        logging.info(f"External shutdown monitor started. Watching for: {flag_file_path}")
        while not self.shutdown_event.is_set():
            if flag_file_path.exists():
                logging.info(f"Shutdown flag file found at {flag_file_path}. Initiating shutdown.")
                try:
                    flag_file_path.unlink() # Delete the file to prevent re-triggering
                    logging.info("Shutdown flag file removed.")
                except OSError as e:
                    logging.error(f"Could not remove shutdown flag file {flag_file_path}: {e}")
                
                self.stop()
                break
            
            # Wait for 2 seconds or until the global shutdown event is set
            if self.shutdown_event.wait(timeout=2):
                break
                
        logging.info("External shutdown monitor thread finished.")

    def start(self):
        input_thread = threading.Thread(target=self._input_monitor_thread, daemon=True)
        threads = [
            input_thread,
            threading.Thread(target=self._periodic_logger_thread),
            threading.Thread(target=self._shutdown_monitor_thread),
            threading.Thread(target=self._file_monitor_thread, args=("ImageMonitor", self.config["IMAGE_BASE_DIR"], ImageFileHandler, True)),
            threading.Thread(target=self._file_monitor_thread, args=("AutofocusMonitor", self.config["NINA_LOG_DIR"], NinaLogHandler, True))
        ]
        logging.info("Starting all threads. Press 'q' then Enter, Ctrl+C, or use flag file to stop.")
        for t in threads:
            t.start()
        try:
            for t in threads:
                if not t.daemon and t.is_alive():
                    t.join()
        except KeyboardInterrupt:
            logging.info("Ctrl+C received. Shutting down...")
            self.stop()
        for t in threads:
             if not t.daemon and t.is_alive():
                 t.join()

    def stop(self):
        if not self.shutdown_event.is_set():
            self.shutdown_event.set()
            self.log_session_event("SESSION_END", "SUCCESS", "Logger shut down by user.", {})
            logging.info("Shutdown signal sent.")

class ImageFileHandler(FileSystemEventHandler):
    def __init__(self, logger):
        self.logger = logger
        self.config = logger.config
        self.processed = set()

    def on_created(self, event):
        if event.is_directory or self.logger.shutdown_event.is_set():
            return
        path = Path(event.src_path)
        if path.suffix.lower() not in [".fits", ".fit"] or str(path) in self.processed:
            return
        if "_calibrated" in path.stem or "_astap" in path.stem:
            return
        if 'LIGHT' not in [p.upper() for p in path.parent.parts]:
            return

        logging.info(f"Processing detected science frame: {path}")
        time.sleep(self.config["FILE_WRITE_DELAY_SEC"])
        self.processed.add(str(path))
        self.process_image(path)

    def process_image(self, image_path):
        try:
            fits_data = self.logger._read_fits_header_data(image_path)
            if fits_data.get('fits_header_error'):
                self.logger.log_session_event("IMAGE_ERROR", "ERROR", "Could not read FITS header.", fits_data)
                return

            self.logger.log_session_event("IMAGE_ACQUIRED", "SUCCESS", "New science image acquired.", fits_data)
            self.logger.log_session_event("CALIBRATION_START", "INFO", "Searching for master calibration frames.", {"file": str(image_path)})
            
            dark_file = self.find_master_dark(fits_data)
            flat_file = self.find_master_flat(fits_data.get('filter'), fits_data.get('camera_id'))

            if not dark_file or not flat_file:
                self.logger.log_session_event("CALIBRATION_FAIL", "WARNING", "Could not find suitable master frames.", {"dark_found": str(dark_file), "flat_found": str(flat_file)})
                return

            calibrated_image_path = self.calibrate_image(image_path, dark_file, flat_file)
            if not calibrated_image_path:
                return

            self.plate_solve_image(calibrated_image_path)
        except Exception as e:
            logging.error(f"Unhandled error in processing pipeline for {image_path}: {e}", exc_info=True)
            self.logger.log_session_event("PROCESS_FAIL", "ERROR", f"Unhandled exception: {e}", {"file": str(image_path)})

    def find_master_dark(self, fits_data):
        exposure = fits_data.get('exposure')
        gain = fits_data.get('gain')
        offset = fits_data.get('offset')
        ccd_temp = fits_data.get('ccd_temp')
        camera_id = fits_data.get('camera_id')
        if not camera_id: logging.warning("No CAMERAID in light frame header, master dark selection may fail.")
        if not all([exposure, gain is not None, offset is not None, ccd_temp is not None, camera_id]):
            logging.warning("Missing header info for finding master dark.")
            return None
        
        best_match, min_temp_diff = None, float('inf')
        master_dark_dir = self.config.get("MASTER_DARK_DIR")
        if not master_dark_dir or not master_dark_dir.exists():
            logging.warning("MASTER_DARK_DIR is not configured or does not exist.")
            return None

        for dirpath, _, filenames in os.walk(master_dark_dir):
            for filename in filenames:
                if "dark" in filename.lower() and filename.lower().endswith(".fits"):
                    try:
                        master_path = Path(dirpath) / filename
                        master_header = fits.getheader(master_path)
                        master_exposure = master_header.get('EXPOSURE') or master_header.get('EXPTIME')
                        master_gain = master_header.get('GAIN')
                        master_offset = master_header.get('OFFSET')
                        master_ccd_temp = master_header.get('CCD-TEMP')
                        master_camera_id = master_header.get('CAMERAID')
                        
                        if (abs(float(master_exposure) - float(exposure)) < 1e-6 and
                            master_gain == gain and
                            master_offset == offset and
                            isinstance(master_ccd_temp, (int, float)) and
                            master_camera_id == camera_id):
                            
                            temp_diff = abs(float(master_ccd_temp) - float(ccd_temp))
                            if temp_diff <= self.config["CCD_TEMP_TOLERANCE"] and temp_diff < min_temp_diff:
                                min_temp_diff, best_match = temp_diff, master_path
                    except Exception as e:
                        logging.warning(f"Could not process master dark {filename}: {e}")
        
        if best_match:
            logging.info(f"Found best dark match: {best_match} (Temp diff: {min_temp_diff:.2f}C)")
        return best_match

    def find_master_flat(self, filter_val, camera_id=None):
        if not filter_val:
            logging.warning("No filter value in header, cannot find master flat.")
            return None

        if not camera_id:
            logging.warning("No CAMERAID in light frame header, master flat selection may fail.")

        master_flat_dir = self.config.get("MASTER_FLAT_DIR")
        if not master_flat_dir or not master_flat_dir.exists():
            logging.warning("MASTER_FLAT_DIR is not configured or does not exist.")
            return None

        for filename in os.listdir(master_flat_dir):
             if "flat" in filename.lower() and filename.lower().endswith(".fits"):
                 try:
                    master_path = master_flat_dir / filename
                    master_header = fits.getheader(master_path)
                    if (master_header.get('FILTER', '').upper() == filter_val.upper() and master_header.get('CAMERAID') == camera_id):
                        logging.info(f"Found master flat match: {master_path}")
                        return master_path
                 except Exception as e:
                     logging.warning(f"Could not process master flat {filename}: {e}")
        return None

    def _get_roi_offsets(self, camera_id, roi_width, roi_height):
        """
        Get ROI offsets for a camera based on camera_id and ROI dimensions.
        Returns (x_offset, y_offset) tuple, or None if not configured.
        Falls back to centered ROI if no configuration found.
        """
        if not camera_id:
            return None
        
        roi_offsets = self.config.get("CAMERA_ROI_OFFSETS", {})
        roi_dim_str = f"{roi_width}x{roi_height}"
        
        # Try to match camera name in config
        # Camera IDs in FITS headers may have different formats, try common variations
        camera_id_upper = camera_id.upper()
        
        # Check for exact match first
        for camera_name, rois in roi_offsets.items():
            camera_name_upper = camera_name.upper().replace("_", " ").replace("-", " ")
            # Check if camera_id contains camera_name (flexible matching)
            if camera_name_upper in camera_id_upper or camera_id_upper in camera_name_upper:
                if roi_dim_str in rois:
                    offsets = rois[roi_dim_str]
                    x_offset = offsets.get("x")
                    y_offset = offsets.get("y")
                    if x_offset is not None and y_offset is not None:
                        logging.info(f"Using configured ROI offsets for {camera_name} {roi_dim_str}: ({x_offset}, {y_offset})")
                        return (x_offset, y_offset)
                    break
        
        logging.debug(f"No configured ROI offsets found for camera_id={camera_id}, roi={roi_dim_str}. Will use centered ROI.")
        return None

    def calibrate_image(self, image_path, dark_path, flat_path):
        try:
            with fits.open(image_path) as hdul:
                image_data = hdul[0].data.astype(np.float32)
                image_header = hdul[0].header
                original_dtype = hdul[0].data.dtype
                image_naxis1 = image_header.get('NAXIS1')
                image_naxis2 = image_header.get('NAXIS2')

            # Identify saturated pixels in the raw image BEFORE any calibration.
            saturated_mask = image_data >= 65504
            num_saturated = np.sum(saturated_mask)
            if num_saturated > 0:
                logging.info(f"Detected {num_saturated} saturated pixels (>=65504) in raw image {image_path.name}.")

            dark_data = fits.getdata(dark_path).astype(np.float32)
            flat_data = fits.getdata(flat_path).astype(np.float32)
            dark_header = fits.getheader(dark_path)
            flat_header = fits.getheader(flat_path)

            # Check for ROI by comparing dimensions
            dark_naxis1, dark_naxis2 = dark_header.get('NAXIS1'), dark_header.get('NAXIS2')
            flat_naxis1, flat_naxis2 = flat_header.get('NAXIS1'), flat_header.get('NAXIS2')
            is_roi = (image_naxis1 != dark_naxis1 or image_naxis2 != dark_naxis2 or
                      image_naxis1 != flat_naxis1 or image_naxis2 != flat_naxis2)
            using_configured_offsets = False

            if is_roi:
                logging.info(f"ROI detected: Light ({image_naxis1}x{image_naxis2}) vs Dark ({dark_naxis1}x{dark_naxis2}) vs Flat ({flat_naxis1}x{flat_naxis2})")
                if (image_naxis1 > dark_naxis1 or image_naxis2 > dark_naxis2 or
                    image_naxis1 > flat_naxis1 or image_naxis2 > flat_naxis2):
                    self.logger.log_session_event("CALIBRATION_FAIL", "ERROR", "ROI dimensions exceed master frame dimensions.", {
                        "image_shape": (image_naxis1, image_naxis2),
                        "dark_shape": (dark_naxis1, dark_naxis2),
                        "flat_shape": (flat_naxis1, flat_naxis2)
                    })
                    return None

                # Try to get configured ROI offsets, fall back to centered if not available
                camera_id = image_header.get('CAMERAID')
                configured_offsets = self._get_roi_offsets(camera_id, image_naxis1, image_naxis2)
                
                if configured_offsets:
                    # Use configured offsets (measured empirically)
                    x_offset_dark = configured_offsets[0]
                    y_offset_dark = configured_offsets[1]
                    x_offset_flat = configured_offsets[0]  # Same offsets for both dark and flat
                    y_offset_flat = configured_offsets[1]
                    using_configured_offsets = True
                    logging.info(f"Using configured ROI offsets: ({x_offset_dark}, {y_offset_dark})")
                else:
                    # Fall back to centered ROI assumption
                    x_offset_dark = (dark_naxis1 - image_naxis1) // 2
                    y_offset_dark = (dark_naxis2 - image_naxis2) // 2
                    x_offset_flat = (flat_naxis1 - image_naxis1) // 2
                    y_offset_flat = (flat_naxis2 - image_naxis2) // 2
                    logging.info(f"No configured offsets found, using centered ROI assumption: dark=({x_offset_dark}, {y_offset_dark}), flat=({x_offset_flat}, {y_offset_flat})")

                try:
                    dark_data = dark_data[y_offset_dark:y_offset_dark + image_naxis2, x_offset_dark:x_offset_dark + image_naxis1]
                    flat_data = flat_data[y_offset_flat:y_offset_flat + image_naxis2, x_offset_flat:x_offset_flat + image_naxis1]
                    logging.info(f"Cropped dark to {dark_data.shape}, flat to {flat_data.shape} for ROI calibration.")
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
                offset_method = "configured offsets" if using_configured_offsets else "centered assumption"
                image_header.add_history(f"Calibrated as ROI ({offset_method}): Cropped dark by ({x_offset_dark}, {y_offset_dark}), flat by ({x_offset_flat}, {y_offset_flat})")
            if num_saturated > 0:
                image_header.add_history(f"Flagged {num_saturated} saturated pixels (>=65504) as 65535.")
            
            new_file_path = image_path.with_stem(f"{image_path.stem}_calibrated")
            fits.PrimaryHDU(data=clipped_data, header=image_header).writeto(new_file_path, overwrite=True)
            self.logger.log_session_event("CALIBRATION_SUCCESS", "SUCCESS", "Image calibrated.", {
                "original": str(image_path),
                "calibrated": str(new_file_path),
                "dark": str(dark_path),
                "flat": str(flat_path),
                "is_roi": is_roi
            })
            return new_file_path
        except Exception as e:
            logging.error(f"Error during calibration of {image_path}: {e}", exc_info=True)
            self.logger.log_session_event("CALIBRATION_FAIL", "ERROR", f"Exception: {e}", {"file": str(image_path)})
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
