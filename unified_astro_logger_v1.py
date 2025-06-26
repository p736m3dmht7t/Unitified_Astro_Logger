# unified_astro_logger.py (v14)
#
# Changelog:
# - Adds Pegasus PPB Advance dew heater power logging (via PEGASUS_PPBA_API_URL).
# - Adds an external shutdown mechanism via a flag file (configured by SHUTDOWN_FLAG_FILE).
# - Full-featured, robust logger with API integration and real-time image processing.
# - Loads all configuration from a .env file.
# - Creates a unified session log and a persistent focus log.
# - Handles wildcard target directories for science frames (e.g., .../{target_name}/LIGHT/).
# - Patiently waits for acquisition directories to be created.
# - Handles API connection errors gracefully and quietly.
# - Supports clean shutdown via 'q' + Enter, Ctrl+C, or flag file.
# - Fully readable, standard Python formatting.

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
                            "PEGASUS_PPBA_API_URL", "SHUTDOWN_FLAG_FILE"]
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
        tf = TimezoneFinder()
        tz_str = tf.timezone_at(lat=self.config["LATITUDE"], lng=self.config["LONGITUDE"])
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
                    "MASTER_DARK_DIR", "MASTER_FLAT_DIR", "SHUTDOWN_FLAG_FILE"]:
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
                with open(file_path, 'a', newline='', encoding='utf--8') as f:
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
            flat_file = self.find_master_flat(fits_data.get('filter'))

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
        if not all([exposure, gain is not None, offset is not None, ccd_temp is not None]):
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
                        
                        if (abs(float(master_exposure) - float(exposure)) < 1e-6 and
                            master_gain == gain and master_offset == offset and
                            isinstance(master_ccd_temp, (int, float))):
                            
                            temp_diff = abs(float(master_ccd_temp) - float(ccd_temp))
                            if temp_diff <= self.config["CCD_TEMP_TOLERANCE"] and temp_diff < min_temp_diff:
                                min_temp_diff, best_match = temp_diff, master_path
                    except Exception as e:
                        logging.warning(f"Could not process master dark {filename}: {e}")
        
        if best_match:
            logging.info(f"Found best dark match: {best_match} (Temp diff: {min_temp_diff:.2f}C)")
        return best_match

    def find_master_flat(self, filter_val):
        if not filter_val:
            logging.warning("No filter value in header, cannot find master flat.")
            return None
            
        master_flat_dir = self.config.get("MASTER_FLAT_DIR")
        if not master_flat_dir or not master_flat_dir.exists():
            logging.warning("MASTER_FLAT_DIR is not configured or does not exist.")
            return None

        for filename in os.listdir(master_flat_dir):
             if "flat" in filename.lower() and filename.lower().endswith(".fits"):
                 try:
                    master_path = master_flat_dir / filename
                    master_header = fits.getheader(master_path)
                    if master_header.get('FILTER', '').upper() == filter_val.upper():
                        logging.info(f"Found master flat match: {master_path}")
                        return master_path
                 except Exception as e:
                     logging.warning(f"Could not process master flat {filename}: {e}")
        return None

    def calibrate_image(self, image_path, dark_path, flat_path):
        try:
            with fits.open(image_path) as hdul:
                image_data = hdul[0].data.astype(np.float32)
                image_header = hdul[0].header
                original_dtype = hdul[0].data.dtype
            
            dark_data = fits.getdata(dark_path).astype(np.float32)
            flat_data = fits.getdata(flat_path).astype(np.float32)

            if image_data.shape != dark_data.shape or image_data.shape != flat_data.shape:
                self.logger.log_session_event("CALIBRATION_FAIL", "ERROR", "Shape mismatch.", {"image": image_data.shape, "dark": dark_data.shape, "flat": flat_data.shape})
                return None
            
            dark_subtracted = image_data - dark_data
            flat_mean = np.mean(flat_data)
            if flat_mean == 0:
                self.logger.log_session_event("CALIBRATION_FAIL", "ERROR", "Master flat mean is zero, cannot divide.", {"file": str(flat_path)})
                return None
            normalized_flat = flat_data / flat_mean
            
            calibrated_data = dark_subtracted / normalized_flat

            info = np.iinfo(original_dtype)
            clipped_data = np.clip(calibrated_data, info.min, info.max).astype(original_dtype)
            
            image_header['CALSTAT'] = 'BDF'
            image_header.add_history(f"Calibrated with Dark: {dark_path.name}")
            image_header.add_history(f"Calibrated with Flat: {flat_path.name}")
            
            new_file_path = image_path.with_stem(f"{image_path.stem}_calibrated")
            fits.PrimaryHDU(data=clipped_data, header=image_header).writeto(new_file_path, overwrite=True)
            self.logger.log_session_event("CALIBRATION_SUCCESS", "SUCCESS", "Image calibrated.", {"original": str(image_path), "calibrated": str(new_file_path), "dark": str(dark_path), "flat": str(flat_path)})
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
            
            self.logger.log_session_event("PLATESOLVE_START", "INFO", "Attempting to plate-solve.", {"file": str(calibrated_path)})
            cmd = [str(astap_cli), "-f", str(calibrated_path), "-z", "2"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            
            if calibrated_path.with_suffix(".wcs").exists():
                self.logger.log_session_event("PLATESOLVE_SUCCESS", "SUCCESS", "Plate-solving successful.", {"file": str(calibrated_path)})
            else:
                 self.logger.log_session_event("PLATESOLVE_FAIL", "WARNING", "ASTAP ran but did not produce a .wcs solution.", {"output": result.stdout})
        except subprocess.CalledProcessError as e:
            logging.error(f"ASTAP failed for {calibrated_path}: {e.stderr}", exc_info=True)
            self.logger.log_session_event("PLATESOLVE_FAIL", "ERROR", f"ASTAP failed: {e.stderr}", {"file": str(calibrated_path)})
        except Exception as e:
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