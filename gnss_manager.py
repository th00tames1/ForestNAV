"""
This module provides a GNSSManager class for reading NMEA sentences from
a serial port and emitting parsed data via Qt signals.  It is adapted
from the original ``GNSS_manager.py`` contained in the ``gnss.zip``
archive but uses PyQt5 rather than PyQt6 so that it can be integrated
into the ForestNAV application.  See the top‑level ``gnss_manager.py``
in the root of this repository for full documentation.  This file
exists inside the ``ForestNAV_src`` package so that ``main_updated.py``
can import it when deployed as part of the integrated ForestNAV
software.

The public API and behaviour of the class remain the same: after
instantiation with a serial port and optional baud rate, the
``start()`` method spawns a background thread that continuously reads
from the serial port, parses RMC and GGA NMEA sentences, and updates
the latest position, speed, bearing and fix quality.  Whenever new
RMC data is parsed the ``newDataAvailable`` signal is emitted.  The
``status`` signal is emitted when the GNSSManager is started or
stopped, and if any serial errors occur.
"""

import serial
import time
import re
from threading import Event, Thread
from typing import Optional, Tuple
from PyQt5.QtCore import QObject, pyqtSignal

# --- simple NMEA -> decimal degrees conversion ---
def _nmea_to_decimal(coord: str, direction: str) -> Optional[float]:
    """Convert NMEA coordinate (ddmm.mmmm or dddmm.mmmm) to decimal degrees.

    Parameters
    ----------
    coord : str
        The NMEA coordinate string.  Must contain a decimal point.
    direction : str
        One of 'N', 'S', 'E' or 'W' indicating the hemisphere.

    Returns
    -------
    float or None
        The coordinate in decimal degrees, rounded to 6 decimal places, or
        ``None`` if the input cannot be parsed.
    """
    if not coord or '.' not in coord:
        return None
    # latitude uses 2-digit degrees, longitude uses 3-digit degrees
    d_len = 2 if direction in ('N', 'S') else 3
    try:
        degrees = int(coord[:d_len])
        minutes = float(coord[d_len:])
    except (ValueError, IndexError):
        return None
    dec = degrees + minutes / 60.0
    if direction in ('S', 'W'):
        dec = -dec
    return round(dec, 6)

# --- NMEA regex patterns for RMC and GGA sentences ---
rmc_pattern = re.compile(r'''
    \$(?:GP|GN)RMC,        # RMC sentence (GPRMC or GNRMC)
    [^,]*,                  # UTC time
    [AVR],                  # status (A=active, V=void, R=RTK)
    (?P<lat>[^,]*),(?P<lat_dir>[NS]),
    (?P<lon>[^,]*),(?P<lon_dir>[EW]),
    (?P<spd>[^,]*),         # speed over ground in knots
    (?P<dir>[^,]*),         # track angle in degrees
    .*                      # remainder of sentence
''', re.VERBOSE)

gga_pattern = re.compile(r'''
    \$(?:GP|GN)GGA,        # GGA sentence (GPGGA or GNGGA)
    [^,]*,                  # UTC time
    (?P<lat>[^,]*),(?P<lat_dir>[NS]),
    (?P<lon>[^,]*),(?P<lon_dir>[EW]),
    (?P<fix>\d+),          # fix quality
    .*                      # remainder of sentence
''', re.VERBOSE)


class GNSSManager(QObject):
    """Reads GNSS NMEA data from a serial port in the background.

    The manager parses RMC and GGA sentences to extract position
    (latitude and longitude), speed, bearing and fix quality.  It
    exposes the latest parsed values via :meth:`get_latest_data` and
    emits signals when new data is available or when the status
    changes.

    Parameters
    ----------
    port : str
        Name of the serial port, e.g. ``'COM3'`` on Windows or
        ``'/dev/ttyUSB0'`` on Linux.
    baud : int, optional
        Baud rate for the serial port.  Defaults to 9600.
    """

    newDataAvailable = pyqtSignal()
    status = pyqtSignal(str)

    def __init__(self, port: str, baud: int = 9600) -> None:
        super().__init__()
        self.port = port
        self.baud = baud
        self._stop_event = Event()
        self._thread: Optional[Thread] = None
        self._latest_lat: Optional[float] = None
        self._latest_lon: Optional[float] = None
        self._latest_speed: Optional[float] = None
        self._latest_bearing: Optional[float] = None
        self._latest_fix_quality: Optional[int] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()
        self.status.emit("REAL GNSS started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join()
        self.status.emit("GNSS stopped")

    def get_latest_data(self) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[int]]:
        return (
            self._latest_lat,
            self._latest_lon,
            self._latest_speed,
            self._latest_bearing,
            self._latest_fix_quality,
        )

    def _run(self) -> None:
        try:
            ser = serial.Serial(self.port, self.baud, timeout=1)
        except Exception as e:
            self.status.emit(f"❌ Serial open failed: {e}")
            return
        while not self._stop_event.is_set():
            try:
                raw = ser.readline()
            except Exception:
                time.sleep(0.05)
                continue
            try:
                line = raw.decode('utf-8', errors='ignore').strip()
            except Exception:
                continue
            m_gga = gga_pattern.search(line)
            if m_gga:
                try:
                    self._latest_fix_quality = int(m_gga.group('fix'))
                except ValueError:
                    pass
            m_rmc = rmc_pattern.search(line)
            if m_rmc:
                lat = _nmea_to_decimal(m_rmc.group('lat'), m_rmc.group('lat_dir'))
                lon = _nmea_to_decimal(m_rmc.group('lon'), m_rmc.group('lon_dir'))
                try:
                    spd_knots = float(m_rmc.group('spd')) if m_rmc.group('spd') else 0.0
                except ValueError:
                    spd_knots = 0.0
                speed_m_s = spd_knots * 0.514444
                try:
                    bearing = float(m_rmc.group('dir')) if m_rmc.group('dir') else 0.0
                except ValueError:
                    bearing = 0.0
                if lat is not None and lon is not None:
                    self._latest_lat = lat
                    self._latest_lon = lon
                    self._latest_speed = speed_m_s
                    self._latest_bearing = bearing
                    self.newDataAvailable.emit()
            time.sleep(0.01)
        ser.close()