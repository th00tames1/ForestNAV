import sys, os, time, mmap, chardet
import folium, tempfile, geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
import logging
import fiona
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings
from PyQt5.QtCore import QSettings
from collections import defaultdict
import itertools
from typing import Optional

# Import custom modules
from pri_parser import PRIParser
from data_visualizer import DataVisualizer

# ----- GNSS integration imports -----
# CSV logging and timezone support for GNSS logging
import csv
from datetime import datetime
# Attempt to import pytz for timezone support; fall back to None on failure
try:
    import pytz  # type: ignore
except Exception:
    pytz = None  # type: ignore

# Import our PyQt5‑adapted GNSSManager and tile downloader
from gnss_manager import GNSSManager
from tile_downloader import download_tiles_multi_zoom

# Additional imports for GNSS improvements
import json
try:
    # PySerial is used to enumerate serial ports.  We import both the
    # top‑level ``serial`` package and its ``tools.list_ports`` submodule.
    import serial  # type: ignore[attr-defined]
    import serial.tools.list_ports  # type: ignore[attr-defined]
except Exception:
    # If pyserial is unavailable, set serial to None so that downstream code
    # can gracefully handle the lack of port enumeration.
    serial = None  # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('forestNAV_gui')

class FlowLayout(QtWidgets.QLayout):
    """A flow layout that arranges child widgets horizontally and wraps them."""
    def __init__(self, parent=None, margin=0, spacing=-1):
        super().__init__(parent)
        if parent is not None:
            self.setContentsMargins(margin, margin, margin, margin)
        self._items = []
        self.setSpacing(spacing)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, idx):
        return self._items[idx] if 0 <= idx < len(self._items) else None

    def takeAt(self, idx):
        return self._items.pop(idx) if 0 <= idx < len(self._items) else None

    def expandingDirections(self):
        return QtCore.Qt.Orientations(QtCore.Qt.Horizontal)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._doLayout(QtCore.QRect(0,0,width,0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._doLayout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QtCore.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        size += QtCore.QSize(m.left()+m.right(), m.top()+m.bottom())
        return size

    def _doLayout(self, rect, testOnly):
        x, y = rect.x(), rect.y()
        lineH = 0
        for item in self._items:
            w = item.widget().sizeHint().width()
            h = item.widget().sizeHint().height()
            spaceX = self.spacing()
            spaceY = self.spacing()
            if x + w > rect.right() and lineH > 0:
                x = rect.x()
                y += lineH + spaceY
                lineH = 0
            if not testOnly:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), item.widget().sizeHint()))
            x += w + spaceX
            lineH = max(lineH, h)
        return y + lineH - rect.y()

class PriFile:
    def __init__(self, num, val):
        self.number = num 
        tokens      = val.strip().split()
        self.type   = tokens[0] if tokens else ""
        self.value  = " ".join(tokens[1:])
        self.valueArr = tokens

class PandasModel(QtCore.QAbstractTableModel):
    def __init__(self, df=pd.DataFrame(), parent=None):
        super().__init__(parent)
        self._df = df

    def rowCount(self, parent=None):
        return self._df.shape[0]

    def columnCount(self, parent=None):
        return self._df.shape[1]

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid() and role == QtCore.Qt.DisplayRole:
            value = self._df.iloc[index.row(), index.column()]
            if isinstance(value, float):
                return f"{value:.2f}"
            return str(value)
        return None

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return None
        if orientation == QtCore.Qt.Horizontal:
            if 0 <= section < self._df.shape[1]:
                return str(self._df.columns[section])
            return ""
        else:
            if 0 <= section < self._df.shape[0]:
                return str(self._df.index[section])
            return ""
        
    def flags(self, index):
        default = super().flags(index)
        return default | QtCore.Qt.ItemIsDragEnabled 

    def supportedDragActions(self):
        return QtCore.Qt.CopyAction

    def mimeData(self, indexes):
        rows = {}
        for idx in indexes:
            rows.setdefault(idx.row(), {})[idx.column()] = str(
                self._df.iloc[idx.row(), idx.column()]
            )
        lines = []
        for r in sorted(rows):
            cols = rows[r]
            line = "\t".join(cols.get(c, "") for c in sorted(cols))
            lines.append(line)
        mime = QtCore.QMimeData()
        mime.setText("\n".join(lines))
        return mime

class FileLoaderThread(QtCore.QThread):
    progressChanged = QtCore.pyqtSignal(int)
    loadingFinished = QtCore.pyqtSignal(list, int)
    
    def __init__(self, filename):
        super(FileLoaderThread, self).__init__()
        self.filename = filename
    
    def run(self):
        try:
            with open(self.filename, "rb") as f:
                byteData = f.read()
            
            detected = chardet.detect(byteData)
            encoding = detected.get("encoding")
            if not encoding:
                encoding = "utf-8"
            
            decodedStr = byteData.decode(encoding, errors="replace")
            
            records = decodedStr.split("~")
            pri_list = []
            maxNum = 0
            total_records = len(records)
            for i, rec in enumerate(records):
                rec = rec.strip()
                if rec:
                    parts = rec.split(" ", 1)
                    if len(parts) < 2:
                        continue
                    priKey = parts[0]
                    if priKey in ("257", "267"):
                        priVal = parts[1].strip()
                    else:
                        priVal = parts[1].replace("\n", " ") 
                    pf = PriFile(priKey, priVal)
                    pri_list.append(pf)
                    if len(pf.valueArr) > maxNum:
                        maxNum = len(pf.valueArr)
                if total_records > 0 and i % 1000 == 0:
                    progress_percent = int((i / total_records) * 100)
                    self.progressChanged.emit(progress_percent)
            self.progressChanged.emit(100)
            self.loadingFinished.emit(pri_list, maxNum)
        except Exception as e:
            import traceback
            logger.error(f"Error loading file: {e}\n{traceback.format_exc()}")


# ──────────────────────────────────────────────────────────────────────────────
# Additional worker thread for downloading map tiles
#
class TileDownloadThread(QtCore.QThread):
    """Download OSM tiles in the background and report progress via signals.

    This thread wraps the ``download_tiles_multi_zoom`` function and exposes
    Qt signals for progress and status updates.  It accepts the bounding
    box and a list of zoom levels at construction.
    """
    progressChanged = QtCore.pyqtSignal(int, int)
    status = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()

    def __init__(self, lat_min: float, lat_max: float, lon_min: float, lon_max: float, zoom_levels: list[int]):
        super().__init__()
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.zoom_levels = zoom_levels

        # Internal flag used to request cancellation.  The download loop in
        # ``download_tiles_multi_zoom`` will periodically call a supplied
        # callback to check this flag and exit early if it returns True.
        self._cancelled: bool = False

    def cancel(self) -> None:
        """Request cancellation of the current download.

        The download loop periodically checks this flag and will exit early
        when set.  This method is thread‑safe and may be called from the
        main GUI thread.
        """
        self._cancelled = True

    def run(self) -> None:
        """Perform the tile download and emit progress/status signals."""
        def callback(current: int, total: int) -> None:
            self.progressChanged.emit(current, total)
        # Provide a cancellation callback to the downloader.  It will be called
        # before fetching each tile.  When True is returned the download stops.
        cancel_cb = lambda: self._cancelled
        try:
            download_tiles_multi_zoom(
                self.lat_min,
                self.lat_max,
                self.lon_min,
                self.lon_max,
                self.zoom_levels,
                callback,
                cancel_callback=cancel_cb,
            )
            # Determine whether the operation was cancelled and emit an appropriate status.
            if self._cancelled:
                self.status.emit("Tile download cancelled")
            else:
                self.status.emit("Tile download completed")
        except Exception as e:
            # Forward any unexpected errors to the status signal.
            self.status.emit(f"Tile download error: {e}")
        finally:
            # Always emit the finished signal to notify the GUI that the thread is done.
            self.finished.emit()

class ExportSettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Settings - Coordinate System")
        self.setModal(True)
        self.resize(360, 120)

        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel("Select default coordinate system (CRS):", self)
        layout.addWidget(label)

        self.crs_combo = QtWidgets.QComboBox(self)
        self.crs_combo.addItem("WGS 84 (EPSG:4326)", "EPSG:4326")
        self.crs_combo.addItem("WGS 84 / Pseudo-Mercator (Web Mercator) (EPSG:3857)", "EPSG:3857")
        self.crs_combo.addItem("WGS 84 / World Mercator (EPSG:3395)", "EPSG:3395")
        self.crs_combo.addItem("WGS 84 / Plate Carrée (EPSG:32662)", "EPSG:32662")
        self.crs_combo.addItem("WGS 84 / World Cylindrical Equal Area (EPSG:54034)", "EPSG:54034")
        self.crs_combo.addItem("NAD83 (EPSG:4269)", "EPSG:4269")
        self.crs_combo.addItem("ETRS89 (EPSG:4258)", "EPSG:4258")
        layout.addWidget(self.crs_combo)

        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal, self
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def selected_crs(self):
        return self.crs_combo.currentData()
    
# 메인 윈도우: 탭 기반 인터페이스 및 데이터 표시
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.tree_option_checkboxes = []
        self.log_option_checkboxes  = []
        self.sw_version = "0.12"
        self.setWindowTitle(f"ForestNAV {self.sw_version} - Advanced forestry Systems Lab, Oregon State University")
        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        self.setWindowIcon(QtGui.QIcon(icon_path))
        self.resize(1400, 900)
        self.openingFile = []
        self.maxNum = 0
        self.df = None
        self.settings = QSettings("OSU_AFLab", "ForestNAV")
        
        self.parser = PRIParser()
        self.visualizer = DataVisualizer()
        
        self.current_file = None
        
        self.tree_data = None
        self.log_data = None
        
        self._create_menu()
        self._init_ui()
        
        self.setAcceptDrops(True)

        self.file_cache: Dict[str, Dict[str, Any]] = {}
        self._preload_threads = {}
        
    def _create_menu(self):
        menubar = self.menuBar()

        fileMenu = menubar.addMenu("File")
        openAct = QtWidgets.QAction("Open", self)
        openAct.triggered.connect(self.open_file_dialog)
        fileMenu.addAction(openAct)

        exportAct = QtWidgets.QAction("Export", self)
        exportAct.triggered.connect(self.export_file)
        fileMenu.addAction(exportAct)

        exitAct = QtWidgets.QAction("Exit", self)
        exitAct.triggered.connect(self.close)
        fileMenu.addAction(exitAct)

        viewMenu = menubar.addMenu("View")
        summaryAct = QtWidgets.QAction("Summary", self)
        summaryAct.triggered.connect(lambda: self.tab_control.setCurrentIndex(0))
        viewMenu.addAction(summaryAct)
        
        rawDataAct = QtWidgets.QAction("Raw Data", self)
        rawDataAct.triggered.connect(lambda: self.tab_control.setCurrentIndex(1))
        viewMenu.addAction(rawDataAct)
        
        treeDataAct = QtWidgets.QAction("Tree Data", self)
        treeDataAct.triggered.connect(lambda: self.tab_control.setCurrentIndex(2))
        viewMenu.addAction(treeDataAct)
        
        logDataAct = QtWidgets.QAction("Log Data", self)
        logDataAct.triggered.connect(lambda: self.tab_control.setCurrentIndex(3))
        viewMenu.addAction(logDataAct)
        
        visualizationAct = QtWidgets.QAction("Visualization", self)
        visualizationAct.triggered.connect(lambda: self.tab_control.setCurrentIndex(4))
        viewMenu.addAction(visualizationAct)
        
        settingMenu = menubar.addMenu("Settings")
        filepathAct = QtWidgets.QAction("File Path", self)
        filepathAct.triggered.connect(self.show_file_path_settings)
        settingMenu.addAction(filepathAct)
        
        exportSettingsAct = QtWidgets.QAction("Export Settings", self)
        exportSettingsAct.triggered.connect(self.show_export_settings)
        settingMenu.addAction(exportSettingsAct)
        
        helpMenu = menubar.addMenu("Help")
        aboutAct = QtWidgets.QAction("About", self)
        aboutAct.triggered.connect(self.show_about)
        helpMenu.addAction(aboutAct)

    def _init_ui(self):
        """Initialize UI"""
        # 메인 프레임
        self.main_frame = QtWidgets.QFrame(self)
        self.setCentralWidget(self.main_frame)
        
        # 메인 레이아웃
        self.main_layout = QtWidgets.QHBoxLayout(self.main_frame)
        
        # 왼쪽 패널 (파일 브라우징 및 컨트롤)
        self.left_panel = QtWidgets.QFrame(self.main_frame)
        self.left_panel.setMaximumWidth(300)
        self.left_panel_layout = QtWidgets.QVBoxLayout(self.left_panel)
        
        # 드래그 앤 드롭 영역
        self.drop_frame = QtWidgets.QGroupBox("Open File")
        self.drop_layout = QtWidgets.QVBoxLayout(self.drop_frame)
        self.drop_label = QtWidgets.QLabel("Drag and drop the file here")
        self.drop_label.setAlignment(QtCore.Qt.AlignCenter)
        self.drop_layout.addWidget(self.drop_label)
        self.left_panel_layout.addWidget(self.drop_frame)
        
        # 파일 열기 버튼
        self.open_button = QtWidgets.QPushButton("Open File")
        self.open_button.clicked.connect(self.open_file_dialog)
        self.left_panel_layout.addWidget(self.open_button)

        # 파일 정보
        self.file_info_frame = QtWidgets.QGroupBox("File Information")
        self.file_info_layout = QtWidgets.QVBoxLayout(self.file_info_frame)
        self.file_info_label = QtWidgets.QLabel("No file loaded")
        self.file_info_label.setWordWrap(True)
        self.file_info_layout.addWidget(self.file_info_label)
        self.left_panel_layout.addWidget(self.file_info_frame)

        self.fileLibrary = []
        self.file_list_widget = QtWidgets.QListWidget()
        self.file_list_widget.setMaximumHeight(150)
        self.file_list_widget.itemClicked.connect(self.on_library_item_clicked)
        self.file_info_layout.addWidget(self.file_list_widget)
        
        # 분석 컨트롤
        self.control_frame = QtWidgets.QGroupBox("Analysis Controls")
        self.control_layout = QtWidgets.QVBoxLayout(self.control_frame)
        
        self.analyze_button = QtWidgets.QPushButton("Start Analysis")
        self.analyze_button.clicked.connect(self.analyze_file)
        self.analyze_button.setEnabled(False)
        self.control_layout.addWidget(self.analyze_button)
        
        self.export_button = QtWidgets.QPushButton("Export Results")
        self.export_button.clicked.connect(self.export_results)
        self.export_button.setEnabled(False)
        self.control_layout.addWidget(self.export_button)
        
        self.left_panel_layout.addWidget(self.control_frame)
        
        # 진행 상황 표시
        self.progress_frame = QtWidgets.QGroupBox("Progress")
        self.progress_layout = QtWidgets.QVBoxLayout(self.progress_frame)
        
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QtWidgets.QLabel("0%")
        self.progress_layout.addWidget(self.progress_label)
        
        self.left_panel_layout.addWidget(self.progress_frame)
        
        # 왼쪽 패널에 스트레치 추가
        self.left_panel_layout.addStretch()
        
        # 오른쪽 패널 (데이터 표시)
        self.right_panel = QtWidgets.QFrame(self.main_frame)
        self.right_panel_layout = QtWidgets.QVBoxLayout(self.right_panel)
        
        # 탭 컨트롤
        self.tab_control = QtWidgets.QTabWidget(self.right_panel)
        
        # Summary 탭
        self.summary_tab = QtWidgets.QWidget()
        self.tab_control.addTab(self.summary_tab, "Summary")
        
        # Raw Data 탭 (새로 추가)
        self.raw_data_tab = QtWidgets.QWidget()
        self.tab_control.addTab(self.raw_data_tab, "Raw Data")
        
        # Tree Data 탭
        self.tree_tab = QtWidgets.QWidget()
        self.tab_control.addTab(self.tree_tab, "Tree Data")
        
        # Log Data 탭
        self.log_tab = QtWidgets.QWidget()
        self.tab_control.addTab(self.log_tab, "Log Data")
        
        # Visualization 탭
        self.visualization_tab = QtWidgets.QWidget()
        self.tab_control.addTab(self.visualization_tab, "Visualization")
        
        self.right_panel_layout.addWidget(self.tab_control)
        
        # 메인 레이아웃에 패널 추가
        self.main_layout.addWidget(self.left_panel)
        self.main_layout.addWidget(self.right_panel)
        
        # 상태 표시줄
        self.statusBar().showMessage("Ready")
        
        # 탭 내용 초기화
        self._init_summary_tab()
        self._init_raw_data_tab()  # 새로 추가된 Raw Data 탭 초기화
        self._init_tree_tab()
        self._init_log_tab()
        self._init_visualization_tab()
        # Initialize a single combined map/GNSS tab instead of separate Map and GNSS tabs.
        # This unified tab will host the map along with GNSS controls and data display.
        self._init_combined_tab()

        # Connect signals for parser after all tabs are created
        self.parser.progressChanged.connect(self._update_progress)
        self.parser.parsingFinished.connect(self._on_parsing_finished)

        # Initially enable only the Raw Data tab and the unified Map tab.
        # When combining the Map and GNSS functionality into a single tab
        # (`self.map_tab`), the legacy `self.gnss_tab` attribute is no longer
        # defined.  Instead, keep the unified map tab enabled alongside the
        # raw data tab and disable all other tabs until a file is loaded.
        raw_idx = self.tab_control.indexOf(self.raw_data_tab)
        map_idx = self.tab_control.indexOf(self.map_tab)
        for i in range(self.tab_control.count()):
            if i != raw_idx and i != map_idx:
                self.tab_control.setTabEnabled(i, False)

    def _init_summary_tab(self):
        """Initialize summary tab"""
        # Summary 정보 표시 영역
        self.summary_layout = QtWidgets.QVBoxLayout(self.summary_tab)
        
        # 파일 요약 정보
        self.file_summary_frame = QtWidgets.QGroupBox("File Summary")
        self.file_summary_layout = QtWidgets.QVBoxLayout(self.file_summary_frame)
        
        self.file_summary_text = QtWidgets.QTextEdit()
        self.file_summary_text.setReadOnly(True)
        self.file_summary_text.setText("No file loaded.")
        self.file_summary_layout.addWidget(self.file_summary_text)
        
        self.summary_layout.addWidget(self.file_summary_frame)
        
        # → Tree Summary
        self.tree_summary_frame = QtWidgets.QGroupBox("Tree Summary")
        self.tree_summary_layout = QtWidgets.QVBoxLayout(self.tree_summary_frame)
        self.tree_options_frame = QtWidgets.QGroupBox("Select Tree Summary Fields")
        self.tree_options_frame.setLayout(FlowLayout(self.tree_options_frame))
        self.tree_summary_layout.addWidget(self.tree_options_frame)
        self.tree_summary_text = QtWidgets.QTextEdit()
        self.tree_summary_text.setReadOnly(True)
        self.tree_summary_layout.addWidget(self.tree_summary_text)
        self.summary_layout.addWidget(self.tree_summary_frame)

        # → Log Summary
        self.log_summary_frame = QtWidgets.QGroupBox("Log Summary")
        self.log_summary_layout = QtWidgets.QVBoxLayout(self.log_summary_frame)
        self.log_options_frame = QtWidgets.QGroupBox("Select Log Summary Fields")
        self.log_options_frame.setLayout(FlowLayout(self.log_options_frame))
        self.log_summary_layout.addWidget(self.log_options_frame)
        self.log_summary_text = QtWidgets.QTextEdit()
        self.log_summary_text.setReadOnly(True)
        self.log_summary_layout.addWidget(self.log_summary_text)
        self.summary_layout.addWidget(self.log_summary_frame)

        # ── 고정된 체크박스 생성 ─────────────────────────────────────────
        # Tree options
        self.tree_option_checkboxes = []
        tree_opts = ["# of trees", "DBH", "Coordinates", "Altitude", "Stem Type", "Species Number"]
        for idx, name in enumerate(tree_opts):
            cb = QtWidgets.QCheckBox(name)
            cb.setChecked(True)
            cb.stateChanged.connect(self._update_summary_tab)
            self.tree_options_frame.layout().addWidget(cb)
            self.tree_option_checkboxes.append(cb)

        # Log options
        self.log_option_checkboxes = []
        log_opts = [
            "# of logs",
            "Diameter ob (Top)", "Diameter ob (Mid)",
            "Diameter ub (Top)", "Diameter ub (Mid)",
            "Length (cm)", "Volume (m3)", "Volume (dl)",
            "Volume (Decimal)"
        ]
        for idx, name in enumerate(log_opts):
            cb = QtWidgets.QCheckBox(name)
            cb.setChecked(True)
            cb.stateChanged.connect(self._update_summary_tab)
            self.log_options_frame.layout().addWidget(cb)
            self.log_option_checkboxes.append(cb)

    def _init_raw_data_tab(self):
        """Initialize raw data tab"""
        # Raw 데이터 표시 영역
        self.raw_data_layout = QtWidgets.QVBoxLayout(self.raw_data_tab)
        
        # Raw 데이터 테이블
        self.raw_data_table = QtWidgets.QTableView()
        self.raw_data_layout.addWidget(self.raw_data_table)

    def _init_tree_tab(self):
        """Initialize tree data tab"""
        # 트리 데이터 표시 영역
        self.tree_layout = QtWidgets.QVBoxLayout(self.tree_tab)
        
        # 트리 데이터 테이블
        self.tree_table = QtWidgets.QTableView()
        self.tree_layout.addWidget(self.tree_table)

    def _init_log_tab(self):
        """Initialize log data tab"""
        self.log_layout = QtWidgets.QVBoxLayout(self.log_tab)

        self.log_table = QtWidgets.QTableView()
        self.log_layout.addWidget(self.log_table)

    def _init_visualization_tab(self):
        self.visualization_layout = QtWidgets.QVBoxLayout(self.visualization_tab)

        # ── control bar ───────────────────────────────────────
        ctrl_bar = QtWidgets.QHBoxLayout()
        ctrl_bar.addWidget(QtWidgets.QLabel("Plot Type:"))
        self.viz_type_combo = QtWidgets.QComboBox()
        self.viz_type_combo.addItems([
            "DBH Distribution",
            "Log Length Distribution",
            "Diameter ob Top Distribution",
            "Diameter ob Mid Distribution",
            "Diameter ub Top Distribution",
            "Diameter ub Mid Distribution",
            "Species Distribution"
        ])
        self.viz_type_combo.currentIndexChanged.connect(self._update_visualization)
        ctrl_bar.addWidget(self.viz_type_combo)

        for label, attr in [("Start", "bin_start"),
                            ("End",   "bin_end"),
                            ("Width", "bin_width")]:
            ctrl_bar.addSpacing(12)
            ctrl_bar.addWidget(QtWidgets.QLabel(f"{label}:"))
            le = QtWidgets.QLineEdit()
            le.setFixedWidth(80)
            le.setPlaceholderText("auto")
            le.editingFinished.connect(self._update_visualization)
            setattr(self, f"{attr}_edit", le)
            ctrl_bar.addWidget(le)

        self.refresh_button = QtWidgets.QPushButton("Refresh")
        self.refresh_button.setToolTip("Refresh plot")
        self.refresh_button.clicked.connect(self._update_visualization)
        ctrl_bar.addWidget(self.refresh_button)

        self.auto_button = QtWidgets.QPushButton("Auto")
        self.auto_button.setToolTip("Start/End/Width → auto")
        self.auto_button.clicked.connect(self._on_auto_clicked)
        ctrl_bar.addWidget(self.auto_button)

        ctrl_bar.addStretch()
        ctrl_frame = QtWidgets.QFrame()
        ctrl_frame.setLayout(ctrl_bar)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        plot_frame = QtWidgets.QFrame()
        plot_layout = QtWidgets.QVBoxLayout(plot_frame)
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        splitter.addWidget(plot_frame)

        self.viz_table = QtWidgets.QTableView()
        splitter.addWidget(self.viz_table)
        splitter.setSizes([400, 150])

        self.visualization_layout.addWidget(ctrl_frame)
        self.visualization_layout.addWidget(splitter)

    def _init_map_tab(self):
        self.map_tab = QtWidgets.QWidget()
        self.tab_control.addTab(self.map_tab, "Map View")

        layout = QtWidgets.QVBoxLayout(self.map_tab)

        self.map_view = QWebEngineView()
        layout.addWidget(self.map_view, stretch=5)

        btn_bar = QtWidgets.QHBoxLayout()
        btn_bar.addStretch()
        btn_wrap = QtWidgets.QWidget()
        btn_wrap.setLayout(btn_bar)
        layout.addWidget(btn_wrap, stretch=0)

        self.map_msg = QtWidgets.QLabel("No coordinate data available.")
        self.map_msg.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.map_msg, 0, QtCore.Qt.AlignCenter)

    def _init_combined_tab(self) -> None:
        """Initialise a unified tab that combines map visualisation and GNSS features.

        This method replaces the separate `_init_map_tab` and `_init_gnss_tab` methods.
        It constructs a single tab that contains the GNSS control bar, GNSS data display,
        a map view for both static datasets and real‑time GNSS updates, and a tile
        downloader panel. A message label is included to indicate when no coordinate
        data are available. The GNSS attributes and timezone settings are also
        initialised here.
        """
        # Create the unified tab and add it to the tab control
        self.map_tab = QtWidgets.QWidget()
        self.tab_control.addTab(self.map_tab, "Map")

        # Root layout for the unified tab
        layout = QtWidgets.QVBoxLayout(self.map_tab)

        # -----------------------------------------------------------------
        # GNSS control group
        ctrl_group = QtWidgets.QGroupBox("GNSS Control")
        ctrl_layout = QtWidgets.QHBoxLayout(ctrl_group)
        # Serial port selection: a drop‑down listing available serial ports
        ctrl_layout.addWidget(QtWidgets.QLabel("Port:"))
        self.gnss_port_combo = QtWidgets.QComboBox()
        # Populate available ports at startup
        self._refresh_serial_ports()
        ctrl_layout.addWidget(self.gnss_port_combo)
        # Button to refresh the list of serial ports
        self.gnss_refresh_ports_btn = QtWidgets.QPushButton("Refresh")
        self.gnss_refresh_ports_btn.setToolTip("Rescan available serial ports")
        self.gnss_refresh_ports_btn.clicked.connect(self._refresh_serial_ports)
        ctrl_layout.addWidget(self.gnss_refresh_ports_btn)
        # Start/Stop toggle button.  This single button toggles GNSS on and off.
        self.gnss_start_btn = QtWidgets.QPushButton("Connect")
        self.gnss_start_btn.clicked.connect(self._toggle_gnss)
        ctrl_layout.addWidget(self.gnss_start_btn)
        # Logging toggle button
        self.gnss_log_btn = QtWidgets.QPushButton("Start Logging")
        self.gnss_log_btn.setEnabled(False)
        self.gnss_log_btn.clicked.connect(self._toggle_gnss_logging)
        ctrl_layout.addWidget(self.gnss_log_btn)
        # Add control group to the unified layout
        layout.addWidget(ctrl_group)

        # -----------------------------------------------------------------
        # GNSS data display group
        data_group = QtWidgets.QGroupBox("GNSS Data")
        data_layout = QtWidgets.QFormLayout(data_group)
        # Create labels for each data field
        self.gnss_lat_label = QtWidgets.QLabel("—")
        self.gnss_lon_label = QtWidgets.QLabel("—")
        self.gnss_speed_label = QtWidgets.QLabel("—")
        self.gnss_bearing_label = QtWidgets.QLabel("—")
        self.gnss_fix_label = QtWidgets.QLabel("—")
        data_layout.addRow("Latitude:", self.gnss_lat_label)
        data_layout.addRow("Longitude:", self.gnss_lon_label)
        data_layout.addRow("Speed (m/s):", self.gnss_speed_label)
        data_layout.addRow("Bearing (deg):", self.gnss_bearing_label)
        data_layout.addRow("Fix quality:", self.gnss_fix_label)
        layout.addWidget(data_group)

        # -----------------------------------------------------------------
        # Map display group
        map_group = QtWidgets.QGroupBox("Map")
        map_layout = QtWidgets.QVBoxLayout(map_group)
        # Use a single QWebEngineView for both static datasets and GNSS updates
        self.gnss_map_view = QWebEngineView()
        map_layout.addWidget(self.gnss_map_view)
        layout.addWidget(map_group)

        # Label shown when no coordinate data are available
        self.map_msg = QtWidgets.QLabel("No coordinate data available.")
        self.map_msg.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.map_msg, 0, QtCore.Qt.AlignCenter)

        # -----------------------------------------------------------------
        # Tile downloader group
        tile_group = QtWidgets.QGroupBox("Tile Downloader")
        tile_layout = QtWidgets.QFormLayout(tile_group)
        # Latitude/Longitude inputs are no longer required; hide them
        self.tile_lat_min_edit = QtWidgets.QLineEdit()
        self.tile_lat_max_edit = QtWidgets.QLineEdit()
        self.tile_lon_min_edit = QtWidgets.QLineEdit()
        self.tile_lon_max_edit = QtWidgets.QLineEdit()
        for w in (self.tile_lat_min_edit, self.tile_lat_max_edit,
                  self.tile_lon_min_edit, self.tile_lon_max_edit):
            w.setVisible(False)
        # Predetermined zoom levels for tile download (0–18 inclusive)
        self.tile_zoom_levels = list(range(0, 19))
        # Progress bar and status label
        self.tile_progress_bar = QtWidgets.QProgressBar()
        self.tile_status_label = QtWidgets.QLabel("")
        tile_layout.addRow("Progress:", self.tile_progress_bar)
        tile_layout.addRow("Status:", self.tile_status_label)
        # Download button
        self.tile_download_btn = QtWidgets.QPushButton("Download Tiles")
        # Connect to the handler that uses the map bounds
        self.tile_download_btn.clicked.connect(self._download_tiles)
        tile_layout.addRow(self.tile_download_btn)
        layout.addWidget(tile_group)

        # -----------------------------------------------------------------
        # Initialise GNSS attributes
        self.gnss_manager = None
        self.gnss_logging = False
        self.gnss_log_file = None  # file handle
        self.gnss_log_writer = None  # csv writer
        # Use Pacific timezone for logging; fallback to UTC if pytz missing
        if pytz:
            try:
                self.gnss_tz = pytz.timezone('US/Pacific')  # type: ignore[attr-defined]
            except Exception:
                self.gnss_tz = None
        else:
            self.gnss_tz = None
        # Tile download thread holder
        self.tile_thread = None
        # Initialise history for GNSS tracking
        self.gnss_history = []

        # -----------------------------------------------------------------
        # Prepare and load the HTML template for the unified map.  This will
        # render a Leaflet map and provide JS functions for updating the marker
        # and adding dataset layers.  It also defines variables 'online' and
        # 'offline' tile layers so they can be toggled after downloading tiles.
        self._prepare_gnss_map_html()
        # Ensure local file and remote URLs are permitted in the embedded view
        settings = self.gnss_map_view.settings()
        try:
            # PyQt6 style: attributes nested under WebAttribute
            settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
            settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        except Exception:
            try:
                # PyQt5 style: attributes defined directly on QWebEngineSettings
                settings.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
                settings.setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
            except Exception:
                pass
        # Load the map
        self.gnss_map_view.load(QtCore.QUrl.fromLocalFile(self.gnss_map_html_path))

    def _update_map_tab(self):
        """Update the unified map tab with currently loaded datasets.

        This implementation forwards dataset coordinates into the embedded Leaflet map
        instead of regenerating a static Folium map.  Datasets are passed to the
        JavaScript side via the `addDataset` function defined in the HTML
        template.  Existing dataset layers are cleared before new layers are
        added.  The map is centred on the mean of all coordinates.
        """
        # Gather datasets from the loaded files.  Each dataset consists of a
        # label (file base name), a DataFrame of coordinates, and the full
        # tree DataFrame.  Only files with both Latitude and Longitude columns
        # and non‑empty coordinates are included.
        datasets = []
        for fp in getattr(self, 'fileLibrary', []):
            cache = self.file_cache.get(fp)
            if not cache or 'tree_data' not in cache:
                continue
            tdf = cache['tree_data']
            if {"Latitude", "Longitude"} - set(tdf.columns):
                continue
            df_coords = (tdf[["Latitude", "Longitude"]]
                         .dropna().astype(float)
                         .reset_index(drop=False)
                         .rename(columns={"index": "TreeID"}))
            if df_coords.empty:
                continue
            datasets.append((os.path.basename(fp), df_coords, tdf))

        # If no datasets are available, show a message and return without
        # modifying the map.  This leaves any existing dataset layers intact.
        if not datasets:
            if hasattr(self, 'map_msg'):
                self.map_msg.show()
            return

        # Hide the message label since data will be plotted
        if hasattr(self, 'map_msg'):
            self.map_msg.hide()

        # Compute mean latitude and longitude across all datasets for centering
        lat_mean = float(np.mean([d[1]["Latitude"].mean() for d in datasets]))
        lon_mean = float(np.mean([d[1]["Longitude"].mean() for d in datasets]))

        # Colour palette and cycling iterator for dataset layers
        palette = [
            "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
            "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
            "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000",
            "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080"
        ]
        color_cycle = itertools.cycle(palette)

        # Clear existing dataset layers on the JavaScript side.  The clearDatasets()
        # function is defined in the HTML template.  If it is not present, this
        # call has no effect.
        clear_js = "if (typeof clearDatasets === 'function') { clearDatasets(); }"
        try:
            self.gnss_map_view.page().runJavaScript(clear_js)
        except Exception:
            pass

        # Add each dataset as a separate overlay layer.  For each dataset, build
        # a list of [lat, lon] pairs and then invoke addDataset(name, points, color).
        for label, df_coords, tdf in datasets:
            # Use a detailed points list that includes info for popups/tooltips.
            # Each element is [lat, lon, tooltip, popup].
            color = next(color_cycle)
            points = []
            for r in df_coords.itertuples():
                try:
                    lat = float(r.Latitude)
                    lon = float(r.Longitude)
                except Exception:
                    continue
                # Tree index from the DataFrame row; use to lookup full tree info.
                tree_idx = int(r.TreeID)
                info_row = None
                try:
                    info_row = tdf.loc[tree_idx]
                except Exception:
                    info_row = None
                # Tooltip uses the stem number if available, otherwise the tree index.
                tooltip = None
                if info_row is not None:
                    try:
                        stem = info_row.get('Tree ID (Stem Number)', None)
                        if stem is not None and pd.notna(stem):
                            tooltip = f"Tree ID (Stem Number): {stem}"
                    except Exception:
                        pass
                if tooltip is None:
                    tooltip = f"Tree {tree_idx}"
                # Build an HTML popup string with all available attributes.
                popup = None
                if info_row is not None:
                    try:
                        popup_lines = []
                        for k, v in info_row.items():
                            try:
                                if pd.notna(v):
                                    popup_lines.append(f"<b>{k}</b>: {v}")
                            except Exception:
                                continue
                        popup = "<br>".join(popup_lines)
                    except Exception:
                        popup = None
                points.append([lat, lon, tooltip, popup])
            # Serialise points to JSON for injection into JS.
            try:
                import json as _json
                points_json = _json.dumps(points)
            except Exception:
                points_json = '[]'
            js = (
                f"if (typeof addDataset === 'function') "
                f"{{ addDataset('{label}', {points_json}, '{color}'); }}"
            )
            try:
                self.gnss_map_view.page().runJavaScript(js)
            except Exception:
                pass

        # Centre the map on the computed mean location with a reasonable zoom level.
        js_center = (
            f"if (typeof map !== 'undefined' && map.setView) "
            f"{{ map.setView([{lat_mean}, {lon_mean}], 16); }}"
        )
        try:
            self.gnss_map_view.page().runJavaScript(js_center)
        except Exception:
            pass

        # For export functionality, retain the last computed tree_data
        self._map_df_for_export = getattr(self, 'tree_data', None)

        # At this point the unified map has been updated; return early so that
        # the original Folium-based implementation is skipped entirely.
        return

        # ── 1) 데이터 모으기 ─────────────────────────────
        datasets = []
        for fp in self.fileLibrary:
            cache = self.file_cache.get(fp)
            if not cache or "tree_data" not in cache:
                continue
            tdf = cache["tree_data"]
            if {"Latitude", "Longitude"} - set(tdf.columns):
                continue

            df_coords = (tdf[["Latitude", "Longitude"]]
                        .dropna().astype(float)
                        .reset_index(drop=False)
                        .rename(columns={"index": "TreeID"}))
            if df_coords.empty:
                continue
            datasets.append((os.path.basename(fp), df_coords, tdf))

        if not datasets:
            return

        # ── 2) 지도 베이스 생성 (모든 좌표 평균) ─────────────
        lat_mean = np.mean([d[1]["Latitude"].mean() for d in datasets])
        lon_mean = np.mean([d[1]["Longitude"].mean() for d in datasets])
        fmap = folium.Map(location=[lat_mean, lon_mean],
                        zoom_start=16, tiles="Esri.WorldImagery")

        # ── 3) 컬러 팔레트 준비 ───────────────────────────
        palette = [
            "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
            "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
            "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000",
            "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080"
        ]
        color_cycle = itertools.cycle(palette)

        # ── 4) 각 데이터셋을 개별 레이어로 추가 ───────────
        for label, df_coords, tdf in datasets:
            color = next(color_cycle)
            feat_grp = folium.FeatureGroup(name=label)

            for _, row in df_coords.iterrows():
                tree_idx = int(row.TreeID)
                # popup_html = "<br>".join(
                #     f"<b>{k}</b>: {v}"
                #     for k, v in tdf.loc[tree_idx].items() if pd.notna(v)
                # )
                # folium.CircleMarker(
                #     location=(row.Latitude, row.Longitude),
                #     radius=4, color=color, fill=True,
                #     fill_color=color, fill_opacity=0.8,
                #     popup=folium.Popup(popup_html, max_width=300),
                #     tooltip=f"{label} – Tree {tree_idx}"
                # ).add_to(feat_grp)

                stem_number = tdf.loc[tree_idx, 'Tree ID (Stem Number)']
                tooltip_text = f"Tree ID (Stem Number): {stem_number}"

                popup_html = "<br>".join(
                    f"<b>{k}</b>: {v}"
                    for k, v in tdf.loc[tree_idx].items() if pd.notna(v)
                )
                folium.CircleMarker(
                    location=(row.Latitude, row.Longitude),
                    radius=3, color=color, fill=True,
                    fill_color=color, fill_opacity=0.8,
                    tooltip=tooltip_text,
                    popup=folium.Popup(popup_html, max_width=300),
                ).add_to(feat_grp)

            feat_grp.add_to(fmap)

        # ── 5) 레이어 컨트롤 & 렌더링 ─────────────────────
        folium.LayerControl().add_to(fmap)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        fmap.save(tmp.name)
        self.map_view.load(QtCore.QUrl.fromLocalFile(tmp.name))

        # 화면 전환
        self.map_view.show()
        self.map_msg.hide()

        # 내보내기용 백업 (현재 파일만 우선 저장)
        self._map_df_for_export = self.tree_data

    # ------------------------------------------------------------------
    # GNSS tab helpers and callbacks
    # ------------------------------------------------------------------
    def _gnss_start(self) -> None:
        """Start the GNSS reader using the selected serial port."""
        # Retrieve the selected serial port from the combo box.  The combo
        # is editable, so the user may also type a custom port.  If no
        # value is selected, warn the user.
        if hasattr(self, 'gnss_port_combo'):
            port = self.gnss_port_combo.currentText().strip()
        else:
            # Fallback to legacy line edit if combo is not present.  Guard
            # against None since the legacy attribute may no longer exist.
            port_edit = getattr(self, 'gnss_port_edit', None)
            if port_edit is not None:
                port = port_edit.text().strip()
            else:
                port = ''
        if not port:
            QtWidgets.QMessageBox.warning(self, "GNSS", "Please select or enter a serial port to start GNSS.")
            return
        # Stop any existing manager
        if self.gnss_manager is not None:
            self.gnss_manager.stop()
            self.gnss_manager = None
        # Create and configure the manager
        self.gnss_manager = GNSSManager(port)
        self.gnss_manager.newDataAvailable.connect(self._on_new_gnss_data)
        self.gnss_manager.status.connect(self._on_gnss_status)
        self.gnss_manager.start()
        # Update UI: change the toggle button text to indicate GNSS is running and enable logging
        self.gnss_start_btn.setText("Disconnect")
        self.gnss_start_btn.setEnabled(True)
        self.gnss_log_btn.setEnabled(True)
        self.statusBar().showMessage(f"GNSS started on {port}")

    def _gnss_stop(self) -> None:
        """Stop the GNSS reader and disable logging."""
        if self.gnss_manager is not None:
            self.gnss_manager.stop()
            self.gnss_manager = None
        # If logging is active, stop it
        if self.gnss_logging:
            self._toggle_gnss_logging()
        # Reset UI: revert the toggle button text and disable logging
        self.gnss_start_btn.setText("Connect")
        self.gnss_start_btn.setEnabled(True)
        self.gnss_log_btn.setEnabled(False)
        self.statusBar().showMessage("GNSS stopped")

    def _toggle_gnss(self) -> None:
        """Toggle the GNSS reader on or off using a single button.

        When no GNSS manager exists, this calls :meth:`_gnss_start` to
        initialise and start a new manager.  When a manager is present,
        this stops it via :meth:`_gnss_stop`.  The button text and
        enabled state are updated by those methods.
        """
        if self.gnss_manager is None:
            self._gnss_start()
        else:
            self._gnss_stop()

    def _on_gnss_status(self, msg: str) -> None:
        """Display status messages from the GNSS manager."""
        # Show GNSS status in the status bar and GNSS tab status label
        self.statusBar().showMessage(msg)

    def _on_new_gnss_data(self) -> None:
        """Handle new GNSS data: update UI, map, and optionally log."""
        if self.gnss_manager is None:
            return
        lat, lon, speed, bearing, fix = self.gnss_manager.get_latest_data()
        # Update textual labels
        self.gnss_lat_label.setText(f"{lat:.6f}" if lat is not None else "—")
        self.gnss_lon_label.setText(f"{lon:.6f}" if lon is not None else "—")
        self.gnss_speed_label.setText(f"{speed:.2f}" if speed is not None else "—")
        self.gnss_bearing_label.setText(f"{bearing:.2f}" if bearing is not None else "—")
        self.gnss_fix_label.setText(str(fix) if fix is not None else "—")
        # Update map if lat/lon available
        if lat is not None and lon is not None:
            self._update_gnss_map(lat, lon)
        # If logging is enabled, append a row
        if self.gnss_logging and self.gnss_log_writer is not None:
            # Determine timestamp in configured timezone
            if self.gnss_tz:
                dt = datetime.now(self.gnss_tz)
            else:
                dt = datetime.utcnow()
            time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            try:
                self.gnss_log_writer.writerow([time_str, lat, lon, speed, bearing, fix])
                self.gnss_log_file.flush()
            except Exception:
                pass

    def _toggle_gnss_logging(self) -> None:
        """Start or stop logging of GNSS data to a CSV file."""
        if not self.gnss_logging:
            # Begin logging: prompt user to select a CSV file location
            if self.gnss_manager is None:
                QtWidgets.QMessageBox.information(self, "GNSS Logging", "Please start GNSS before logging.")
                return
            # Suggest a default filename incorporating the current timestamp
            if self.gnss_tz:
                dt = datetime.now(self.gnss_tz)
            else:
                dt = datetime.utcnow()
            default_name = dt.strftime("gnss_log_%Y%m%d_%H%M%S.csv")
            # Open a file save dialog
            file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Select GNSS Log File",
                default_name,
                "CSV Files (*.csv)"
            )
            # If the user cancels the dialog, abort starting logging
            if not file_path:
                return
            try:
                # Open the selected file for writing and create a CSV writer
                self.gnss_log_file = open(file_path, "w", newline="")
                self.gnss_log_writer = csv.writer(self.gnss_log_file)
                # Write header row
                self.gnss_log_writer.writerow(["Time", "Latitude", "Longitude", "Speed_mps", "Bearing_deg", "Fix_Quality"])
                # Update state and UI
                self.gnss_logging = True
                self.gnss_log_btn.setText("Stop Logging")
                self.statusBar().showMessage(f"Logging GNSS data to {file_path}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Logging Error", f"Unable to open log file: {e}")
        else:
            # Stop logging
            try:
                if self.gnss_log_file:
                    self.gnss_log_file.close()
            except Exception:
                pass
            self.gnss_log_file = None
            self.gnss_log_writer = None
            self.gnss_logging = False
            self.gnss_log_btn.setText("Start Logging")
            self.statusBar().showMessage("GNSS logging stopped")

    def _download_tiles(self) -> None:
        """Retrieve the visible map bounds and download tiles for the specified zoom levels.

        Rather than requiring the user to manually enter latitude and longitude
        bounds, this method queries the embedded Leaflet map for its current
        bounding box.  The user only needs to specify one or more zoom levels
        as a comma‑separated list.  Once the bounds are retrieved from the
        JavaScript context, a background thread is started to download all
        tiles covering the area for the requested zooms.
        """
        # Use predetermined zoom levels defined in _init_gnss_tab.  The user
        # no longer enters zoom levels; if none are defined, abort.
        zoom_levels: list[int] = getattr(self, 'tile_zoom_levels', [])
        if not zoom_levels:
            QtWidgets.QMessageBox.warning(self, "Tile Downloader", "No zoom levels defined for download.")
            return
        # Disable the download button and reset progress
        self.tile_download_btn.setEnabled(False)
        self.tile_progress_bar.setValue(0)
        self.tile_status_label.setText("Retrieving map bounds…")
        # JavaScript to compute map bounds: returns "south,north,west,east"
        js = """
            (function() {
                if (typeof getBounds === 'function') {
                    return getBounds();
                } else {
                    var b = map.getBounds();
                    return [b.getSouth(), b.getNorth(), b.getWest(), b.getEast()].join(',');
                }
            })();
        """
        # Capture zoom_levels for the callback
        def bounds_callback(bbox_str: str) -> None:
            self._on_tiles_bounds_received(bbox_str, zoom_levels)
        # Invoke the JavaScript on the map view
        try:
            self.gnss_map_view.page().runJavaScript(js, bounds_callback)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Tile Downloader", f"Failed to retrieve map bounds: {e}")
            # Re‑enable the button since we didn't start downloading
            self.tile_download_btn.setEnabled(True)

    def _on_tiles_progress(self, current: int, total: int) -> None:
        """Update progress bar based on tile download progress."""
        if total > 0:
            self.tile_progress_bar.setMaximum(total)
            self.tile_progress_bar.setValue(current)

    def _on_tiles_status(self, msg: str) -> None:
        """Update the status label for tile download."""
        self.tile_status_label.setText(msg)

    def _on_tiles_finished(self) -> None:
        """Handle completion of tile downloads.

        Re‑enable the tile download button, clear the thread reference and
        attempt to swap the map layers so that downloaded offline tiles
        are used instead of the online OpenStreetMap layer.  This is
        accomplished by reloading the map view and executing a small
        JavaScript snippet that removes the online layer and adds the
        offline layer.
        """
        self.tile_download_btn.setEnabled(True)
        # Clear thread reference
        self.tile_thread = None
        # Disable the cancel button now that the thread is done
        try:
            if hasattr(self, 'tile_cancel_btn'):
                self.tile_cancel_btn.setEnabled(False)
        except Exception:
            pass
        # Reload the view to ensure any new tiles are picked up
        try:
            self.gnss_map_view.reload()
            # Toggle layers: remove online if present and add offline if absent
            js = """
                (function() {
                    if (typeof online !== 'undefined' && map.hasLayer(online)) {
                        map.removeLayer(online);
                    }
                    if (typeof offline !== 'undefined' && !map.hasLayer(offline)) {
                        map.addLayer(offline);
                    }
                })();
            """
            # Delay execution slightly to allow map to reload
            QtCore.QTimer.singleShot(500, lambda: self.gnss_map_view.page().runJavaScript(js))
        except Exception as e:
            logger.error(f"Error toggling offline map layer: {e}")

    def _cancel_tile_download(self) -> None:
        """Handle user request to cancel the tile download.

        If a download thread is running, mark it as cancelled.  This will cause
        the thread to exit early on its next iteration.  Immediately re‑enable
        the download button and disable the cancel button.  Update the status
        label to inform the user that cancellation has been requested.
        """
        try:
            if getattr(self, 'tile_thread', None):
                # Request cancellation.  The thread periodically checks this flag.
                try:
                    self.tile_thread.cancel()
                except Exception:
                    pass
                # Provide immediate UI feedback
                self.tile_status_label.setText("Cancelling…")
        except Exception:
            pass
        # Re‑enable the download button so the user can start a new download if desired.
        try:
            self.tile_download_btn.setEnabled(True)
        except Exception:
            pass
        # Disable the cancel button to prevent duplicate cancellation attempts.
        try:
            if hasattr(self, 'tile_cancel_btn'):
                self.tile_cancel_btn.setEnabled(False)
        except Exception:
            pass

    def _on_tiles_bounds_received(self, bbox_str: str, zoom_levels: list[int]) -> None:
        """Callback invoked with the map bounds string and zoom levels.

        This method parses the bounding box returned from the JavaScript
        function, validates numeric values, starts the background tile
        download thread and connects its signals to update the UI.

        Parameters
        ----------
        bbox_str : str
            A comma‑separated string containing south, north, west, east.
        zoom_levels : list[int]
            List of integer zoom levels to download.
        """
        try:
            south, north, west, east = [float(x) for x in bbox_str.split(',')]
        except Exception:
            QtWidgets.QMessageBox.critical(self, "Tile Downloader", "Failed to parse map bounds from viewer.")
            self.tile_download_btn.setEnabled(True)
            return
        # Normalise values so that lat_min <= lat_max and lon_min <= lon_max
        lat_min, lat_max = sorted([south, north])
        lon_min, lon_max = sorted([west, east])
        # Update status and reset progress bar
        self.tile_progress_bar.setValue(0)
        self.tile_status_label.setText("Starting download…")
        # Start the download thread
        self.tile_thread = TileDownloadThread(lat_min, lat_max, lon_min, lon_max, zoom_levels)
        self.tile_thread.progressChanged.connect(self._on_tiles_progress)
        self.tile_thread.status.connect(self._on_tiles_status)
        self.tile_thread.finished.connect(self._on_tiles_finished)
        self.tile_thread.start()
        # Enable the cancel button since a download is now running
        try:
            if hasattr(self, 'tile_cancel_btn'):
                self.tile_cancel_btn.setEnabled(True)
        except Exception:
            pass

    def _update_gnss_map(self, lat: Optional[float], lon: Optional[float]) -> None:
        """Update the GNSS map with the latest position.

        Instead of regenerating the map for every update, this method
        communicates with the embedded Leaflet map via JavaScript.  It
        calls the ``updatePosition`` function defined in the HTML to
        move the main marker, center the view and add a new circle
        marker for the current fix.  A history of fixes is maintained
        so that each point receives a unique index and timestamp.  If
        no position is provided, the map is not updated.
        """
        # Do nothing when no valid coordinates are provided
        if lat is None or lon is None:
            return
        try:
            # Always update the main marker to the latest position.
            js_marker = f"updateMarker({lat}, {lon});"
            self.gnss_map_view.page().runJavaScript(js_marker)
            # Only record and draw history points when logging is enabled
            if self.gnss_logging:
                # Compose tooltip information including an incrementing ID, timestamp and coordinates
                idx = len(self.gnss_history) + 1
                # Determine timestamp in configured timezone (Pacific by default) or UTC
                if self.gnss_tz:
                    dt = datetime.now(self.gnss_tz)
                else:
                    dt = datetime.utcnow()
                ts_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                info = f"#{idx}<br>Time: {ts_str}<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}"
                info_json = json.dumps(info)
                js_point = f"updatePosition({lat}, {lon}, {info_json});"
                self.gnss_map_view.page().runJavaScript(js_point)
                # Append to Python-side history for indexing
                self.gnss_history.append((lat, lon, ts_str))
        except Exception as e:
            logger.error(f"Failed to update GNSS map: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # GNSS Tab
    #
    def _init_gnss_tab(self):
        """Initialize the GNSS tab and set up widgets for real‑time GNSS."""
        # Create tab and add it to the tab control
        self.gnss_tab = QtWidgets.QWidget()
        self.tab_control.addTab(self.gnss_tab, "GNSS")

        # Root layout for GNSS tab
        layout = QtWidgets.QVBoxLayout(self.gnss_tab)

        # -----------------------------------------------------------------
        # GNSS control group
        ctrl_group = QtWidgets.QGroupBox("GNSS Control")
        ctrl_layout = QtWidgets.QHBoxLayout(ctrl_group)

        # Serial port selection
        # Replace the manual text entry with a drop‑down listing available serial ports.
        ctrl_layout.addWidget(QtWidgets.QLabel("Port:"))
        self.gnss_port_combo = QtWidgets.QComboBox()
        # Populate available ports at startup
        self._refresh_serial_ports()
        # Allow the user to refresh the list by clicking the drop‑down arrow
        ctrl_layout.addWidget(self.gnss_port_combo)
        # Maintain a legacy attribute for backward compatibility; unused now
        self.gnss_port_edit = None
        # Optional: add a button to refresh ports manually
        self.gnss_refresh_ports_btn = QtWidgets.QPushButton("Refresh")
        self.gnss_refresh_ports_btn.setToolTip("Rescan available serial ports")
        self.gnss_refresh_ports_btn.clicked.connect(self._refresh_serial_ports)
        ctrl_layout.addWidget(self.gnss_refresh_ports_btn)

        # Start/Stop toggle button.  This single button toggles GNSS on and off.
        self.gnss_start_btn = QtWidgets.QPushButton("Connect")
        # Connect to the toggle handler rather than separate start/stop slots
        self.gnss_start_btn.clicked.connect(self._toggle_gnss)
        ctrl_layout.addWidget(self.gnss_start_btn)

        # Logging toggle button
        self.gnss_log_btn = QtWidgets.QPushButton("Start Logging")
        self.gnss_log_btn.setEnabled(False)
        self.gnss_log_btn.clicked.connect(self._toggle_gnss_logging)
        ctrl_layout.addWidget(self.gnss_log_btn)

        # Add control group to layout
        layout.addWidget(ctrl_group)

        # -----------------------------------------------------------------
        # GNSS data display group
        data_group = QtWidgets.QGroupBox("GNSS Data")
        data_layout = QtWidgets.QFormLayout(data_group)

        # Create labels for each data field
        self.gnss_lat_label = QtWidgets.QLabel("—")
        self.gnss_lon_label = QtWidgets.QLabel("—")
        self.gnss_speed_label = QtWidgets.QLabel("—")
        self.gnss_bearing_label = QtWidgets.QLabel("—")
        self.gnss_fix_label = QtWidgets.QLabel("—")

        data_layout.addRow("Latitude:", self.gnss_lat_label)
        data_layout.addRow("Longitude:", self.gnss_lon_label)
        data_layout.addRow("Speed (m/s):", self.gnss_speed_label)
        data_layout.addRow("Bearing (deg):", self.gnss_bearing_label)
        data_layout.addRow("Fix quality:", self.gnss_fix_label)

        layout.addWidget(data_group)

        # -----------------------------------------------------------------
        # GNSS map display group
        map_group = QtWidgets.QGroupBox("GNSS Map")
        map_layout = QtWidgets.QVBoxLayout(map_group)
        # QWebEngineView to show the current position on a map
        self.gnss_map_view = QWebEngineView()
        map_layout.addWidget(self.gnss_map_view)
        layout.addWidget(map_group)

        # -----------------------------------------------------------------
        # Tile downloader group
        tile_group = QtWidgets.QGroupBox("Tile Downloader")
        tile_layout = QtWidgets.QFormLayout(tile_group)
        # Latitude/Longitude inputs are no longer required; hide them but keep the zoom level input
        self.tile_lat_min_edit = QtWidgets.QLineEdit()
        self.tile_lat_max_edit = QtWidgets.QLineEdit()
        self.tile_lon_min_edit = QtWidgets.QLineEdit()
        self.tile_lon_max_edit = QtWidgets.QLineEdit()
        # Hide these fields since the bounding box will be taken from the map
        for w in (self.tile_lat_min_edit, self.tile_lat_max_edit, self.tile_lon_min_edit, self.tile_lon_max_edit):
            w.setVisible(False)
        # Predetermined zoom levels for tile download.  The user no longer
        # selects zoom levels manually; instead we download a reasonable
        # range of zooms (0–18 inclusive) as in the original GNSS viewer.
        self.tile_zoom_levels = list(range(0, 19))
        # Progress bar and status label
        self.tile_progress_bar = QtWidgets.QProgressBar()
        self.tile_status_label = QtWidgets.QLabel("")
        tile_layout.addRow("Progress:", self.tile_progress_bar)
        tile_layout.addRow("Status:", self.tile_status_label)
        # Download button
        self.tile_download_btn = QtWidgets.QPushButton("Download Tiles")
        # Cancel button for aborting an in‑progress download
        self.tile_cancel_btn = QtWidgets.QPushButton("Cancel")
        # Disable the cancel button until a download is started
        self.tile_cancel_btn.setEnabled(False)
        # Connect to our new download handler that uses the map bounds
        self.tile_download_btn.clicked.connect(self._download_tiles)
        # Connect the cancel button to the cancel handler
        self.tile_cancel_btn.clicked.connect(self._cancel_tile_download)
        # Place the two buttons side by side using a horizontal layout.  This ensures
        # they occupy a single row in the form layout.
        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.addWidget(self.tile_download_btn)
        buttons_layout.addWidget(self.tile_cancel_btn)
        tile_layout.addRow(buttons_layout)
        layout.addWidget(tile_group)
        # Adjust size policies and stretch factors so that the map occupies the majority
        # of available vertical space.  The control, data and tile downloader groups
        # should not expand when the window is resized; only the map group should.
        ctrl_group.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        data_group.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        map_group.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        tile_group.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        # Assign stretch factors: 0 for non‑expanding panels, 1 for the map panel.
        # The widgets are added in order: ctrl_group (index 0), data_group (1),
        # map_group (2), tile_group (3).
        layout.setStretch(0, 0)
        layout.setStretch(1, 0)
        layout.setStretch(2, 1)
        layout.setStretch(3, 0)

        # Initialise GNSS attributes
        self.gnss_manager = None
        self.gnss_logging = False
        self.gnss_log_file = None  # file handle
        self.gnss_log_writer = None  # csv writer
        # Use Pacific timezone for logging; fallback to UTC if pytz missing
        # Determine timezone for logging: if pytz is available, use US/Pacific; otherwise None
        if pytz:
            try:
                self.gnss_tz = pytz.timezone('US/Pacific')  # type: ignore[attr-defined]
            except Exception:
                self.gnss_tz = None
        else:
            self.gnss_tz = None

        # Tile download thread holder
        self.tile_thread = None

        # Initialise history for GNSS tracking
        self.gnss_history = []

        # Prepare and load the HTML template for the GNSS map.  This will render
        # a Leaflet map and provide JS functions for updating the marker and
        # retrieving the current map bounds.  It also defines variables
        # 'online' and 'offline' tile layers so they can be toggled after
        # downloading tiles.
        self._prepare_gnss_map_html()
        # Ensure local file and remote URLs are permitted in the embedded view
        settings = self.gnss_map_view.settings()
        # Enable access to both remote and local content from file:// URLs.  The
        # attribute names differ between PyQt5 and PyQt6; attempt both.
        try:
            # PyQt6 style: attributes nested under WebAttribute
            settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
            settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        except Exception:
            try:
                # PyQt5 style: attributes defined directly on QWebEngineSettings
                settings.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
                settings.setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
            except Exception:
                pass
        # Load the map
        self.gnss_map_view.load(QtCore.QUrl.fromLocalFile(self.gnss_map_html_path))

    # ------------------------------------------------------------------
    # GNSS helper methods
    # ------------------------------------------------------------------
    def _refresh_serial_ports(self) -> None:
        """Populate the serial port combo box with available ports.

        This uses the serial.tools.list_ports API if available.  If
        pyserial is not installed, the combo will remain empty.
        """
        if not hasattr(self, 'gnss_port_combo'):
            return
        self.gnss_port_combo.clear()
        if serial:
            try:
                ports = list(serial.tools.list_ports.comports())
            except Exception:
                ports = []
            for p in ports:
                # Add the device identifier (e.g. COM5 or /dev/ttyUSB0)
                self.gnss_port_combo.addItem(p.device)
        # Always allow manual entry by making the combo editable
        self.gnss_port_combo.setEditable(True)

    def _prepare_gnss_map_html(self) -> None:
        """Generate or update the HTML used to render the GNSS map.

        The HTML includes a Leaflet map with both online and offline tile
        layers defined.  It also defines a marker for the current GNSS
        position, a history array to store past points, and functions
        updatePosition(lat, lon, info) to move the marker and add
        history points, and getBounds() to return the current map
        bounds.  The file is written to a persistent location in the
        application directory so that relative paths to the tiles
        directory remain valid.
        """
        # Determine where to write the HTML.  Put it alongside this script
        base_dir = os.path.abspath(os.path.dirname(__file__))
        html_path = os.path.join(base_dir, 'gnss_map.html')
        # Starting position placeholder (center of world)
        lat0 = 0.0
        lon0 = 0.0
        # Compute relative path from HTML to tiles directory
        tiles_path = 'tiles'
        # Build the HTML content
        html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>GNSS Map</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
  <style>
    html, body, #map {{ height: 100%; margin: 0; padding: 0; }}
    /* Style for the auto-center control when active */
    .active-center {{
      background-color: #0078A8;
      color: #fff;
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <script>
    // Initialize map and layers
    var map = L.map('map').setView([{lat0}, {lon0}], 2);
    // Online satellite imagery layer (Esri World Imagery)
    // Use Esri's World Imagery service as the default base map so users see satellite
    // imagery instead of the standard OpenStreetMap tiles.  The curly braces are
    // doubled to escape them in the Python f-string.
    var online = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{ maxZoom: 19, attribution: 'Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics' }}).addTo(map);
    // Offline tile layer (not added by default)
    var offline = L.tileLayer('{tiles_path}/{{z}}/{{x}}/{{y}}.png', {{ maxZoom: 19, attribution: 'Offline tiles' }});
    // Current position marker
    var marker = L.marker([{lat0}, {lon0}]).addTo(map);
    // History of visited points
    var history = [];
    // Whether the map should automatically center on the current position.
    // By default this is false; the user can toggle it via a custom control.
    var autoCenter = false;
    // ---------------------------------------------------------------------
    // Dataset layer storage and control.  The unified map needs to display
    // static datasets loaded from files alongside the real‑time GNSS marker.
    // Each dataset is stored in the `datasetLayers` object so that it can be
    // removed or toggled via the Leaflet layer control.  A base layer
    // dictionary defines the online and offline tile layers.
    var datasetLayers = {{}};
    var baseLayers = {{ 'Satellite': online, 'Offline': offline }};
    // The controlLayers object will manage overlays for datasets.  The
    // `collapsed: false` option keeps the control expanded by default.
    var controlLayers = L.control.layers(baseLayers, {{}}, {{ collapsed: false }}).addTo(map);
    /**
     * Remove all previously added dataset layers from the map and the control.
     * Called from Python before new datasets are added.
     */
    function clearDatasets() {{
      for (var name in datasetLayers) {{
        var layer = datasetLayers[name];
        if (map.hasLayer(layer)) {{
          map.removeLayer(layer);
        }}
        try {{
          controlLayers.removeLayer(layer);
        }} catch (e) {{
          // ignore missing layer
        }}
      }}
      datasetLayers = {{}};
    }}
    /**
     * Add a new dataset to the map.
     * @param {{string}} name  Dataset name to appear in the layer control
     * @param {{Array}} points Array of [lat, lon, tooltip, popup] entries
     * @param {{string}} color Hex colour code for markers
     */
    function addDataset(name, points, color) {{
      // Use a feature group so that we can compute bounds when toggling layers.
      var layer = L.featureGroup();
      for (var i = 0; i < points.length; i++) {{
        var pt = points[i];
        var lat = pt[0], lon = pt[1];
        var tooltip = (pt.length > 2 ? pt[2] : null);
        var popup = (pt.length > 3 ? pt[3] : null);
        var circle = L.circleMarker([lat, lon], {{ radius: 3, color: color, fillColor: color, fillOpacity: 0.8 }});
        if (tooltip) {{
          circle.bindTooltip(tooltip);
        }}
        if (popup) {{
          circle.bindPopup(popup);
        }}
        circle.addTo(layer);
      }}
      layer.addTo(map);
      datasetLayers[name] = layer;
      controlLayers.addOverlay(layer, name);
      // When the user toggles this overlay via the layer control, recenter the map
      // on the bounds of the dataset.  Use the overlayadd event to detect when
      // the layer is added to the map.
      map.on('overlayadd', function(e) {{
        if (e.layer === layer) {{
          var b = layer.getBounds();
          if (b && b.isValid && b.isValid()) {{
            map.fitBounds(b);
          }}
        }}
      }});
    }}
    /**
     * Helper to move the main marker to a new location and optionally recenter the map.
     * @param {{number}} lat Latitude in decimal degrees
     * @param {{number}} lon Longitude in decimal degrees
     */
    function updateMarker(lat, lon) {{
      // Move the main marker
      marker.setLatLng([lat, lon]);
      // Recenter the map only if autoCenter is enabled
      if (autoCenter) {{
        map.setView([lat, lon]);
      }}
    }}
    /**
     * Update the current position on the map and record it in the history.
     * This calls updateMarker internally and adds a small circle marker for logging.
     * @param {{number}} lat  Latitude in decimal degrees
     * @param {{number}} lon  Longitude in decimal degrees
     * @param {{string}} info Tooltip content shown when hovering over the history point
     */
    function updatePosition(lat, lon, info) {{
      // Always update the main marker and conditionally recenter
      updateMarker(lat, lon);
      // Create a small red circle at the given location
      var pt = L.circleMarker([lat, lon], {{ radius: 4, color: 'red', fillColor: 'red', fillOpacity: 0.8 }}).addTo(map);
      if (info) {{
        // Attach a tooltip for hover display; tooltips show on mouseover by default
        pt.bindTooltip(info);
      }}
      history.push(pt);
    }}
    /**
     * Get current bounds of the map as [south, north, west, east].
     */
    function getBounds() {{
      var b = map.getBounds();
      return [b.getSouth(), b.getNorth(), b.getWest(), b.getEast()].join(',');
    }}
    /**
     * Create a control button to toggle automatic centering on the user's position.
     * The button is inactive by default.  Clicking it toggles autoCenter and updates its style.
     */
    var centerControl = L.control({{position: 'topleft'}});
    centerControl.onAdd = function (map) {{
      var div = L.DomUtil.create('div', 'leaflet-bar leaflet-control');
      var button = L.DomUtil.create('a', '', div);
      button.href = '#';
      button.title = 'Toggle auto-center';
      // Use a simple icon for the button; Unicode target symbol
      // Use a location pin icon for the auto-center button to make its purpose clearer.
      button.innerHTML = '&#x1F4CD;';
      function updateStyle() {{
        if (autoCenter) {{
          L.DomUtil.addClass(button, 'active-center');
        }} else {{
          L.DomUtil.removeClass(button, 'active-center');
        }}
      }}
      updateStyle();
      L.DomEvent.on(button, 'click', L.DomEvent.stopPropagation)
                .on(button, 'click', L.DomEvent.preventDefault)
                .on(button, 'click', function() {{
                  autoCenter = !autoCenter;
                  // If enabling, immediately recenter the map on the current marker location
                  if (autoCenter) {{
                    var loc = marker.getLatLng();
                    map.setView(loc);
                  }}
                  updateStyle();
                }});
      return div;
    }};
    centerControl.addTo(map);
  </script>
</body>
</html>
"""
        # Write the HTML to disk
        try:
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html)
        except Exception as e:
            logger.error(f"Failed to write GNSS map HTML: {e}")
        # Store path for later loading
        self.gnss_map_html_path = html_path

    def _get_bin_params(self):
        try:
            s = self.bin_start_edit.text().strip()
            e = self.bin_end_edit.text().strip()
            w = self.bin_width_edit.text().strip()

            bin_range = None
            if s and e:
                start = float(s); end = float(e)
                if start < end:
                    bin_range = (start, end)

            bins = None
            if w:
                width = float(w)
                if width > 0:
                    if bin_range:
                        span = bin_range[1] - bin_range[0]
                        bins = max(1, int(round(span / width)))
                    else:
                        bins = max(1, int(width))  # 해석: width 입력을 ‘bin 수’로 간주
            return bin_range, bins
        except ValueError:
            return None, None

    def _get_bin_range(self):
        try:
            start_txt = self.bin_start_edit.text().strip()
            end_txt   = self.bin_end_edit.text().strip()
            if start_txt == "" or end_txt == "":
                return None
            start = float(start_txt)
            end   = float(end_txt)
            if start >= end:
                return None
            return (start, end)
        except ValueError:
            return None

    def open_file_dialog(self):
        default_dir = self.settings.value("defaultFilePath", os.getcwd())
        fnames, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Open StanforD Files",
            default_dir,
            "Files (*.pri *.apt *.prd *.stm *.drf)"
        )
        if not fnames:
            return
        new_dir = os.path.dirname(fnames[0])
        self.settings.setValue("defaultFilePath", new_dir)

        for f in fnames:
            self._add_to_library(f)
        self.load_file(fnames[0])
    
    # 드래그 앤 드롭 관련 이벤트 처리
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                fp = url.toLocalFile().lower()
                if fp.endswith(".pri") or fp.endswith(".apt") or fp.endswith(".prd") or fp.endswith(".stm") or fp.endswith(".drf"):
                    event.acceptProposedAction()
                    return
        
        event.ignore()
    
    def dropEvent(self, event):
        if not event.mimeData().hasUrls():
            return
        
        files = []
        for url in event.mimeData().urls():
            fp = url.toLocalFile()
            lower_fp = fp.lower()
            if lower_fp.endswith(".pri") or lower_fp.endswith(".apt") or lower_fp.endswith(".prd") or lower_fp.endswith(".stm") or lower_fp.endswith(".drf"):
                files.append(fp)
        if not files:
            return
        
        for f in files:
            self._add_to_library(f)
        self.load_file(files[0])        

    def _reset_data(self):
        """Reset all data when loading a new file"""
        # 데이터 초기화
        self.openingFile = []
        self.maxNum = 0
        self.df = None
        self.tree_data = None
        self.log_data = None
        
        # UI 초기화
        self.file_summary_text.setText("No file loaded.")
        self.tree_summary_text.setText("No data available.")
        self.log_summary_text.setText("No data available.")
        
        # 테이블 초기화
        empty_model = PandasModel(pd.DataFrame())
        self.raw_data_table.setModel(empty_model)
        self.tree_table.setModel(empty_model)
        self.log_table.setModel(empty_model)
        
        # 시각화 초기화
        self.figure.clear()
        self.canvas.draw()
        
        # 버튼 상태 초기화
        self.export_button.setEnabled(False)

        raw_idx = self.tab_control.indexOf(self.raw_data_tab)
        for i in range(self.tab_control.count()):
            if i != raw_idx:
                self.tab_control.setTabEnabled(i, False)

    def _add_to_library(self, filepath):
        if filepath in self.fileLibrary:
            return
        self.fileLibrary.append(filepath)
        item = QtWidgets.QListWidgetItem(os.path.basename(filepath))
        item.setData(QtCore.Qt.UserRole, filepath)
        self.file_list_widget.addItem(item)

        self._preload_file(filepath)

    def on_library_item_clicked(self, item):
        filepath = item.data(QtCore.Qt.UserRole)
        if filepath:
            self.load_file(filepath)    

    # 파일 로딩: QThread와 QProgressDialog 사용
    def load_file(self, filename):
        try:
            if filename in self.file_cache:
                self.apply_cached_file(filename)
                return

            self._reset_data()
            self.current_file = filename

            self.progressDialog = QtWidgets.QProgressDialog(
                "Loading file...", "Cancel", 0, 100, self)
            self.progressDialog.setWindowModality(QtCore.Qt.WindowModal)
            self.progressDialog.setMinimumDuration(0)
            self.progressDialog.show()
            
            self.loaderThread = FileLoaderThread(filename)
            self.loaderThread.progressChanged.connect(self.progressDialog.setValue)
            self.loaderThread.loadingFinished.connect(self.on_file_loaded)
            self.loaderThread.start()
            
            # 파일 정보 업데이트
            file_name = os.path.basename(filename)
            file_size = os.path.getsize(filename) / 1024  # KB
            self.file_info_label.setText(f"File: {file_name}\nSize: {file_size:.2f} KB")
            
            # 분석 버튼 활성화
            self.analyze_button.setEnabled(True)
            
            # 상태 표시줄 업데이트
            self.statusBar().showMessage(f"Loading file: {file_name}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
    
    def on_file_loaded(self, pri_list, maxNum):
        self.openingFile = pri_list
        self.maxNum = maxNum
        self.populate_all()
        self.setWindowTitle("ForestNAV " + self.sw_version)
        # close the loader dialog and update status
        self.progressDialog.close()
        
        self.statusBar().showMessage(f"File loaded: {os.path.basename(self.loaderThread.filename)}")
        raw_model  = PandasModel(self.df)
        cache = {
            "openingFile": self.openingFile,
            "maxNum":      self.maxNum,
            "df":          self.df,
            "raw_model":   raw_model
        }
        if hasattr(self, "tree_data") and self.tree_data is not None:
            cache.update({
                "tree_data":  self.tree_data,
                "log_data":   self.log_data,
                "tree_model": PandasModel(self.tree_data),
                "log_model":  PandasModel(self.log_data),
            })
        self.file_cache[self.current_file] = cache

    def _on_auto_clicked(self):
        self.bin_start_edit.clear()
        self.bin_end_edit.clear()
        self.bin_width_edit.clear()
        self._update_visualization()

    def populate_all(self):
        try:
            max_len = max((len(pf.valueArr) for pf in self.openingFile), default=0)

            matrix = []
            cols   = []
            for pf in self.openingFile:
                arr = list(pf.valueArr)
                # 패딩
                if len(arr) < max_len:
                    arr += [""] * (max_len - len(arr))
                matrix.append(arr)
                cols.append(str(pf.number))

            df = pd.DataFrame(matrix).T
            df.columns = cols

            self.df = df
            self.raw_data_table.setModel(PandasModel(self.df))
            self.raw_data_table.resizeColumnsToContents()
            self.statusBar().showMessage(
                f"{self.df.shape[1]} columns × {self.df.shape[0]} rows loaded"
            )

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error in populate_all", str(e))

    # def analyze_file(self):
    #     if not self.current_file: return
    #     self.analyze_button.setEnabled(False); self.statusBar().showMessage("Analyzing…")
    #     self.parser.parse_file(self.current_file)
    
    def analyze_file(self):
        """Library에 담긴 모든 PRI 파일을 차례로 분석한다."""
        if not self.fileLibrary:
            return

        # UI 상태 잠금
        self.analyze_button.setEnabled(False)
        self.statusBar().showMessage("Analyzing all datasets…")
        self.progress_bar.setValue(0)

        # ① 잠시 팝업/재귀 호출을 막기 위해 parsingFinished 시그널 해제
        try:
            self.parser.parsingFinished.disconnect(self._on_parsing_finished)
        except TypeError:
            pass   # 이미 끊겨 있을 수도 있음

        # ② 라이브러리 순회 • 캐시에 누적
        for fp in self.fileLibrary:
            self.current_file = fp          # “현재 파일” 포인터 갱신
            self.statusBar().showMessage(f"Analyzing {os.path.basename(fp)} …")

            ok = self.parser.parse_file(fp)  # ← 동기식 파싱
            if not ok:
                continue                    # 실패한 파일은 건너뜀

            tree_df = self.parser.get_tree_data()
            log_df  = self.parser.get_log_data()

            cache = self.file_cache.get(fp, {})
            cache.update({
                "tree_data":  tree_df,
                "log_data":   log_df,
                "tree_model": PandasModel(tree_df),
                "log_model":  PandasModel(log_df),
            })
            self.file_cache[fp] = cache

        # ③ 시그널 복구
        self.parser.parsingFinished.connect(self._on_parsing_finished)

        # ④ 마지막 파일을 화면에 띄우고, 탭/시각화/지도 업데이트
        if self.fileLibrary:
            self.current_file = self.fileLibrary[-1]
            self.tree_data = self.file_cache[self.current_file]["tree_data"]
            self.log_data  = self.file_cache[self.current_file]["log_data"]
            self.visualizer.set_data(self.tree_data, self.log_data)

        self._update_ui_after_analysis()     # → 모든 탭 다시 그림
        self._update_map_tab()               # → 여러 세트 동시 표기

        # ⑤ UI 복구
        self.export_button.setEnabled(True)
        self.analyze_button.setEnabled(True)
        self.statusBar().showMessage("Analysis complete (all files)")
        QtWidgets.QMessageBox.information(
            self, "Analysis", "All loaded datasets have been analyzed."
        )

    def _update_progress(self, p):
        self.progress_bar.setValue(int(p)); self.progress_label.setText(f"{p:.1f}%")

    def _on_parsing_finished(self, ok):
        if not ok:
            QtWidgets.QMessageBox.critical(self,"Error","Parse failed"); self.analyze_button.setEnabled(True); return
        self.tree_data = self.parser.get_tree_data(); self.log_data = self.parser.get_log_data()
        self.visualizer.set_data(self.tree_data, self.log_data)
        for i in range(self.tab_control.count()): self.tab_control.setTabEnabled(i, True)
        self._update_summary_tab(); self._update_tree_tab(); self._update_log_tab(); self._update_visualization(); self._update_map_tab()
        self.export_button.setEnabled(True); self.analyze_button.setEnabled(True)
        self.statusBar().showMessage("Analysis complete")
        QtWidgets.QMessageBox.information(
            self,
            "Analysis",
            "Analysis complete."
        )

        cache = self.file_cache.get(self.current_file, {})
        cache.update({
            "tree_data":  self.tree_data,
            "log_data":   self.log_data,
            "tree_model": PandasModel(self.tree_data),
            "log_model":  PandasModel(self.log_data),
        })
        self.file_cache[self.current_file] = cache
    
    def _update_ui_after_analysis(self):
        for i in range(self.tab_control.count()):
            self.tab_control.setTabEnabled(i, True)

        self._update_summary_tab()
        self._update_tree_tab()
        self._update_log_tab()
        self._update_visualization()
        self._update_map_tab()
        
        self.export_button.setEnabled(True)
        
        self.analyze_button.setEnabled(True)

    def _find_var(self, var_no:str, default="N/A"):
        for pf in self.openingFile:
            if str(pf.number) == var_no:
                return pf.value
        return default
    
    def _update_summary_tab(self, *args):
        info = self.parser.get_file_info()
        fs = f"File: {info['file_name']}\nSize: {info['file_size']:.2f} KB\nSoftware: {self._find_var('5')}"
        self.file_summary_text.setText(fs)

        td = self.tree_data
        if td is None or td.empty:
            self.tree_summary_text.setText("No tree data.")
        else:
            lines = []
            for cb in self.tree_option_checkboxes:
                if not cb.isChecked(): continue
                name = cb.text()
                if name == "# of trees":
                    lines.append(f"The number of trees: {len(td)}")
                elif name == "DBH" and "DBH" in td.columns:
                    s = td["DBH"].astype(float).replace(0, np.nan).dropna()
                    lines.append(f"DBH (mm): mean {s.mean():.2f} | min {s.min():.2f} | max {s.max():.2f}")
                elif name == "Coordinates" and {"Latitude","Longitude"}<=set(td.columns):
                    lat = td["Latitude"].astype(float).dropna()
                    lon = td["Longitude"].astype(float).dropna()
                    lines.append(f"Coordinates (mean): ({lat.mean():.6f}, {lon.mean():.6f})")
                elif name == "Altitude" and "Altitude" in td.columns:
                    s = td["Altitude"].astype(float).dropna()
                    lines.append(f"Altitude (m): mean {s.mean():.2f}")
                elif name == "Stem Type" and "Stem Type" in td.columns:
                    for val,cnt in td["Stem Type"].value_counts().items():
                        lines.append(f"Stem Type {val}: {cnt}")
                elif name == "Species Number" and "Species Number" in td.columns:
                    for val,cnt in td["Species Number"].value_counts().items():
                        lines.append(f"Species {val}: {cnt}")
            self.tree_summary_text.setText("\n".join(lines) or "Please select at least one field.")

        ld = self.log_data
        if ld is None or ld.empty:
            self.log_summary_text.setText("No log data.")
        else:
            lines = []
            for cb in self.log_option_checkboxes:
                if not cb.isChecked(): continue
                name = cb.text()
                if name == "# of logs":
                    lines.append(f"The number of logs: {len(ld)}")
                elif name.startswith("Diameter ob") or name.startswith("Diameter ub"):
                    side = "ob" if "ob" in name else "ub"
                    pos  = "Top" if "Top" in name else "Mid"
                    col  = f"Diameter ({pos} mm {side})"
                    if col in ld.columns:
                        s = ld[col].astype(float).replace(0, np.nan).dropna()
                        lines.append(f"{col}: mean {s.mean():.2f} | min {s.min():.2f} | max {s.max():.2f}")
                elif name == "Length (cm)":
                    col = self.visualizer.column_mapping["length"]
                    if col in ld.columns:
                        s = ld[col].astype(float).replace(0, np.nan).dropna()
                        lines.append(f"{col}: mean {s.mean():.2f} | min {s.min():.2f} | max {s.max():.2f}")
                elif name == "Volume (m3)":
                    for c in ["Volume (m3sob)", "Volume (m3sub)"]:
                        if c in ld.columns:
                            s = ld[c].astype(float).replace(0, np.nan).dropna()
                            lines.append(f"{c}: mean {s.mean():.3f} | min {s.min():.3f} | max {s.max():.3f}")
                            break
                elif name == "Volume (dl)":
                    col = "Volume (Var161) in dl"
                    if col in ld.columns:
                        s = ld[col].astype(float).replace(0, np.nan).dropna()
                        lines.append(f"{col}: mean {s.mean():.2f} | min {s.min():.2f} | max {s.max():.2f}")
                
                elif name == "Volume (Decimal)":
                    col = "Volume (Decimal)"
                    if col in ld.columns:
                        s = ld[col].astype(float).replace(0, np.nan).dropna()
                        lines.append(f"{col}: mean {s.mean():.2f} | min {s.min():.2f} | max {s.max():.2f}")
                        
            self.log_summary_text.setText("\n".join(lines) or "Please select at least one field.")
    
    def _update_tree_tab(self):
        """Update tree data tab"""
        if self.tree_data is not None and not self.tree_data.empty:
            model = PandasModel(self.tree_data)
            self.tree_table.setModel(model)
            self.tree_table.resizeColumnsToContents()
    
    def _update_log_tab(self):
        """Update log data tab"""
        if self.log_data is not None and not self.log_data.empty:
            model = PandasModel(self.log_data)
            self.log_table.setModel(model)
            self.log_table.resizeColumnsToContents()

    def _update_visualization(self):
        if hasattr(self, "figure"):
            self.figure.clear()
            self.canvas.draw()
        ax = self.figure.add_subplot(111)

        viz_type  = self.viz_type_combo.currentText()
        bin_range, bins_override = self._get_bin_params()
        counts_df = None

        if self.tree_data is not None and not self.tree_data.empty:
            if viz_type == "DBH Distribution":
                counts_df = self.visualizer.plot_dbh_distribution(ax, self.tree_data, bins=bins_override or 20, bin_range=bin_range)

            elif viz_type == "Volume Distribution (m3)":
                counts_df = self.visualizer.plot_volume_m3_distribution(ax, self.tree_data, bins=bins_override or 20, bin_range=bin_range)
            
            elif viz_type == "Volume Distribution (dl)":
                counts_df = self.visualizer.plot_volume_dl_distribution(ax, self.tree_data, bins=bins_override or 20, bin_range=bin_range)

            elif viz_type == "Species Distribution":
                counts_df = self.visualizer.plot_species_distribution(ax, self.tree_data)

        if self.log_data is not None and not self.log_data.empty:
            if viz_type == "Log Length Distribution":
                counts_df = self.visualizer.plot_log_length_distribution(ax, self.log_data,
                                                                        bin_range=bin_range)
            elif viz_type == "Diameter ob Top Distribution":
                counts_df = self.visualizer.plot_log_diameter_ob_top(ax, self.log_data, bins=bins_override or 20, bin_range=bin_range)

            elif viz_type == "Diameter ob Mid Distribution":
                counts_df = self.visualizer.plot_log_diameter_ob_mid(ax, self.log_data, bins=bins_override or 20, bin_range=bin_range)

            elif viz_type == "Diameter ub Top Distribution":
                counts_df = self.visualizer.plot_log_diameter_ub_top(ax, self.log_data, bins=bins_override or 20, bin_range=bin_range)

            elif viz_type == "Diameter ub Mid Distribution":
                counts_df = self.visualizer.plot_log_diameter_ub_mid(ax, self.log_data, bins=bins_override or 20, bin_range=bin_range)

        ylabel_map = {
            "tree": ["DBH", "Volume", "Species"],
            "log":  ["Log Length", "Diameter ob Top", "Diameter ob Mid", "Diameter ub Top", "Diameter ub Mid"]
        }

        if viz_type in ylabel_map["tree"]:
            ax.set_ylabel("The number of trees")
        elif viz_type in ylabel_map["log"]:
            ax.set_ylabel("The number of logs")

        if counts_df is not None:
            self.viz_table.setModel(PandasModel(counts_df))
            self.viz_table.resizeColumnsToContents()
        else:
            self.viz_table.setModel(PandasModel(pd.DataFrame()))
        
        for patch in ax.patches:
            height = patch.get_height()
            if height is not None and height > 0:
                x = patch.get_x() + patch.get_width() / 2
                ax.text(x, height, f"{int(height)}",
                        ha='center', va='bottom')
            
        self.figure.tight_layout()
        self.canvas.draw()
    
    def export_results(self):
        """Export analysis results for all loaded PRI files"""
        if not self.fileLibrary:
            QtWidgets.QMessageBox.information(self, "Info", "No files to export.")
            return

        # 최상위 내보내기 폴더 선택 (기본 경로는 이전에 저장된 defaultFilePath 사용)
        default_export_dir = self.settings.value("defaultFilePath", os.getcwd())
        base_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Base Export Directory", default_export_dir
        )
        if not base_dir:
            return

        # 사용자가 선택한 경로를 다시 defaultFilePath 로 저장 (혹시 export 디렉토리도 기본으로 쓸 수 있게)
        self.settings.setValue("defaultFilePath", base_dir)

        # 기본 좌표계 불러오기 (없으면 EPSG:4326 로 간주)
        default_crs = self.settings.value("defaultExportCRS", "EPSG:4326")

        for fp in self.fileLibrary:
            name = os.path.splitext(os.path.basename(fp))[0]
            tgt_dir = os.path.join(base_dir, f"{name}_export")
            os.makedirs(tgt_dir, exist_ok=True)

            # 캐시에서 해당 파일의 데이터 가져오기
            cache = self.file_cache.get(fp, {})
            tree_df = cache.get("tree_data")
            log_df  = cache.get("log_data")

            # 1) Tree CSV
            if tree_df is not None and not tree_df.empty:
                # 일반 CSV 저장
                tree_csv_path = os.path.join(tgt_dir, f"{name}_tree.csv")
                tree_df.to_csv(tree_csv_path, index=False)

                # GeoDataFrame 을 만들어 좌표계 정보와 함께 shapefile 로 저장 (예시)
                if {"Latitude", "Longitude"} <= set(tree_df.columns):
                    try:
                        gdf = gpd.GeoDataFrame(
                            tree_df.copy(),
                            geometry=gpd.points_from_xy(
                                tree_df["Longitude"].astype(float),
                                tree_df["Latitude"].astype(float)
                            ),
                            crs="EPSG:4326"  # 원본 위도/경도가 WGS84 라 가정
                        )
                        # 설정된 기본 CRS 로 변환
                        if default_crs and default_crs != "EPSG:4326":
                            gdf = gdf.to_crs(default_crs)
                        shp_path = os.path.join(tgt_dir, f"{name}_tree.shp")
                        gdf.to_file(shp_path)
                    except Exception as e:
                        logger.warning(f"Could not export tree shapefile for {name}: {e}")

            # 2) Log CSV
            if log_df is not None and not log_df.empty:
                log_csv_path = os.path.join(tgt_dir, f"{name}_log.csv")
                log_df.to_csv(log_csv_path, index=False)

        QtWidgets.QMessageBox.information(self, "Export", "Export complete.")
    
    def export_file(self):
        try:
            if self.df is None or self.df.empty:
                QtWidgets.QMessageBox.information(self, "Info", "Please open a PRI File first.")
                return
            fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export CSV", "", "CSV Files (*.csv)")
            if fname:
                self.df.to_csv(fname, index=False)
                QtWidgets.QMessageBox.information(self, "Info", "Export Completed")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

    def show_about(self):
        """Show about dialog"""
        QtWidgets.QMessageBox.about(self, "About forestNAV", 
                                   "forestNAV - StanforD Harvester Head Data Analysis Tool\n"
                                   f"Version: {self.sw_version}\n"
                                   "Heechan Jeong, Taehyung Kim, and Heesung Woo\n"
                                   "Advanced forestry Systems Lab\n"
                                   "Oregon State University"
                                   )
    
    def closeEvent(self, event):
        reply = QtWidgets.QMessageBox.question(
            self,
            "Exit",
            "Are you sure you want to quit the forestNAV?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def apply_cached_file(self, filepath):
        cache = self.file_cache[filepath]

        # --- 핵심 데이터 복원 -------------------------------------------
        self.current_file = filepath
        self.openingFile  = cache["openingFile"]
        self.maxNum       = cache["maxNum"]
        self.df           = cache["df"]

        # --- Raw Data 탭 -------------------------------------------------
        self.raw_data_table.setModel(cache.get("raw_model")
                                 or PandasModel(self.df))
        self.raw_data_table.resizeColumnsToContents()

        if "tree_data" in cache and "log_data" in cache:
            self.tree_data = cache["tree_data"]
            self.log_data  = cache["log_data"]

            # ▸ 모델이 없으면 즉석에서 만들어 꽂아 줍니다.
            self.tree_table.setModel(cache.get("tree_model")
                                    or PandasModel(self.tree_data))
            self.log_table.setModel (cache.get("log_model")
                                    or PandasModel(self.log_data))

            self.visualizer.set_data(self.tree_data, self.log_data)
            self._update_ui_after_analysis()
        else:
            # 아직 분석 전인 파일이면 Raw Data 탭만 살려 둡니다.
            raw_idx = self.tab_control.indexOf(self.raw_data_tab)
            for i in range(self.tab_control.count()):
                self.tab_control.setTabEnabled(i, i == raw_idx)
            self.analyze_button.setEnabled(True)

        # --- 상태바·파일 정보 -------------------------------------------
        size_kb = os.path.getsize(filepath) / 1024
        self.file_info_label.setText(
            f"File: {os.path.basename(filepath)}\nSize: {size_kb:.2f} KB"
        )
        self.statusBar().showMessage(f"Loaded from cache: {os.path.basename(filepath)}")

    def _preload_file(self, filepath: str):
        """파일을 미리 파싱해 cache 에 넣는다(진행바 없이)."""
        if filepath in self.file_cache or filepath in self._preload_threads:
            return                      # 이미 끝났거나 진행 중
        th = FileLoaderThread(filepath)
        th.loadingFinished.connect(
            lambda pri, mx, fp=filepath: self._on_preload_finished(fp, pri, mx)
        )
        th.start()
        self._preload_threads[filepath] = th

    def _on_preload_finished(self, filepath, pri_list, max_num):
        # populate_all()과 동일한 방식으로 DataFrame 생성
        max_len = max((len(p.valueArr) for p in pri_list), default=0)
        matrix, cols = [], []
        for p in pri_list:
            arr = list(p.valueArr) + [""] * (max_len - len(p.valueArr))
            matrix.append(arr)
            cols.append(str(p.number))
        df = pd.DataFrame(matrix).T
        df.columns = cols

        # 캐시에 통째로 보관(PandasModel까지)
        self.file_cache[filepath] = {
            "openingFile": pri_list,
            "maxNum":      max_num,
            "df":          df,
            "raw_model":   PandasModel(df),
        }

        # 스레드 정리
        th = self._preload_threads.pop(filepath, None)
        if th:
            th.deleteLater()

    def show_file_path_settings(self):
        current_dir = self.settings.value("defaultFilePath", os.getcwd())
        selected_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Default File Path", current_dir
        )
        if selected_dir:
            self.settings.setValue("defaultFilePath", selected_dir)
            QtWidgets.QMessageBox.information(
                self, "Settings", f"Default File Path set to:\n{selected_dir}"
            )

    def show_export_settings(self):
        dlg = ExportSettingsDialog(self)
        saved_crs = self.settings.value("defaultExportCRS", "EPSG:4326")
        idx = dlg.crs_combo.findData(saved_crs)
        if idx >= 0:
            dlg.crs_combo.setCurrentIndex(idx)

        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            selected_crs = dlg.selected_crs()
            if selected_crs:
                self.settings.setValue("defaultExportCRS", selected_crs)
                QtWidgets.QMessageBox.information(
                    self, "Settings", f"Default Export CRS set to:\n{selected_crs}"
                )

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    img_path = os.path.join(os.path.dirname(__file__), "FIELD.png")
    if os.path.exists(img_path):
        splash_pix = QtGui.QPixmap(img_path)
    else:                                  
        splash_pix = QtGui.QPixmap(400, 300)
        splash_pix.fill(QtGui.QColor('gray'))
    splash = QtWidgets.QSplashScreen(splash_pix)
    splash.show()
    QtWidgets.qApp.processEvents()
    time.sleep(2)

    window = MainWindow()
    window.show()
    splash.finish(window)
    sys.exit(app.exec_())
