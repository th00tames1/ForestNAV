import os, chardet
import folium, tempfile
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from PyQt5 import QtCore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('pri_parser')

class StanForDVariable:
    """Class representing a StanForD variable"""
    
    def __init__(self, var: int, type_val: int, description: str, unit: str = ""):
        """
        Initialize StanForD variable.
        
        Args:
            var: Variable number
            type_val: Type value
            description: Variable description
            unit: Unit of measurement
        """
        self.var = var
        self.type_val = type_val
        self.description = description
        self.unit = unit

class PRIParser(QtCore.QObject):
    """PRI File Parser Class with PyQt5 integration"""
    
    progressChanged = QtCore.pyqtSignal(float)
    parsingFinished = QtCore.pyqtSignal(bool)
    
    LOG_HEADER_MAP = {
        "1"   : "Type",
        "2"   : "Species Number",
        "20"  : "Unique ID",
        "201" : "Diameter (Top mm ob)",
        "202" : "Diameter (Top mm ub)",
        "203" : "Diameter (Mid mm ob)",
        "204" : "Diameter (Mid mm ub)",
        "205" : "Diameter (Root mm ob)",
        "206" : "Diameter (Root mm ub)",
        "207" : "Middle diameter (HKS measurement mm ob)",
        "208" : "Middle diameter (HKS measurement mm ub)",
        "300" : "Forced cross-cut",
        "301" : "Length (cm)",
        "302" : "Length class",
        "400" : "Volume (Var161)",
        "1400": "Volume (Decimal)",
        "401" : "Volume (m3sob)",
        "1401": "Volume (m3sob Decimal)",
        "402" : "Volume (m3sub)",
        "1402": "Volume (m3sub Decimal)",
        "403" : "Volume (m3topob)",
        "1403": "Volume (m3topob Decimal)",
        "404" : "Volume (m3topub)",
        "1404": "Volume (m3topub Decimal)",
        "405" : "Volume (m3smiob)",
        "1405": "Volume (m3smiob Decimal)",
        "406" : "Volume (m3smiub)",
        "1406": "Volume (m3smiub Decimal)",
        "420" : "Volume (Var161) in dl",
        "421" : "Volume (dlsob)",
        "422" : "Volume (dlsub)",
        "423" : "Volume (dltopob)",
        "424" : "Volume (dltopub)",
        "425" : "Volume (dlsmiob)",
        "426" : "Volume (dlsmiub)",
        "500" : "Tree ID (Stem number)",
        "501" : "Log number (Stem log number)",
        "600" : "Number of log",
        "2001": "Reserved"
    }

    TREE_HEADER_MAP = {
        "1"   : "Type",
        "2"   : "Species Number",
        "500" : "Tree ID (Stem Number)",
        "505" : "Suitable for Bio Energy Flag",
        "723" : "Reference Diameter for DBH",
        "724" : "Reference Diameter Height",
        "740" : "DBH (mm)",
        "741" : "Stem Type",
        "750" : "Operator Number",
        "760" : "Latitude",
        "761" : "North/South Flag",
        "762" : "Longitude",
        "763" : "East/West Flag",
        "764" : "Altitude",
        "765" : "Height (dm)",
        "766" : "Height (m)",
        "767" : "Volume (dm3)",
        "768" : "Volume (m3)",
        "769" : "DBH (mm)",
        "770" : "DBH (cm)",
        "771" : "Log Count",
        "772" : "Number of log",
        "2001": "Reserved"
    }
    
    def __init__(self):
        """Initialize PRI parser"""
        super(PRIParser, self).__init__()
        
        self.tree_header = []
        self.tree_raw_data = []
        self.log_header = []
        self.log_raw_data = []

        self.log_header_map  = PRIParser.LOG_HEADER_MAP
        self.tree_header_map = PRIParser.TREE_HEADER_MAP

        self.file_info = {
            'file_name': '',
            'file_size': 0,
            'tree_count': 0,
            'log_count': 0
        }

    def _build_table(self, header: List[str], raw_data: List[str]) -> List[List[str]]:
        colCount = len(header)
        rows = []
        rec = []
        colidx = 0

        for i, token in enumerate(raw_data):
            if colidx == 0:
                rec = []
                
            rec.append(token)
            
            if colidx < colCount - 1:
                colidx += 1
            else:
                rows.append(rec)
                colidx = 0
        
        if colidx > 0:
            if len(rec) < colCount:
                rec += [""] * (colCount - len(rec))
            rows.append(rec)
            
        return rows

    def _process_coordinates(self, rows, header):
        if not rows:
            return rows

        lat_idx = lat_flag_idx = lon_idx = lon_flag_idx = -1
        for i, col in enumerate(header):
            if   col == "Latitude":            lat_idx = i
            elif col == "North/South Flag":    lat_flag_idx = i
            elif col == "Longitude":           lon_idx = i
            elif col == "East/West Flag":      lon_flag_idx = i

        # --- latitude -------------------------------------------------------
        if lat_idx >= 0 and lat_flag_idx >= 0:
            for row in rows:
                try:
                    raw   = int(row[lat_idx])
                    flag  = str(row[lat_flag_idx])
                    value = -raw * 1e-5 if flag == "2" else raw * 1e-5
                    row[lat_idx] = f"{value:.5f}"
                except Exception:
                    continue

        # --- longitude ------------------------------------------------------
        if lon_idx >= 0 and lon_flag_idx >= 0:
            for row in rows:
                try:
                    raw   = int(row[lon_idx])
                    flag  = str(row[lon_flag_idx])
                    value = -raw * 1e-5 if flag == "2" else raw * 1e-5
                    row[lon_idx] = f"{value:.5f}"
                except Exception:
                    continue

        return rows

    def parse_file(self, file_path: str) -> bool:
        try:
            self.tree_header = []
            self.tree_raw_data = []
            self.log_header = []
            self.log_raw_data = []
            
            self.file_info['file_name'] = os.path.basename(file_path)
            self.file_info['file_size'] = os.path.getsize(file_path) / 1024  # KB
            
            with open(file_path, "rb") as f:
                content = f.read()

            detect = chardet.detect(content[:4000])
            encoding = detect.get("encoding") or "utf-8"
            try:
                decoded_content = content.decode(encoding, errors="replace")
            except Exception as e:
                logger.error(f"Error decoding file: {e}")
                self.parsingFinished.emit(False)
                return False
            
            lines = decoded_content.split("~")
            total_lines = len(lines)
            
            self.progressChanged.emit(0)
            
            for i, line in enumerate(lines):
                if i % max(1, total_lines // 10) == 0:
                    progress = (i / total_lines) * 100
                    self.progressChanged.emit(progress)

                if not line.strip():
                    continue

                parts = line.split('~')
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                    
                    tokens = part.split()
                    if len(tokens) < 2:
                        continue
                    try:
                        var = int(tokens[0])
                        type_val = int(tokens[1])
            
                        if var in (256, 257, 266, 267):
                            token_list = tokens[2:]
                            if var == 256:
                                self.log_header = token_list
                            elif var == 257:
                                self.log_raw_data.extend(token_list)
                            elif var == 266:
                                self.tree_header = token_list
                            elif var == 267:
                                self.tree_raw_data.extend(token_list)
                    except (ValueError, IndexError):
                        continue

            self.progressChanged.emit(100)
            
            mapped_log_header = [self.log_header_map.get(h, h) for h in self.log_header]
            mapped_tree_header = [self.tree_header_map.get(h, h) for h in self.tree_header]
            
            log_rows = self._build_table(mapped_log_header, self.log_raw_data)
            tree_rows = self._build_table(mapped_tree_header, self.tree_raw_data)
            
            tree_rows = self._process_coordinates(tree_rows, mapped_tree_header)
            
            self.file_info['tree_count'] = len(tree_rows)
            self.file_info['log_count'] = len(log_rows)
            
            self.parsingFinished.emit(True)
            return True
            
        except Exception as e:
            logger.error(f"Error parsing file: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.parsingFinished.emit(False)
            return False
    
    def get_tree_data(self) -> pd.DataFrame:
        if not self.tree_header or not self.tree_raw_data:
            return pd.DataFrame()
        
        mapped_tree_header = [self.tree_header_map.get(h, h) for h in self.tree_header]
        
        tree_rows = self._build_table(mapped_tree_header, self.tree_raw_data)
        
        tree_rows = self._process_coordinates(tree_rows, mapped_tree_header)
        
        df = pd.DataFrame(tree_rows, columns=mapped_tree_header)
        
        for col in df.columns:
            if col in ["DBH", "Height (dm)", "Volume (dm3)", "Log Count", "Number of Log"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def get_log_data(self) -> pd.DataFrame:
        if not self.log_header or not self.log_raw_data:
            return pd.DataFrame()

        mapped_log_header = [self.log_header_map.get(h, h) for h in self.log_header]

        log_rows = self._build_table(mapped_log_header, self.log_raw_data)

        df = pd.DataFrame(log_rows, columns=mapped_log_header)
        
        for col in df.columns:
            if col in ["Physical Length", "Diameter (Top mm ob)", "Diameter (Root mm ob)", "Volume (Var161)"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def get_file_info(self) -> Dict[str, Any]:
        return self.file_info
    
    def get_variable_description(self, var: int, type_val: int) -> str:
        key = (var, type_val)
        if key in self.variables:
            return self.variables[key].description
        return "Unknown"
