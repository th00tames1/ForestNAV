"""
PRI File Parser (PyQt5 Version)

This module provides functionality for parsing PRI files.
"""

import os, chardet
import folium, tempfile
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from PyQt5 import QtCore

# Configure logging
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
    
    # 시그널 정의
    progressChanged = QtCore.pyqtSignal(float)
    parsingFinished = QtCore.pyqtSignal(bool)
    
    def __init__(self):
        """Initialize PRI parser"""
        super(PRIParser, self).__init__()
        
        # 파싱 결과 저장 변수
        self.tree_header = []
        self.tree_raw_data = []
        self.log_header = []
        self.log_raw_data = []

        self.file_info = {
            'file_name': '',
            'file_size': 0,
            'tree_count': 0,
            'log_count': 0
        }

        self.variables = {}
        self._init_variables()
    
    def _init_variables(self):
        """Initialize StanForD variables"""
        self.variables[(1, 0)] = StanForDVariable(1, 0, "Type", "Code")
        self.variables[(2, 0)] = StanForDVariable(2, 0, "Species Number", "Code")
        self.variables[(20, 0)] = StanForDVariable(20, 0, "Unique ID", "Number")
        self.variables[(201, 0)] = StanForDVariable(201, 0, "Diameter (Top mm ob)", "mm")
        self.variables[(202, 0)] = StanForDVariable(202, 0, "Diameter (Top mm ub)", "mm")
        self.variables[(203, 0)] = StanForDVariable(203, 0, "Diameter (Mid mm ob)", "mm")
        self.variables[(204, 0)] = StanForDVariable(204, 0, "Diameter (Mid mm ub)", "mm")
        self.variables[(205, 0)] = StanForDVariable(205, 0, "Diameter (Root mm ob)", "mm")
        self.variables[(206, 0)] = StanForDVariable(206, 0, "Diameter (Root mm ub)", "mm")
        self.variables[(207, 0)] = StanForDVariable(207, 0, "Middle diameter (HKS measurement mm ob)", "mm")
        self.variables[(208, 0)] = StanForDVariable(208, 0, "Middle diameter (HKS measurement mm ub)", "mm")
        self.variables[(300, 0)] = StanForDVariable(300, 0, "Forced cross-cut", "Flag")
        self.variables[(301, 0)] = StanForDVariable(301, 0, "Length (cm)", "cm")
        self.variables[(302, 0)] = StanForDVariable(302, 0, "Length class", "Code")
        self.variables[(400, 0)] = StanForDVariable(400, 0, "Volume (Var161)", "dl")
        self.variables[(1400, 0)] = StanForDVariable(1400, 0, "Volume (Decimal)", "dl")
        self.variables[(401, 0)] = StanForDVariable(401, 0, "Volume (m3sob)", "m3")
        self.variables[(1401, 0)] = StanForDVariable(1401, 0, "Volume (m3sob Decimal)", "m3")
        self.variables[(402, 0)] = StanForDVariable(402, 0, "Volume (m3sub)", "m3")
        self.variables[(1402, 0)] = StanForDVariable(1402, 0, "Volume (m3sub Decimal)", "m3")
        self.variables[(403, 0)] = StanForDVariable(403, 0, "Volume (m3topob)", "m3")
        self.variables[(1403, 0)] = StanForDVariable(1403, 0, "Volume (m3topob Decimal)", "m3")
        self.variables[(404, 0)] = StanForDVariable(404, 0, "Volume (m3topub)", "m3")
        self.variables[(1404, 0)] = StanForDVariable(1404, 0, "Volume (m3topub Decimal)", "m3")
        self.variables[(405, 0)] = StanForDVariable(405, 0, "Volume (m3smiob)", "m3")
        self.variables[(1405, 0)] = StanForDVariable(1405, 0, "Volume (m3smiob Decimal)", "m3")
        self.variables[(406, 0)] = StanForDVariable(406, 0, "Volume (m3smiub)", "m3")
        self.variables[(1406, 0)] = StanForDVariable(1406, 0, "Volume (m3smiub Decimal)", "m3")
        self.variables[(420, 0)] = StanForDVariable(420, 0, "Volume (Var161) in dl", "dl")
        self.variables[(421, 0)] = StanForDVariable(421, 0, "Volume (dlsob)", "dl")
        self.variables[(422, 0)] = StanForDVariable(422, 0, "Volume (dlsub)", "dl")
        self.variables[(423, 0)] = StanForDVariable(423, 0, "Volume (dltopob)", "dl")
        self.variables[(424, 0)] = StanForDVariable(424, 0, "Volume (dltopub)", "dl")
        self.variables[(425, 0)] = StanForDVariable(425, 0, "Volume (dlsmiob)", "dl")
        self.variables[(426, 0)] = StanForDVariable(426, 0, "Volume (dlsmiub)", "dl")
        self.variables[(500, 0)] = StanForDVariable(500, 0, "Stem Number", "Number")
        self.variables[(501, 0)] = StanForDVariable(501, 0, "Stem Log number", "Number")
        self.variables[(600, 0)] = StanForDVariable(600, 0, "Number of Log", "Number")
        self.variables[(2001, 0)] = StanForDVariable(2001, 0, "Reserved", "Text")
        
        # 트리 데이터 헤더 변수 정의
        self.variables[(1, 0)] = StanForDVariable(1, 0, "Type", "Code")
        self.variables[(2, 0)] = StanForDVariable(2, 0, "Species Number", "Code")
        self.variables[(500, 0)] = StanForDVariable(500, 0, "Stem Number", "Number")
        self.variables[(505, 0)] = StanForDVariable(505, 0, "Suitable for Bio Energy", "Flag")
        self.variables[(723, 0)] = StanForDVariable(723, 0, "Reference Diameter for DBH", "mm")
        self.variables[(724, 0)] = StanForDVariable(724, 0, "Reference Diameter Height", "cm")
        self.variables[(740, 0)] = StanForDVariable(740, 0, "DBH", "mm")
        self.variables[(741, 0)] = StanForDVariable(741, 0, "Stem Type", "Code")
        self.variables[(750, 0)] = StanForDVariable(750, 0, "Operator Number", "Number")
        self.variables[(760, 0)] = StanForDVariable(760, 0, "Latitude", "Coordinate")
        self.variables[(761, 0)] = StanForDVariable(761, 0, "North/South Flag", "Flag")
        self.variables[(762, 0)] = StanForDVariable(762, 0, "Longitude", "Coordinate")
        self.variables[(763, 0)] = StanForDVariable(763, 0, "East/West Flag", "Flag")
        self.variables[(764, 0)] = StanForDVariable(764, 0, "Altitude", "m")
        self.variables[(2001, 0)] = StanForDVariable(2001, 0, "Reserved", "Text")

    def _build_table(self, header: List[str], raw_data: List[str]) -> List[List[str]]:
        colCount = len(header)
        rows = []
        rec = []
        colidx = 0
        
        # 첫 번째 행은 인덱스 0부터 시작
        for i, token in enumerate(raw_data):
            # 새 행의 시작인 경우
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
            # 초기화
            self.tree_header = []
            self.tree_raw_data = []
            self.log_header = []
            self.log_raw_data = []
            
            # 파일 정보 저장
            self.file_info['file_name'] = os.path.basename(file_path)
            self.file_info['file_size'] = os.path.getsize(file_path) / 1024  # KB
            
            # 파일 읽기
            with open(file_path, "rb") as f:
                content = f.read()

            # 인코딩 감지: UTF-16 BOM이 있으면 utf-16-sig 사용
            detect = chardet.detect(content[:4000])
            encoding = detect.get("encoding") or "utf-8"
            # 디코딩
            try:
                decoded_content = content.decode(encoding, errors="replace")
            except Exception as e:
                logger.error(f"Error decoding file: {e}")
                self.parsingFinished.emit(False)
                return False
            
            # 레코드 분할
            lines = decoded_content.split("~")
            total_lines = len(lines)
            
            # 진행률 초기화
            self.progressChanged.emit(0)
            
            # 레코드 처리
            for i, line in enumerate(lines):
                # 진행률 업데이트 (10% 간격)
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

            # 진행률 100% 표시
            self.progressChanged.emit(100)

            # 컬럼명 매핑 (원본 코드의 colHeaderDict 참고)
            log_header_map = {
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
                "500" : "Stem Number",
                "501" : "Stem Log number",
                "600" : "Number of Log",
                "2001": "Reserved"
            }
            
            tree_header_map = {
                "1"   : "Type",
                "2"   : "Species Number",
                "500" : "Stem Number",
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
                "772" : "Number of Log",
                "2001": "Reserved"
            }
            
            # 헤더 매핑
            mapped_log_header = []
            for h in self.log_header:
                mapped_log_header.append(log_header_map.get(h, h))
            
            mapped_tree_header = []
            for h in self.tree_header:
                mapped_tree_header.append(tree_header_map.get(h, h))
            
            # 테이블 구성
            log_rows = self._build_table(mapped_log_header, self.log_raw_data)
            tree_rows = self._build_table(mapped_tree_header, self.tree_raw_data)
            
            # 좌표 처리
            tree_rows = self._process_coordinates(tree_rows, mapped_tree_header)
            
            # 파일 정보 업데이트
            self.file_info['tree_count'] = len(tree_rows)
            self.file_info['log_count'] = len(log_rows)
            
            # 파싱 완료 시그널 발생
            self.parsingFinished.emit(True)
            return True
            
        except Exception as e:
            logger.error(f"Error parsing file: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.parsingFinished.emit(False)
            return False
    
    def get_tree_data(self) -> pd.DataFrame:
        """
        Get tree data as pandas DataFrame.
        
        Returns:
            pandas.DataFrame: Tree data
        """
        if not self.tree_header or not self.tree_raw_data:
            return pd.DataFrame()
        
        # 헤더 매핑
        tree_header_map = {
            "1"   : "Type",
            "2"   : "Species Number",
            "500" : "Stem Number",
            "505" : "Suitable for Bio Energy",
            "723" : "Reference Diameter for DBH",
            "724" : "Reference Diameter Height",
            "740" : "DBH",
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
            "772" : "Number of Log",
            "2001": "Reserved"
        }
        
        mapped_tree_header = []
        for h in self.tree_header:
            mapped_tree_header.append(tree_header_map.get(h, h))
        
        # 테이블 구성
        tree_rows = self._build_table(mapped_tree_header, self.tree_raw_data)
        
        # 좌표 처리
        tree_rows = self._process_coordinates(tree_rows, mapped_tree_header)
        
        # DataFrame 생성
        df = pd.DataFrame(tree_rows, columns=mapped_tree_header)
        
        # 숫자형 데이터 변환
        for col in df.columns:
            if col in ["DBH", "Height (dm)", "Volume (dm3)", "Log Count", "Number of Log"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def get_log_data(self) -> pd.DataFrame:
        """
        Get log data as pandas DataFrame.
        
        Returns:
            pandas.DataFrame: Log data
        """
        if not self.log_header or not self.log_raw_data:
            return pd.DataFrame()
        
        # 헤더 매핑
        log_header_map = {
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
            "500" : "Stem Number",
            "501" : "Log number",
            "600" : "Number of Log",
            "2001": "Reserved"
        }
        
        mapped_log_header = []
        for h in self.log_header:
            mapped_log_header.append(log_header_map.get(h, h))

        log_rows = self._build_table(mapped_log_header, self.log_raw_data)

        df = pd.DataFrame(log_rows, columns=mapped_log_header)
        
        for col in df.columns:
            if col in ["Physical Length", "Diameter (Top mm ob)", "Diameter (Root mm ob)", "Volume (Var161)"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def get_file_info(self) -> Dict[str, Any]:
        """
        Get file information.
        
        Returns:
            dict: File information
        """
        return self.file_info
    
    def get_variable_description(self, var: int, type_val: int) -> str:
        """
        Get variable description.
        
        Args:
            var: Variable number
            type_val: Type value
            
        Returns:
            str: Variable description
        """
        key = (var, type_val)
        if key in self.variables:
            return self.variables[key].description
        return "Unknown"
