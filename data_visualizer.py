"""
Data Visualization Component (PyQt5 Version)

Visualizes tree and log data parsed from StanForD PRI files.
"""

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# -----------------------------------------------------------------------------


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("data_visualizer")

plt.style.use("ggplot")


class DataVisualizer(QtCore.QObject):
    """Matplotlib/Seaborn visualizer with PyQt5 integration."""

    def __init__(self) -> None:
        super().__init__()
        self.tree_data: Optional[pd.DataFrame] = None
        self.log_data: Optional[pd.DataFrame] = None

        # Maps generic keys to actual column names after preprocessing
        self.column_mapping: Dict[str, Optional[str]] = {
            "dbh": None,
            "height": None,
            "volume": None,
            "log_count": None,
            "length": None,
            "diameter_top": None,
            "diameter_butt": None,
            "tree_number": None,
            "log_number": None,
            "species": None,
        }

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def set_data(
        self,
        tree_data: pd.DataFrame,
        log_data: pd.DataFrame,
    ) -> None:
        """Stores data and builds column mapping."""
        self.tree_data = tree_data
        self.log_data = log_data
        self._preprocess_data()
        logger.info("Visualizer data set")

    def get_summary_statistics(self) -> Dict[str, pd.DataFrame]:
        """Returns describe() tables for tree and log subsets."""
        summary: Dict[str, pd.DataFrame] = {}

        if self.tree_data is not None and not self.tree_data.empty:
            cols = [
                self.column_mapping[k]
                for k in ("dbh", "height", "volume", "log_count")
                if self.column_mapping[k] is not None
            ]
            if cols:
                summary["tree_stats"] = self.tree_data[cols].describe()

        if self.log_data is not None and not self.log_data.empty:
            cols = [
                self.column_mapping[k]
                for k in ("length", "diameter_top", "diameter_butt")
                if self.column_mapping[k] is not None
            ]
            if cols:
                summary["log_stats"] = self.log_data[cols].describe()

        return summary

    # --------------------------------------------------------------------- #
    # Pre‑processing
    # --------------------------------------------------------------------- #

    def _preprocess_data(self) -> None:
        """Detects column names in incoming DataFrames."""
        if self.tree_data is not None and not self.tree_data.empty:
            cols = self.tree_data.columns.tolist()

            if "DBH" in cols:
                self.column_mapping["dbh"] = "DBH"
            elif "DBH (mm)" in cols:
                self.column_mapping["dbh"] = "DBH (mm)"

            if "Height" in cols:
                self.column_mapping["height"] = "Height"
            elif "Height (dm)" in cols:
                self.column_mapping["height"] = "Height (dm)"

            if "Volume" in cols:
                self.column_mapping["volume"] = "Volume"
            elif "Volume (dm3)" in cols:
                self.column_mapping["volume"] = "Volume (dm3)"
            elif "Volume (Var161)" in cols:
                self.column_mapping["volume"] = "Volume (Var161)"

            if "Log Count" in cols:
                self.column_mapping["log_count"] = "Log Count"
            elif "Number of Log" in cols:
                self.column_mapping["log_count"] = "Number of Log"

            if "Tree Number" in cols:
                self.column_mapping["tree_number"] = "Tree Number"
            elif "Stem Number" in cols:
                self.column_mapping["tree_number"] = "Stem Number"

            if "Species" in cols:
                self.column_mapping["species"] = "Species"
            elif "Species Number" in cols:
                self.column_mapping["species"] = "Species Number"

            for col in (
                self.column_mapping["dbh"],
                self.column_mapping["height"],
                self.column_mapping["volume"],
                self.column_mapping["log_count"],
            ):
                if col in self.tree_data.columns:
                    self.tree_data[col] = pd.to_numeric(
                        self.tree_data[col], errors="coerce"
                    )

        if self.log_data is not None and not self.log_data.empty:
            cols = self.log_data.columns.tolist()

            if "Length (cm)" in cols:
                self.column_mapping["length"] = "Length (cm)"
            elif "Physical Length" in cols:
                self.column_mapping["length"] = "Physical Length"

            if "Diameter Top (mm)" in cols:
                self.column_mapping["diameter_top"] = "Diameter Top (mm)"
            elif "Diameter (Top mm ob)" in cols:
                self.column_mapping["diameter_top"] = "Diameter (Top mm ob)"

            if "Diameter Butt (mm)" in cols:
                self.column_mapping["diameter_butt"] = "Diameter Butt (mm)"
            elif "Diameter (Root mm ob)" in cols:
                self.column_mapping["diameter_butt"] = "Diameter (Root mm ob)"

            if "Tree Number" in cols:
                self.column_mapping["tree_number"] = "Tree Number"
            elif "Stem Number" in cols:
                self.column_mapping["tree_number"] = "Stem Number"

            if "Log Number" in cols:
                self.column_mapping["log_number"] = "Log Number"
            elif "Stem Log number" in cols:
                self.column_mapping["log_number"] = "Stem Log number"

            for col in (
                self.column_mapping["length"],
                self.column_mapping["diameter_top"],
                self.column_mapping["diameter_butt"],
            ):
                if col in self.log_data.columns:
                    self.log_data[col] = pd.to_numeric(
                        self.log_data[col], errors="coerce"
                    )

    # ---------------------------------------------------------------------
    # helper (변경된 버전)
    # ---------------------------------------------------------------------
    @staticmethod
    def _hist_df(values: pd.Series,
             bins: int,
             rng: Optional[Tuple[float, float]]) -> pd.DataFrame:
        counts, edges = np.histogram(values, bins=bins, range=rng)
        return pd.DataFrame({
            "bin_start": edges[:-1].round(2),
            "bin_end":   edges[1:].round(2),
            "count":     counts
        })

    # ---------------------------------------------------------------------
    # 1) DBH distribution
    # ---------------------------------------------------------------------
    def plot_dbh_distribution(self, ax,
                            tree_df: Optional[pd.DataFrame] = None,
                            bins: int = 20,
                            bin_range: Optional[Tuple[float, float]] = None
                            ) -> Optional[pd.DataFrame]:
        if tree_df is None:
            tree_df = self.tree_data
        col = self.column_mapping["dbh"]
        if tree_df is None or col is None:
            ax.set_title("DBH data not available")
            return None

        data = tree_df[col].dropna()
        if data.empty:
            ax.set_title("No valid DBH data")
            return None

        sns.histplot(data, ax=ax, kde=True, bins=bins, binrange=bin_range)
        ax.set_title("Tree Diameter (DBH) Distribution")
        ax.set_xlabel("DBH (mm)")
        ax.set_ylabel("The number of trees")

        df_counts = self._hist_df(data, bins, rng=bin_range)
        df_counts.columns = ["DBH_bin_start", "DBH_bin_end", "count"]
        return df_counts

    # ---------------------------------------------------------------------
    # 3) Volume distribution
    # ---------------------------------------------------------------------
    def plot_volume_distribution(self, ax,
                                tree_df: Optional[pd.DataFrame] = None,
                                bins: int = 20,
                                bin_range: Optional[Tuple[float, float]] = None
                                ) -> Optional[pd.DataFrame]:
        if tree_df is None:
            tree_df = self.tree_data
        col = self.column_mapping["volume"]
        if tree_df is None or col is None:
            ax.set_title("Volume data not available")
            return None

        data = tree_df[col].dropna()
        if data.empty:
            ax.set_title("No valid volume data")
            return None

        sns.histplot(data, ax=ax, kde=True, bins=bins, binrange=bin_range)
        ax.set_title("Tree Volume Distribution")
        ax.set_xlabel("Volume (dm3)")
        ax.set_ylabel("The number of trees")

        df_counts = self._hist_df(data, bins, rng=bin_range)
        df_counts.columns = ["Volume_bin_start", "Volume_bin_end", "count"]
        return df_counts


    # ---------------------------------------------------------------------
    # 4) Log length distribution
    # ---------------------------------------------------------------------
    def plot_log_length_distribution(self, ax,
                                    log_df: Optional[pd.DataFrame] = None,
                                    bins: int = 20,
                                    bin_range: Optional[Tuple[float, float]] = None
                                    ) -> Optional[pd.DataFrame]:
        if log_df is None:
            log_df = self.log_data
        col = self.column_mapping["length"]
        if log_df is None or col is None:
            ax.set_title("Log length data not available")
            return None

        data = log_df[col].dropna()
        if data.empty:
            ax.set_title("No valid log length data")
            return None

        sns.histplot(data, ax=ax, kde=True, bins=bins, binrange=bin_range)
        ax.set_title("Log Length Distribution")
        ax.set_xlabel("Length (cm)")
        ax.set_ylabel("The number of logs")

        df_counts = self._hist_df(data, bins, rng=bin_range)
        df_counts.columns = ["Length_bin_start", "Length_bin_end", "count"]
        return df_counts


    # ---------------------------------------------------------------------
    # 5) Log diameter distribution (top & butt)
    # ---------------------------------------------------------------------
    def plot_log_diameter_distribution(self, ax,
                                    log_df: Optional[pd.DataFrame] = None,
                                    bins: int = 20,
                                    bin_range: Optional[Tuple[float, float]] = None
                                    ) -> Optional[pd.DataFrame]:
        if log_df is None:
            log_df = self.log_data

        top_col = self.column_mapping["diameter_top"]
        butt_col = self.column_mapping["diameter_butt"]
        if log_df is None or top_col is None or butt_col is None:
            ax.set_title("Log diameter data not available")
            return None

        top_data = log_df[top_col].dropna()
        butt_data = log_df[butt_col].dropna()
        if top_data.empty and butt_data.empty:
            ax.set_title("No valid log diameter data")
            return None

        if not top_data.empty:
            sns.histplot(top_data, ax=ax, kde=True, bins=bins,
                        binrange=bin_range, label="Top")
        if not butt_data.empty:
            sns.histplot(butt_data, ax=ax, kde=True, bins=bins,
                        binrange=bin_range, alpha=0.5, label="Butt")

        ax.set_title("Log Diameter Distribution")
        ax.set_xlabel("Diameter (mm)")
        ax.set_ylabel("The number of logs")
        ax.legend()

        df_top = self._hist_df(top_data, bins, rng=bin_range).rename(
            columns={"count": "top_count"}
        )
        df_butt = self._hist_df(butt_data, bins, rng=bin_range).rename(
            columns={"count": "butt_count"}
        )
        return pd.merge(df_top, df_butt, on=["bin_start", "bin_end"], how="outer")


    # ---------------------------------------------------------------------
    # 6) Species distribution (bin_range 밖에 해당 없음)
    # ---------------------------------------------------------------------
    def plot_species_distribution(self, ax,
                                tree_df: Optional[pd.DataFrame] = None,
                                **_) -> Optional[pd.DataFrame]:
        if tree_df is None:
            tree_df = self.tree_data
        col = self.column_mapping["species"]
        if tree_df is None or col is None:
            ax.set_title("Species data not available")
            return None

        counts = tree_df[col].value_counts()
        if counts.empty:
            ax.set_title("No valid species data")
            return None

        counts.plot(kind="bar", ax=ax)
        ax.set_title("Species Distribution")
        ax.set_xlabel("Species")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)

        return counts.reset_index().rename(
            columns={"index": "species", col: "count"}
        )
    
    def plot_volume_m3_distribution(self, ax,
                                    tree_df: Optional[pd.DataFrame] = None,
                                    bins: int = 20,
                                    bin_range: Optional[Tuple[float, float]] = None
                                    ) -> Optional[pd.DataFrame]:
        if tree_df is None:
            tree_df = self.tree_data
        if tree_df is None or "Volume (m3)" not in tree_df.columns:
            ax.set_title("Volume (m3) data not available")
            return None

        data = pd.to_numeric(tree_df["Volume (m3)"], errors="coerce").dropna()
        sns.histplot(data, ax=ax, kde=True, bins=bins, binrange=bin_range)
        ax.set_title("Tree Volume Distribution (m³)")
        ax.set_xlabel("Volume (m³)")
        ax.set_ylabel("The number of trees")

        df = self._hist_df(data, bins, rng=bin_range)
        df.columns = ["bin_start", "bin_end", "count"]
        return df

    def plot_volume_dl_distribution(self, ax,
                                    tree_df: Optional[pd.DataFrame] = None,
                                    bins: int = 20,
                                    bin_range: Optional[Tuple[float, float]] = None
                                    ) -> Optional[pd.DataFrame]:
        if tree_df is None:
            tree_df = self.tree_data
        if tree_df is None or "Volume (dm3)" not in tree_df.columns:
            ax.set_title("Volume (dl) data not available")
            return None

        data = pd.to_numeric(tree_df["Volume (dm3)"], errors="coerce").dropna()
        sns.histplot(data, ax=ax, kde=True, bins=bins, binrange=bin_range)
        ax.set_title("Tree Volume Distribution (dl)")
        ax.set_xlabel("Volume (dl)")
        ax.set_ylabel("The number of trees")

        df = self._hist_df(data, bins, rng=bin_range)
        df.columns = ["bin_start", "bin_end", "count"]
        return df

    def plot_log_diameter_ob_top(self, ax, log_df=None, bins=20, bin_range=None):
        return self._single_diameter_hist(ax, log_df, "Diameter (Top mm ob)", "Log Diameter ob Top", bins, bin_range)

    def plot_log_diameter_ob_mid(self, ax, log_df=None, bins=20, bin_range=None):
        return self._single_diameter_hist(ax, log_df, "Diameter (Mid mm ob)", "Log Diameter ob Mid", bins, bin_range)

    def plot_log_diameter_ub_top(self, ax, log_df=None, bins=20, bin_range=None):
        return self._single_diameter_hist(ax, log_df, "Diameter (Top mm ub)", "Log Diameter ub Top", bins, bin_range)

    def plot_log_diameter_ub_mid(self, ax, log_df=None, bins=20, bin_range=None):
        return self._single_diameter_hist(ax, log_df, "Diameter (Mid mm ub)", "Log Diameter ub Mid", bins, bin_range)

    def _single_diameter_hist(self, ax, log_df, col_name, title, bins, bin_range):
        if log_df is None:
            log_df = self.log_data
        if log_df is None or col_name not in log_df.columns:
            ax.set_title(f"{title} data not available")
            return None

        data = pd.to_numeric(log_df[col_name], errors="coerce").dropna()
        if data.empty:
            ax.set_title(f"No valid data for {title}")
            return None

        sns.histplot(data, ax=ax, kde=True, bins=bins, binrange=bin_range)
        ax.set_title(f"{title} Distribution")
        ax.set_xlabel(f"{col_name}")
        ax.set_ylabel("The number of logs")

        df = self._hist_df(data, bins, rng=bin_range)
        df.columns = ["bin_start", "bin_end", "count"]
        return df
