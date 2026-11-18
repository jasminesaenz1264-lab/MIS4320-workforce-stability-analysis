from dataclasses import dataclass
import io
from pathlib import Path
import sys
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd


# =====================================================================
# 1. CONSTANTS & CONFIGURATION
# =====================================================================

# Expected input schema column names
COL_ROW_LABELS = "Row Labels"
COL_STAYED = "No"
COL_LEFT = "Yes"
COL_TOTAL = "Grand Total"
REQUIRED_DATASET_COLUMNS = {COL_ROW_LABELS, COL_STAYED, COL_LEFT, COL_TOTAL}

# Derived analytics column names
COL_SATISFACTION_LEVEL = "Satisfaction Level"
COL_DISTRIBUTION_PCT = "Distribution (%)"
COL_ATTRITION_RATE_PCT = "Attrition Rate (%)"
COL_RETENTION_RATE_PCT = "Retention Rate (%)"

# Visualization styling constants
CHART_FIGSIZE = (13, 5)
CHART_DPI = 300
DEFAULT_OUTPUT_CHART_PATH = Path("satisfaction_attrition_charts.png")
COLOR_PRIMARY_BLUE = "#2b5c8f"
COLOR_EDGE_BLUE = "#1f3f63"
COLOR_BENCHMARK_LINE = "#333333"
COLOR_GRID = "#cccccc"

# Attrition level risk palette (Rating 1 = High Risk to Rating 4 = Low Risk)
PALETTE_ATTRITION_LEVELS = ["#d9534f", "#f0ad4e", "#f0ad4e", "#5cb85c"]

# Search candidates for automatic dataset discovery
CANDIDATE_DATASET_FILENAMES = (
    "WA_Fn-UseC_-HR-Employee-Attrition.csv",
    "job_satisfaction_attrition.csv",
)

# Embedded contingency table fallback (guarantees offline execution)
FALLBACK_RAW_CSV = """Count of Attrition,Column Labels,,
Row Labels,No,Yes,Grand Total
1,223,66,289
2,234,46,280
3,369,73,442
4,407,52,459
Grand Total,1233,237,1470
"""


# =====================================================================
# 2. DATA STRUCTURES & EXCEPTIONS
# =====================================================================

class DataValidationError(Exception):
    """Raised when the ingested dataset does not match the expected schema."""
    pass


@dataclass(frozen=True)
class AttritionMetrics:
    """Strongly-typed container for high-level analytical findings."""
    grand_total_employees: int
    grand_total_stayed: int
    grand_total_left: int
    overall_attrition_pct: float
    overall_retention_pct: float
    most_common_level: int
    most_common_count: int
    most_common_pct: float


# =====================================================================
# 3. DATA INGESTION & PARSING
# =====================================================================

def resolve_dataset_path(custom_path: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """Resolve the dataset file path using modern pathlib search strategies.

    Args:
        custom_path: Optional user-specified file path.

    Returns:
        Optional[Path]: Validated Path object if found, otherwise None.
    """
    if custom_path:
        path_candidate = Path(custom_path).expanduser().resolve()
        if path_candidate.is_file():
            return path_candidate

    search_directories = [
        Path.cwd(),
        Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd(),
        Path.home() / "Downloads",
        Path.home() / "OneDrive" / "Documents",
    ]

    for directory in search_directories:
        for file_name in CANDIDATE_DATASET_FILENAMES:
            candidate = directory / file_name
            if candidate.is_file():
                return candidate

    return None


def load_satisfaction_data(
    source: Optional[Union[str, Path, io.StringIO]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and sanitize the job satisfaction vs. attrition contingency table.

    Dynamically detects header offset caused by pivot table exports and splits
    rating-level data from the aggregate total row.

    Args:
        source: File path, Path object, StringIO buffer, or None for auto-discovery.

    Returns:
        Tuple[pd.DataFrame, pd.Series]:
            - Cleaned rating-level DataFrame indexed by satisfaction rating (1 to 4).
            - Series containing the aggregate summary 'Grand Total' row.

    Raises:
        DataValidationError: If expected columns are missing or unreadable.
    """
    if source is None:
        resolved_path = resolve_dataset_path()
        source = resolved_path if resolved_path else io.StringIO(FALLBACK_RAW_CSV)
    elif isinstance(source, (str, Path)):
        source = Path(source)

    # Detect header offset from Excel pivot table export format
    if isinstance(source, Path):
        with source.open("r", encoding="utf-8-sig") as file_handle:
            first_line = file_handle.readline().strip()
        header_row = 1 if "Count of Attrition" in first_line else 0
        raw_df = pd.read_csv(source, header=header_row)
    else:
        raw_df = pd.read_csv(source, header=1)

    # Standardize column headers
    raw_df.columns = [str(col).strip() for col in raw_df.columns]

    if not REQUIRED_DATASET_COLUMNS.issubset(set(raw_df.columns)):
        raise DataValidationError(
            f"Dataset missing required columns. Expected at least: {REQUIRED_DATASET_COLUMNS}"
        )

    # Isolate summary Grand Total row from individual rating levels
    is_total_row = (
        raw_df[COL_ROW_LABELS].astype(str).str.contains("Total", case=False)
    )
    totals_row = raw_df[is_total_row].iloc[0]
    ratings_df = raw_df[~is_total_row].copy()

    # Enforce numeric types across all quantitative columns
    for column in [COL_STAYED, COL_LEFT, COL_TOTAL]:
        ratings_df[column] = pd.to_numeric(ratings_df[column])

    # Standardize satisfaction rating as numeric primary index
    ratings_df[COL_SATISFACTION_LEVEL] = ratings_df[COL_ROW_LABELS].astype(int)
    ratings_df = ratings_df.sort_values(by=COL_SATISFACTION_LEVEL)
    ratings_df = ratings_df.set_index(COL_SATISFACTION_LEVEL)

    return ratings_df, totals_row


# =====================================================================
# 4. ANALYTICS & COMPUTATIONS
# =====================================================================

def compute_satisfaction_analytics(
    ratings_df: pd.DataFrame, total_row: pd.Series
) -> Tuple[pd.DataFrame, AttritionMetrics]:
    """Calculate employee distributions, attrition rates, and summary metrics.

    Args:
        ratings_df: DataFrame containing rating levels (1 to 4).
        total_row: Series containing grand total figures.

    Returns:
        Tuple[pd.DataFrame, AttritionMetrics]:
            - Enriched contingency table with calculated rate columns.
            - Strongly-typed AttritionMetrics container.
    """
    grand_total_employees = int(total_row[COL_TOTAL])
    grand_total_stayed = int(total_row[COL_STAYED])
    grand_total_left = int(total_row[COL_LEFT])

    analytics_df = ratings_df[[COL_STAYED, COL_LEFT, COL_TOTAL]].copy()

    # Proportion of workforce at each rating level
    analytics_df[COL_DISTRIBUTION_PCT] = (
        (analytics_df[COL_TOTAL] / grand_total_employees) * 100
    ).round(1)

    # Specific attrition rate: Left (Yes) / Total within satisfaction category
    analytics_df[COL_ATTRITION_RATE_PCT] = (
        (analytics_df[COL_LEFT] / analytics_df[COL_TOTAL]) * 100
    ).round(1)

    # Specific retention rate: Stayed (No) / Total within satisfaction category
    analytics_df[COL_RETENTION_RATE_PCT] = (
        (analytics_df[COL_STAYED] / analytics_df[COL_TOTAL]) * 100
    ).round(1)

    # Determine highest-frequency satisfaction level
    most_common_level = int(analytics_df[COL_TOTAL].idxmax())
    most_common_count = int(analytics_df.loc[most_common_level, COL_TOTAL])
    most_common_pct = float(
        analytics_df.loc[most_common_level, COL_DISTRIBUTION_PCT]
    )

    overall_attrition_pct = round((grand_total_left / grand_total_employees) * 100, 1)
    overall_retention_pct = round((grand_total_stayed / grand_total_employees) * 100, 1)

    metrics = AttritionMetrics(
        grand_total_employees=grand_total_employees,
        grand_total_stayed=grand_total_stayed,
        grand_total_left=grand_total_left,
        overall_attrition_pct=overall_attrition_pct,
        overall_retention_pct=overall_retention_pct,
        most_common_level=most_common_level,
        most_common_count=most_common_count,
        most_common_pct=most_common_pct,
    )

    return analytics_df, metrics


# =====================================================================
# 5. VISUALIZATION HELPERS & GENERATION
# =====================================================================

def _annotate_bar_chart(
    axis: plt.Axes,
    bars: Sequence,
    labels: Sequence[str],
    font_size: int = 9,
) -> None:
    """Reusable helper to place formatted text annotations centered above bars.

    Eliminates duplicate annotation logic between distribution and rate charts.
    """
    for bar, text in zip(bars, labels):
        height = bar.get_height()
        axis.annotate(
            text,
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=font_size,
            fontweight="bold",
        )


def _plot_distribution_panel(
    axis: plt.Axes,
    analytics_df: pd.DataFrame,
    x_positions: List[int],
    level_labels: List[str],
) -> None:
    """Render the employee distribution count bar chart (Subplot 1)."""
    counts = analytics_df[COL_TOTAL]
    percentages = analytics_df[COL_DISTRIBUTION_PCT]

    bars = axis.bar(
        x_positions,
        counts,
        color=COLOR_PRIMARY_BLUE,
        alpha=0.88,
        edgecolor=COLOR_EDGE_BLUE,
    )

    axis.set_title(
        "Employee Distribution by Job Satisfaction",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    axis.set_xlabel("Job Satisfaction Rating (1=Low, 4=High)", fontsize=10)
    axis.set_ylabel("Number of Employees", fontsize=10)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(level_labels)
    axis.set_ylim(0, max(counts) * 1.18)
    axis.grid(axis="y", linestyle="--", alpha=0.5, color=COLOR_GRID)

    annotation_texts = [
        f"{cnt:,}\n({pct:.1f}%)" for cnt, pct in zip(counts, percentages)
    ]
    _annotate_bar_chart(axis, bars, annotation_texts)


def _plot_attrition_rate_panel(
    axis: plt.Axes,
    analytics_df: pd.DataFrame,
    benchmark_rate: float,
    x_positions: List[int],
    level_labels: List[str],
) -> None:
    """Render the attrition rate by satisfaction level bar chart (Subplot 2)."""
    attrition_rates = analytics_df[COL_ATTRITION_RATE_PCT]

    bars = axis.bar(
        x_positions,
        attrition_rates,
        color=PALETTE_ATTRITION_LEVELS,
        alpha=0.88,
        edgecolor=COLOR_BENCHMARK_LINE,
    )

    axis.set_title(
        "Attrition Rate by Job Satisfaction Rating",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    axis.set_xlabel("Job Satisfaction Rating (1=Low, 4=High)", fontsize=10)
    axis.set_ylabel("Attrition Rate (%)", fontsize=10)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(level_labels)
    axis.set_ylim(0, max(attrition_rates) * 1.25)
    axis.grid(axis="y", linestyle="--", alpha=0.5, color=COLOR_GRID)

    # Reference benchmark dashed line for overall company baseline
    axis.axhline(
        benchmark_rate,
        color=COLOR_BENCHMARK_LINE,
        linestyle="--",
        linewidth=1.2,
        label=f"Company Average ({benchmark_rate}%)",
    )
    axis.legend(loc="upper right", frameon=True)

    annotation_texts = [f"{rate:.1f}%" for rate in attrition_rates]
    _annotate_bar_chart(axis, bars, annotation_texts)


def create_satisfaction_charts(
    analytics_df: pd.DataFrame,
    metrics: AttritionMetrics,
    output_path: Union[str, Path] = DEFAULT_OUTPUT_CHART_PATH,
) -> None:
    """Generate, display, and export dual-panel analytical visualizations.

    Subplot 1: Workforce Distribution across Satisfaction Ratings.
    Subplot 2: Attrition Rate (%) by Satisfaction Rating with Company Baseline.

    Args:
        analytics_df: DataFrame containing calculated metrics.
        metrics: AttritionMetrics instance containing summary statistics.
        output_path: Target path for saving the PNG image.
    """
    levels = [f"Rating {lvl}" for lvl in analytics_df.index]
    x_positions = list(range(len(levels)))
    save_path = Path(output_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=CHART_FIGSIZE)

    _plot_distribution_panel(ax1, analytics_df, x_positions, levels)
    _plot_attrition_rate_panel(
        ax2, analytics_df, metrics.overall_attrition_pct, x_positions, levels
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=CHART_DPI)
    plt.show()
    plt.close()


# =====================================================================
# 6. REPORTING & DISPLAY
# =====================================================================

def print_formatted_report(
    analytics_df: pd.DataFrame,
    metrics: AttritionMetrics,
) -> None:
    """Print the complete narrative analysis and formatted summary table.

    Args:
        analytics_df: Enriched contingency table DataFrame.
        metrics: AttritionMetrics container.
    """
    divider = "=" * 76
    print(divider)
    print("MIS 4320 TERM PROJECT: JOB SATISFACTION ATTRITION REPORT")
    print(divider)

    print("\nAttrition statistics:")
    print(
        f"{metrics.grand_total_stayed:,} ({metrics.overall_retention_pct}%) employees stayed, and "
        f"{metrics.grand_total_left:,} ({metrics.overall_attrition_pct}%) employees left the company."
    )

    print("\nAttrition by Job Satisfaction:")
    for level, row in analytics_df.iterrows():
        print(
            f"Job satisfaction {level}: "
            f"{int(row[COL_TOTAL])} employees ({row[COL_DISTRIBUTION_PCT]}%)"
        )

    print(
        f"\nThe most common job satisfaction rating is "
        f"{metrics.most_common_level} with "
        f"{metrics.most_common_count} employees "
        f"({metrics.most_common_pct}%). "
        f"The dataset codes job satisfaction from 1 to 4 where 1 is low "
        f"and 4 is the highest satisfaction rating."
    )

    print("\nContingency Table (Count of Attrition & Attrition Rates):")
    print("-" * 76)

    # Format output contingency table with grand total row
    table_display = analytics_df.copy()
    table_display.index = [f"Satisfaction {i}" for i in table_display.index]

    total_summary_series = pd.Series(
        {
            COL_STAYED: metrics.grand_total_stayed,
            COL_LEFT: metrics.grand_total_left,
            COL_TOTAL: metrics.grand_total_employees,
            COL_DISTRIBUTION_PCT: 100.0,
            COL_ATTRITION_RATE_PCT: metrics.overall_attrition_pct,
            COL_RETENTION_RATE_PCT: metrics.overall_retention_pct,
        },
        name="Grand Total",
    )
    table_display.loc["Grand Total"] = total_summary_series

    print(table_display.to_string())
    print(divider)


# =====================================================================
# 7. MAIN EXECUTION PIPELINE
# =====================================================================

def main() -> None:
    """Coordinate end-to-end ingestion, computation, reporting, and charting."""
    custom_argument = sys.argv[1] if len(sys.argv) > 1 else None

    try:
        resolved_path = resolve_dataset_path(custom_argument)
        if resolved_path:
            print(f"[INFO] Ingesting dataset file from: {resolved_path}")
        else:
            print("[INFO] No external file found. Using embedded fallback dataset.")

        ratings_data, totals = load_satisfaction_data(resolved_path)
        analyzed_data, summary_metrics = compute_satisfaction_analytics(
            ratings_data, totals
        )

        print_formatted_report(analyzed_data, summary_metrics)

        chart_destination = DEFAULT_OUTPUT_CHART_PATH
        create_satisfaction_charts(analyzed_data, summary_metrics, chart_destination)
        print(f"[INFO] High-resolution charts saved to: {chart_destination}")

    except (DataValidationError, FileNotFoundError) as err:
        print(f"[ERROR] {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
