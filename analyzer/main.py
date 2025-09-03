"""
Mactop Metrics Analyzer.

This script analyzes a JSON file containing a list of metrics from the mactop program.
It calculates key statistics for various metrics and can optionally generate line charts.

The script expects a JSON file where each object in the list has the following structure:

{
    "cpu_usage_percent": float,
    "gpu_freq_mhz": float,
    "gpu_usage_percent": float,
    "memory_gb": {
        "swap_total": float,
        "swap_used": float,
        "total": float,
        "used": float
    },
    "power_watts": {
        "cpu": float,
        "gpu": float,
        "total": float
    },
    "sample_time": "YYYY-MM-DDTHH:MM:SSZ"
}

The script outputs a human-readable report with statistics (min, max, mean, P50, P95, P99)
for key metrics and can optionally save line charts for each metric to a specified
output directory.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import re

# Try to import dateutil.parser for robust timestamp parsing
try:
    from dateutil import parser as dateparser
    HAS_DATEUTIL = True
except ImportError:
    HAS_DATEUTIL = False
    print("Warning: 'python-dateutil' not found. Timestamps will not be parsed for duration calculation. Install with 'pip install python-dateutil' for full functionality.")


METRICS_TO_ANALYZE = [
    ("CPU Usage (%)", "cpu_usage_percent"),
    ("GPU Usage (%)", "gpu_usage_percent"),
    ("Memory Usage (GB)", "memory_used_gb"),
    ("Memory Usage (%)", "memory_usage_percent"),
    ("Swap Usage (GB)", "swap_used_gb"),
    ("Swap Usage (%)", "swap_usage_percent"),
    ("CPU Power (Watts)", "cpu_power_watts"),
    ("GPU Power (Watts)", "gpu_power_watts"),
]


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for the script.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Analyze mactop metrics from a directory containing JSON part files.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Example usage:
  # Analyze metrics from a directory named '20250622_090908' inside './metrics_data'
  # and save results to the 'analysis_results' directory.
  python mactop_analyzer.py --input-dir ./metrics_data/20250622_090908 --output-dir ./analysis_results

  # Analyze and also create line charts
  python mactop_analyzer.py --input-dir ./metrics_data/20250622_090908 --output-dir ./analysis_results --plot
""",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Path to the input directory containing mactop metrics part files (e.g., mactop_metrics_part_0000.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./analysis_results"),
        help="Directory to save the analysis report and optional plots.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If set, generate and save line charts for each metric.",
    )
    return parser.parse_args()


def natural_sort_key(s):
    """Key for natural sorting of strings containing numbers."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def load_and_prepare_data(input_dir: Path) -> Dict[str, List[Any]]:
    """
    Loads and prepares the metrics data from all JSON part files in the given directory.

    This function reads all 'mactop_metrics_part_XXXX.json' files,
    extracts the relevant metrics, and calculates derived metrics.

    Args:
        input_dir (Path): The path to the input directory.

    Returns:
        Dict[str, List[Any]]: A dictionary where keys are metric names and values
                               are lists of the metric's values over time.

    Raises:
        FileNotFoundError: If the specified input_dir does not exist.
        ValueError: If no JSON part files are found or data is malformed.
    """
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Error: Input directory not found at {input_dir}")

    all_data_records: List[Dict[str, Any]] = []
    
    # Find all part files and sort them naturally
    json_files = sorted(input_dir.glob("mactop_metrics_part_*.json"), key=lambda p: natural_sort_key(p.name))

    if not json_files:
        raise ValueError(f"Error: No 'mactop_metrics_part_*.json' files found in {input_dir}")

    print(f"Found {len(json_files)} metric part files in {input_dir}. Loading data...")

    for file_path in json_files:
        try:
            with open(file_path, "r") as f:
                part_data = json.load(f)
                if not isinstance(part_data, list):
                    print(f"Warning: File {file_path} does not contain a JSON list. Skipping.")
                    continue
                all_data_records.extend(part_data)
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to decode JSON from {file_path}. Details: {e}. Skipping.")
        except Exception as e:
            print(f"Warning: An unexpected error occurred while reading {file_path}. Details: {e}. Skipping.")

    if not all_data_records:
        raise ValueError(f"Error: No valid metric records found in any JSON part files in {input_dir}.")

    metrics: Dict[str, List[Any]] = {
        "cpu_usage_percent": [],
        "gpu_usage_percent": [],
        "memory_used_gb": [],
        "memory_usage_percent": [],
        "swap_used_gb": [],
        "swap_usage_percent": [],
        "cpu_power_watts": [],
        "gpu_power_watts": [],
        "timestamps": [],
    }

    for record in all_data_records:
        try:
            metrics["cpu_usage_percent"].append(record["cpu_usage_percent"])
            metrics["gpu_usage_percent"].append(record["gpu_usage_percent"])

            mem = record["memory_gb"]
            metrics["memory_used_gb"].append(mem["used"])
            if mem["total"] > 0:
                metrics["memory_usage_percent"].append((mem["used"] / mem["total"]) * 100)
            else:
                metrics["memory_usage_percent"].append(0) # Or NaN, depending on desired behavior
            
            metrics["swap_used_gb"].append(mem["swap_used"])
            if mem["swap_total"] > 0:
                metrics["swap_usage_percent"].append((mem["swap_used"] / mem["swap_total"]) * 100)
            else:
                 metrics["swap_usage_percent"].append(0) # Or NaN

            power = record["power_watts"]
            metrics["cpu_power_watts"].append(power["cpu"])
            metrics["gpu_power_watts"].append(power["gpu"])

            metrics["timestamps"].append(record["sample_time"])
        except KeyError as e:
            print(f"Warning: A record is missing key '{e}'. Skipping this record.")
            # Depending on strictness, you might want to raise ValueError here.
            # For now, we'll just skip records with missing keys.
            continue
        except TypeError as e:
            print(f"Warning: Data type error in a record: {e}. Skipping this record.")
            continue


    if not metrics["timestamps"]:
        raise ValueError("Error: No valid metric records found after parsing all files.")

    return metrics


def calculate_statistics(data: List[float]) -> Dict[str, float]:
    """
    Calculates statistical values for a given list of numbers.

    Args:
        data (List[float]): A list of numerical data points.

    Returns:
        Dict[str, float]: A dictionary containing the min, max, mean,
                          P50, P95, and P99.
    """
    if not data:
        return {
            "min": 0,
            "max": 0,
            "mean": 0,
            "p50": 0,
            "p95": 0,
            "p99": 0,
        }

    np_data = np.array(data)
    stats = {
        "min": np.min(np_data),
        "max": np.max(np_data),
        "mean": np.mean(np_data),
        "p50": np.percentile(np_data, 50),
        "p95": np.percentile(np_data, 95),
        "p99": np.percentile(np_data, 99),
    }
    return stats


def generate_report(
    all_metrics_data: Dict[str, List[Any]]
) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """
    Generates a human-readable analysis report from the metrics data.

    Args:
        all_metrics_data (Dict[str, List[Any]]): The prepared metrics data.

    Returns:
        Tuple[str, Dict[str, Dict[str, float]]]: A tuple containing the
        formatted report string and a dictionary of the calculated statistics.
    """
    sample_count = len(all_metrics_data["timestamps"])
    
    duration_str = "N/A (dateutil not installed or timestamps not parsed)"
    if HAS_DATEUTIL and sample_count > 1:
        try:
            start_time = dateparser.parse(all_metrics_data["timestamps"][0])
            end_time = dateparser.parse(all_metrics_data["timestamps"][-1])
            duration_seconds = (end_time - start_time).total_seconds()
            duration_str = f"{duration_seconds:.2f} seconds"
        except Exception as e:
            print(f"Warning: Could not parse timestamps for duration calculation: {e}")


    report_lines = [
        "========================================",
        " Mactop Metrics Analysis Report",
        "========================================",
        f"Total Samples: {sample_count}",
        f"Total Duration: {duration_str}\n",
    ]

    all_stats = {}
    for friendly_name, key in METRICS_TO_ANALYZE:
        metric_data = all_metrics_data.get(key, [])
        stats = calculate_statistics(metric_data)
        all_stats[key] = stats

        report_lines.append(f"--- {friendly_name} ---")
        report_lines.append(f"  Min: {stats['min']:.2f}")
        report_lines.append(f"  Max: {stats['max']:.2f}")
        report_lines.append(f"  Mean: {stats['mean']:.2f}")
        report_lines.append(f"  P50 (Median): {stats['p50']:.2f}")
        report_lines.append(f"  P95: {stats['p95']:.2f}")
        report_lines.append(f"  P99: {stats['p99']:.2f}\n")

    return "\n".join(report_lines), all_stats


def create_and_save_plots_seaborn_optimized(
    all_metrics_data: Dict[str, List[Any]],
    output_dir: Path,
) -> None:
    """
    Creates and saves enhanced, readable line charts for each metric using seaborn.

    For large datasets, markers are decimated to avoid clutter. Each plot includes
    the original data line and a smoothed trendline.

    Args:
        all_metrics_data (Dict[str, List[Any]]): The prepared metrics data.
        output_dir (Path): The directory where plots will be saved.
    """
    print(f"Generating and saving plots to {output_dir}...")
    
    # Set the aesthetic style of the plots
    sns.set_theme(style="whitegrid", palette="muted")

    for friendly_name, key in METRICS_TO_ANALYZE:
        metric_data = all_metrics_data.get(key)
        
        if not metric_data or len(metric_data) < 2:
            num_points = len(metric_data) if metric_data else 0
            print(f"Skipping plot for '{friendly_name}' due to insufficient data (found {num_points} points).")
            continue

        num_points = len(metric_data)
        print(f"  - Processing '{friendly_name}' with {num_points} data points...")

        # Create a pandas DataFrame
        df = pd.DataFrame({
            "Sample Index": range(num_points),
            friendly_name: metric_data
        })

        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 7))

        # 1. Plot the original data with no markers
        sns.lineplot(
            data=df,
            x="Sample Index",
            y=friendly_name,
            ax=ax,
            label="Original Data",
            color=sns.color_palette("muted")[0]
        )

        # 2. Add a smoothed trendline using LOWESS regression
        try:
            sns.regplot(
                data=df,
                x="Sample Index",
                y=friendly_name,
                ax=ax,
                lowess=True,
                scatter=False,
                label="Smoothed Trend",
                color=sns.color_palette("muted")[1],
                line_kws={'linewidth': 2.5},
                ci=None # Disable confidence interval for a cleaner look
            )
        except Exception as e:
            print(f"  - Could not generate smoothed trend for '{friendly_name}': {e}")

        # --- Customize the plot ---
        ax.set_title(f"{friendly_name} Over Time", fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel("Sample Index", fontsize=14)
        ax.set_ylabel(friendly_name, fontsize=14)
        
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xticks(rotation=0)
        
        ax.legend(title="Legend", frameon=True, fontsize=12)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        fig.tight_layout()

        # --- Sanitize filename and save the plot ---
        filename_key = key.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "percent")
        plot_path = output_dir / f"{filename_key}_chart_seaborn.png"
        
        try:
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"  - Saved {plot_path.name}")
        except Exception as e:
            print(f"  - Failed to save plot for '{friendly_name}': {e}")
        finally:
            plt.close(fig)



def main() -> None:
    """
    Main function to run the mactop analysis script.
    """
    args = parse_arguments()

    try:
        # Create output directory if it doesn't exist
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load and prepare data from the directory
        print(f"Loading data from directory {args.input_dir}...")
        metrics_data = load_and_prepare_data(args.input_dir)

        # 2. Generate report
        print("Generating analysis report...")
        report_str, _ = generate_report(metrics_data)

        # Save report to a file
        report_path = args.output_dir / "analysis_report.txt"
        with open(report_path, "w") as f:
            f.write(report_str)
        print(f"Analysis report saved to {report_path}")

        # Print report to console
        print("\n" + report_str)

        # 3. Optionally create and save plots
        if args.plot:
            create_and_save_plots_seaborn_optimized(metrics_data, args.output_dir)

        print("Analysis complete.")

    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"\nAn error occurred:\n{e}")
        exit(1)


if __name__ == "__main__":
    main()
