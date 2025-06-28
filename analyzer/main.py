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
from matplotlib.ticker import MaxNLocator

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
        description="Analyze mactop metrics from a JSON file.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Example usage:
  # Generate a report from metrics.json and save it to the 'output' directory
  python mactop_analyzer.py --file-path ./metrics.json --output-dir ./output

  # Generate a report and also create line charts for each metric
  python mactop_analyzer.py --file-path ./metrics.json --output-dir ./output --plot
""",
    )
    parser.add_argument(
        "--file-path",
        type=Path,
        required=True,
        help="Path to the input JSON file containing mactop metrics.",
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


def load_and_prepare_data(file_path: Path) -> Dict[str, List[Any]]:
    """
    Loads and prepares the metrics data from the given JSON file.

    This function reads the JSON file, extracts the relevant metrics, and calculates
    derived metrics such as memory and swap usage percentages.

    Args:
        file_path (Path): The path to the input JSON file.

    Returns:
        Dict[str, List[Any]]: A dictionary where keys are metric names and values
                               are lists of the metric's values over time.

    Raises:
        FileNotFoundError: If the specified file_path does not exist.
        ValueError: If the JSON file is malformed or the data is not in the
                    expected format.
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"Error: Input file not found at {file_path}")

    with open(file_path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error: Failed to decode JSON from {file_path}. Details: {e}")

    if not isinstance(data, list) or not data:
        raise ValueError("Error: JSON data must be a non-empty list of metric objects.")

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

    for record in data:
        try:
            metrics["cpu_usage_percent"].append(record["cpu_usage_percent"])
            metrics["gpu_usage_percent"].append(record["gpu_usage_percent"])

            mem = record["memory_gb"]
            metrics["memory_used_gb"].append(mem["used"])
            if mem["total"] > 0:
                metrics["memory_usage_percent"].append((mem["used"] / mem["total"]) * 100)
            else:
                metrics["memory_usage_percent"].append(0)
            
            metrics["swap_used_gb"].append(mem["swap_used"])
            if mem["swap_total"] > 0:
                metrics["swap_usage_percent"].append((mem["swap_used"] / mem["swap_total"]) * 100)
            else:
                 metrics["swap_usage_percent"].append(0)

            power = record["power_watts"]
            metrics["cpu_power_watts"].append(power["cpu"])
            metrics["gpu_power_watts"].append(power["gpu"])

            metrics["timestamps"].append(record["sample_time"])
        except KeyError as e:
            raise ValueError(f"Error: A record in the JSON is missing the key: {e}")

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
    
    # Assuming timestamps are sorted, which they should be from a log
    try:
        from dateutil import parser
        start_time = parser.parse(all_metrics_data["timestamps"][0])
        end_time = parser.parse(all_metrics_data["timestamps"][-1])
        duration_seconds = (end_time - start_time).total_seconds()
    except (ImportError, ValueError):
        # Fallback if dateutil is not installed or parsing fails
        duration_seconds = sample_count - 1 if sample_count > 1 else 0


    report_lines = [
        "========================================",
        " Mactop Metrics Analysis Report",
        "========================================",
        f"Total Samples: {sample_count}",
        f"Total Duration: {duration_seconds:.2f} seconds\n",
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


def create_and_save_plots(
    all_metrics_data: Dict[str, List[Any]],
    output_dir: Path,
) -> None:
    """
    Creates and saves line charts for each metric.

    Args:
        all_metrics_data (Dict[str, List[Any]]): The prepared metrics data.
        output_dir (Path): The directory where plots will be saved.
    """
    print(f"Generating and saving plots to {output_dir}...")
    sample_indices = range(len(all_metrics_data["timestamps"]))

    for friendly_name, key in METRICS_TO_ANALYZE:
        metric_data = all_metrics_data.get(key, [])
        if not metric_data:
            print(f"Skipping plot for '{friendly_name}' due to no data.")
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(sample_indices, metric_data, marker=".", linestyle="-", markersize=4)

        ax.set_title(f"{friendly_name} Over Time", fontsize=16)
        ax.set_xlabel("Sample Index", fontsize=12)
        ax.set_ylabel(friendly_name, fontsize=12)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        
        # Ensure x-axis has integer ticks
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        fig.tight_layout()

        # Sanitize filename
        filename_key = key.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "percent")
        plot_path = output_dir / f"{filename_key}_chart.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  - Saved {plot_path.name}")


def main() -> None:
    """
    Main function to run the mactop analysis script.
    """
    args = parse_arguments()

    try:
        # Create output directory if it doesn't exist
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load and prepare data
        print(f"Loading data from {args.file_path}...")
        metrics_data = load_and_prepare_data(args.file_path)

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
            create_and_save_plots(metrics_data, args.output_dir)

        print("Analysis complete.")

    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"\nAn error occurred:\n{e}")
        exit(1)


if __name__ == "__main__":
    main()
