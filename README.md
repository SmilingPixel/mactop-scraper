# Introduction
mactop-scraper is a lightweight client tool designed to collect, store, and analyze system metrics from mactop running in server mode on Apple Silicon Macs. It periodically fetches real-time performance data via the Prometheus-compatible HTTP endpoint exposed by mactop, enabling you to build custom datasets for monitoring, analysis, or visualization.

Typical use cases include:

- Historical tracking of CPU/GPU/memory/battery usage
- Local or remote metric storage for later analysis
- Integrating MacBook hardware telemetry into your own workflows

> Note: This project assumes you have `mactop` installed and running with its server functionality enabled.


# Features

- Real-time Metric Collection: Periodically fetches performance data from mactop.

- Prometheus-compatible Endpoint: Integrates seamlessly with mactop's exposed HTTP endpoint.

- Customizable Data Storage: Supports output to JSON files.

- Flexible Data Analysis: Generates summaries of collected metrics.

- Lightweight and Efficient: Designed for minimal system overhead


# Prerequisites

- mactop: You must have [mactop](https://github.com/context-labs/mactop) installed and running in server mode on your Apple Silicon Mac.

- Python: mactop-analyzer requires Python 3.13, and we strongly recommend [uv](https://github.com/astral-sh/uv) to manage environment and packages.

- Golang (Optional): We have released binaries for scraper, you can download and run directly. Alternatively, you can build from source code, and go >= 1.24 is recommended.


# Usage

## Scraper

Scraper is a command-line tool that collects metrics from a mactop HTTP endpoint and saves them as JSON files.

If you’ve downloaded the released binary and named it scraper, you can run it using the following command:

```bash
./scraper --duration 60 --interval 5 --output-dir "./metrics_data" --mactop-url "http://192.168.1.5:2211"
```

To see all available command-line options, run:

```bash
./scraper --help
```

## Analyzer

Analyzer is a Python script that processes the collected JSON files and generates metric summaries.

Before running the analyzer, make sure the required Python packages are installed. We recommend using `uv`:

```bash
uv sync
```

Then, run the analyzer with:

```bash
uv run main.py --input-dir "./metrics_data"
```

This will generate a summary report of the collected metrics.
To include visualizations, add the `--plot` option:

```bash
uv run main.py --input-dir "./metrics_data" --plot
```


# License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


# Contributing

Contributions are welcome! Please feel free to submit issues or pull requests. For major changes, please open an issue first to discuss what you would like to change.
