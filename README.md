mactop-scraper is a lightweight client tool designed to collect, store, and analyze system metrics from mactop running in server mode on Apple Silicon Macs. It periodically fetches real-time performance data via the Prometheus-compatible HTTP endpoint exposed by mactop, enabling you to build custom datasets for monitoring, analysis, or visualization.

Typical use cases include:

- Historical tracking of CPU/GPU/memory/battery usage
- Local or remote metric storage for later analysis
- Integrating MacBook hardware telemetry into your own workflows

> Note: This project assumes you have `mactop` installed and running with its server functionality enabled.