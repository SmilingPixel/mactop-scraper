package main

import "time"

// MemoryMetrics stores memory usage details in GB.
type MemoryMetrics struct {
	SwapTotal float64 `json:"swap_total"`
	SwapUsed  float64 `json:"swap_used"`
	Total     float64 `json:"total"`
	Used      float64 `json:"used"`
}

// PowerMetrics stores power usage details in Watts by component.
type PowerMetrics struct {
	CPU   float64 `json:"cpu"`
	GPU   float64 `json:"gpu"`
	Total float64 `json:"total"`
}

// MactopMetrics is the main struct that stores all metrics returned by mactop.
type MactopMetrics struct {
	CPUUsagePercent float64       `json:"cpu_usage_percent"`
	GPUFreqMHz      float64       `json:"gpu_freq_mhz"`
	GPUUsagePercent float64       `json:"gpu_usage_percent"`
	MemoryGB        MemoryMetrics `json:"memory_gb"`
	PowerWatts      PowerMetrics  `json:"power_watts"`
	SampleTime            time.Time     `json:"sample_time"`
}
