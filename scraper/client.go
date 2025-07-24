package main

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/cloudwego/hertz/pkg/app/client"
	"github.com/cloudwego/hertz/pkg/protocol"
	"github.com/cloudwego/hertz/pkg/protocol/consts"
	"github.com/prometheus/common/expfmt"
	"github.com/rs/zerolog/log"
)

type MactopMetricsFetcher struct {
	c       *client.Client
	baseUrl string
}

func NewMactopMetricsFetcher(baseUrl string) (*MactopMetricsFetcher, error) {
	c, err := client.NewClient()
	if err != nil {
		log.Err(err).Msg("error during initialization of client")
		return nil, err
	}
	return &MactopMetricsFetcher{
		c:       c,
		baseUrl: baseUrl,
	}, nil
}

// FetchMetrics fetches metrics from the mactop prometheus server.
// The server returns plain text which includes metrics data.
// example of mactop response:
//  # HELP mactop_cpu_usage_percent Current Total CPU usage percentage
//  # TYPE mactop_cpu_usage_percent gauge
//  mactop_cpu_usage_percent 11.915085817524842
//  # HELP mactop_gpu_freq_mhz Current GPU frequency in MHz
//  # TYPE mactop_gpu_freq_mhz gauge
//  mactop_gpu_freq_mhz 482
//  # HELP mactop_gpu_usage_percent Current GPU usage percentage
//  # TYPE mactop_gpu_usage_percent gauge
//  mactop_gpu_usage_percent 12
//  # HELP mactop_memory_gb Memory usage in GB
//  # TYPE mactop_memory_gb gauge
//  mactop_memory_gb{type="swap_total"} 0
//  mactop_memory_gb{type="swap_used"} 0
//  mactop_memory_gb{type="total"} 16
//  mactop_memory_gb{type="used"} 10.139328002929688
//  # HELP mactop_power_watts Current power usage in watts
//  # TYPE mactop_power_watts gauge
//  mactop_power_watts{component="cpu"} 0.6360520000000001
//  mactop_power_watts{component="gpu"} 0.119137
//  mactop_power_watts{component="total"} 0.755189
func (f *MactopMetricsFetcher) FetchMetrics() (*MactopMetrics, error) {
	req, resp := protocol.AcquireRequest(), protocol.AcquireResponse()
	defer func() {
		protocol.ReleaseRequest(req)
		protocol.ReleaseResponse(resp)
	}()

	// Construct the full URL for the metrics endpoint, ensuring no double slashes.
	url := fmt.Sprintf("%s/metrics", strings.TrimSuffix(f.baseUrl, "/"))
	req.SetRequestURI(url)
	req.SetMethod(consts.MethodGet)

	log.Info().Str("url", url).Msg("Fetching metrics from mactop")

	if err := f.c.Do(context.Background(), req, resp); err != nil {
		log.Err(err).Str("url", url).Msg("Failed to fetch metrics")
		return nil, fmt.Errorf("http request to %s failed: %w", url, err)
	}

	if resp.StatusCode() != consts.StatusOK {
		err := fmt.Errorf("failed to fetch metrics, status code: %d, body: %s", resp.StatusCode(), string(resp.Body()))
		log.Err(err).Msg("Received non-200 status from mactop")
		return nil, err
	}

	return parseMactopMetrics(resp.Body())
}

// parseMactopMetrics parses the prometheus text format into a MactopMetrics struct.
func parseMactopMetrics(data []byte) (*MactopMetrics, error) {
	var parser expfmt.TextParser
	metricFamilies, err := parser.TextToMetricFamilies(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("failed to parse prometheus metrics: %w", err)
	}

	metrics := &MactopMetrics{}

	// Helper to get a single value from a metric without labels
	getSingleGauge := func(name string) (float64, bool) {
		mf, ok := metricFamilies[name]
		if !ok || len(mf.GetMetric()) == 0 {
			return 0, false
		}
		return mf.GetMetric()[0].GetGauge().GetValue(), true
	}

	if val, ok := getSingleGauge("mactop_cpu_usage_percent"); ok {
		metrics.CPUUsagePercent = val
	}
	if val, ok := getSingleGauge("mactop_gpu_freq_mhz"); ok {
		metrics.GPUFreqMHz = val
	}
	if val, ok := getSingleGauge("mactop_gpu_usage_percent"); ok {
		metrics.GPUUsagePercent = val
	}

	// Parse metrics with labels
	if mf, ok := metricFamilies["mactop_memory_gb"]; ok {
		for _, m := range mf.GetMetric() {
			for _, label := range m.GetLabel() {
				if label.GetName() == "type" {
					val := m.GetGauge().GetValue()
					switch label.GetValue() {
					case "swap_total":
						metrics.MemoryGB.SwapTotal = val
					case "swap_used":
						metrics.MemoryGB.SwapUsed = val
					case "total":
						metrics.MemoryGB.Total = val
					case "used":
						metrics.MemoryGB.Used = val
					}
				}
			}
		}
	}

	if mf, ok := metricFamilies["mactop_power_watts"]; ok {
		for _, m := range mf.GetMetric() {
			for _, label := range m.GetLabel() {
				if label.GetName() == "component" {
					val := m.GetGauge().GetValue()
					switch label.GetValue() {
					case "cpu":
						metrics.PowerWatts.CPU = val
					case "gpu":
						metrics.PowerWatts.GPU = val
					case "total":
						metrics.PowerWatts.Total = val
					}
				}
			}
		}
	}

	metrics.SampleTime = time.Now()

	return metrics, nil
}
