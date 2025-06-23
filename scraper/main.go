package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/rs/zerolog/log"
)

// runScrapeTask collects metrics at a given interval for a total duration.
// It returns a slice of all collected metrics or an error if the context is canceled prematurely.
func runScrapeTask(ctx context.Context, totalDuration, interval time.Duration, fetcher *MactopMetricsFetcher) ([]*MactopMetrics, error) {
	// a new context that is automatically canceled after totalDuration.
	taskCtx, cancel := context.WithTimeout(ctx, totalDuration)
	defer cancel()

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	var fetchedResults []*MactopMetrics

	log.Info().
		Str("duration", totalDuration.String()).
		Str("interval", interval.String()).
		Msg("Starting timed scrape task.")

	// Perform an initial scrape immediately without waiting for the first tick.
	log.Debug().Msg("Performing initial scrape...")
	metrics, err := fetcher.FetchMetrics()
	if err != nil {
		log.Error().Err(err).Msg("Failed initial scrape")
	} else {
		fetchedResults = append(fetchedResults, metrics)
		log.Info().Interface("metrics", metrics).Msg("Initial scrape successful")
	}

	for {
		select {
		case <-taskCtx.Done():
			// The context's deadline was exceeded or the parent context was canceled.
			err := taskCtx.Err()
			if errors.Is(err, context.DeadlineExceeded) {
				// This is the expected "success" case: the timer ran out.
				log.Info().Msg("Scrape task finished after specified duration.")
				return fetchedResults, nil // Return the metrics collected so far.
			}
			// This means the parent context was canceled before the timeout.
			log.Warn().Err(err).Msg("Scrape task was canceled before completion.")
			return fetchedResults, err // Return collected data and the cancellation error.

		case <-ticker.C:
			log.Debug().Msg("Ticker ticked, fetching new metrics...")
			metrics, err := fetcher.FetchMetrics()
			if err != nil {
				log.Error().Err(err).Msg("Failed to fetch metrics during scrape")
				continue // Don't stop the whole task for one failed scrape.
			}
			fetchedResults = append(fetchedResults, metrics)
			log.Info().Interface("metrics", metrics).Msg("Successfully scraped metrics")
		}
	}
}

func main() {
	// command-line flags
	var (
		outputDir       string
		scrapeIntervalSec  int
		totalDurationSec   int
		mactopBaseURL   string
	)

	// Custom usage message for the help flag
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage of %s:\n", os.Args[0])
		fmt.Fprintln(os.Stderr, "This tool scrapes Mactop metrics at a specified interval for a given duration.")
		fmt.Fprintln(os.Stderr, "\nOptions:")
		flag.PrintDefaults() // Prints the default usage for all defined flags
		fmt.Fprintln(os.Stderr, "\nExample:")
		fmt.Fprintf(os.Stderr, "  %s --duration 60 --interval 5 --output-dir \"./metrics_data\" --mactop-url \"http://192.168.1.5:2211\"\n", os.Args[0])
	}

	flag.StringVar(&outputDir, "output-dir", "output", "Directory to save collected metrics.")
	flag.IntVar(&scrapeIntervalSec, "interval", 2, "Scrape interval in seconds.")
	flag.IntVar(&totalDurationSec, "duration", 10, "Total duration to run the scrape task in seconds.")
	flag.StringVar(&mactopBaseURL, "mactop-url", "localhost:2211", "Base URL for the Mactop metrics API.")
	flag.Parse()

	log.Info().
		Str("outputDir", outputDir).
		Int("scrapeInterval", scrapeIntervalSec).
		Int("totalDuration", totalDurationSec).
		Str("mactopBaseURL", mactopBaseURL).
		Msg("Configuration loaded.")

	// Create a fetcher instance.
	fetcher, err := NewMactopMetricsFetcher(mactopBaseURL)
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to create metrics fetcher")
	}

	// Convert seconds to time.Duration
	totalDuration := time.Duration(totalDurationSec) * time.Second
	scrapeInterval := time.Duration(scrapeIntervalSec) * time.Second

	// Run the task. This call will block until the task is complete (or canceled).
	// We use context.Background() because we don't need any special cancellation from a parent.
	collectedMetrics, err := runScrapeTask(context.Background(), totalDuration, scrapeInterval, fetcher)
	if err != nil && !errors.Is(err, context.Canceled) {
		log.Error().Err(err).Msg("Scrape task failed")
	}

	log.Info().Int("count", len(collectedMetrics)).Msg("Scraping complete.")

// --- Store the collected metrics into a file ---
	if len(collectedMetrics) > 0 {
		// Create a subdirectory named by running datetime
		// Format: YYYYMMDD_HHMMSS (e.g., 20250622_090908)
		timestampDir := time.Now().Format("20060102_150405")
		fullOutputDir := filepath.Join(outputDir, timestampDir)

		// Create the directory if it doesn't exist
		if err := os.MkdirAll(fullOutputDir, 0755); err != nil { // 0755: rwx for owner, rx for others
			log.Fatal().Err(err).Str("path", fullOutputDir).Msg("Failed to create output directory")
		}
		log.Info().Str("path", fullOutputDir).Msg("Output directory created.")

		outputFileName := filepath.Join(fullOutputDir, "mactop_metrics.json")

		jsonData, err := json.MarshalIndent(collectedMetrics, "", "  ")
		if err != nil {
			log.Fatal().Err(err).Msg("Failed to marshal results to JSON")
		}

		// Write the JSON data to the file.
		err = os.WriteFile(outputFileName, jsonData, 0644) // 0644 are file permissions
		if err != nil {
			log.Fatal().Err(err).Msgf("Failed to write metrics to file %s", outputFileName)
		}

		log.Info().Str("filename", outputFileName).Msg("Metrics successfully saved to file.")
		log.Info().Msgf("Collected Metrics Saved to %s", outputFileName)
	} else {
		log.Warn().Msg("No metrics were collected to save.")
	}
}



