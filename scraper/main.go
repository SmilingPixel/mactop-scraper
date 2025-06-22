package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
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

	mactopBaseURL := "localhost:2211"

	// Create a fetcher instance.
	fetcher, err := NewMactopMetricsFetcher(mactopBaseURL)
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to create metrics fetcher")
	}

	// Define the parameters for our scrape task.
	totalTaskDuration := 10 * time.Second // Run the task for 10 seconds.
	scrapeInterval := 2 * time.Second   // Scrape every 2 seconds.

	// Run the task. This call will block until the task is complete (or canceled).
	// We use context.Background() because we don't need any special cancellation from a parent.
	collectedMetrics, err := runScrapeTask(context.Background(), totalTaskDuration, scrapeInterval, fetcher)
	if err != nil && !errors.Is(err, context.Canceled) {
		log.Error().Err(err).Msg("Scrape task failed")
	}

	log.Info().Int("count", len(collectedMetrics)).Msg("Scraping complete.")

	// --- Store the collected metrics into a file ---
	if len(collectedMetrics) > 0 {
		outputFileName := "mactop_metrics.json"

		jsonData, err := json.MarshalIndent(collectedMetrics, "", "  ")
		if err != nil {
			log.Fatal().Err(err).Msg("Failed to marshal results to JSON")
		}

		// Write the JSON data to the file.
		// creates the file if it doesn't exist, or truncates it if it does.
		err = os.WriteFile(outputFileName, jsonData, 0644) // 0644 are file permissions (read/write for owner, read for others)
		if err != nil {
			log.Fatal().Err(err).Msgf("Failed to write metrics to file %s", outputFileName)
		}

		log.Info().Str("filename", outputFileName).Msg("Metrics successfully saved to file.")
		fmt.Printf("\n--- Collected Metrics Saved to %s ---\n", outputFileName)
	} else {
		log.Warn().Msg("No metrics were collected to save.")
	}
}



