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

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

const maxRecordsPerFile = 1024 // Define the maximum records per file
const filenameIndexWidth = 4   // Width for zero-padding file index (e.g., 0001, 0123)

// runScrapeTask collects metrics at a given interval for a total duration and saves them incrementally.
// It returns an error if the task is interrupted prematurely.
func runScrapeTask(ctx context.Context, totalDuration, interval time.Duration, fetcher *MactopMetricsFetcher, fullOutputDir string) error {
	// a new context that is automatically canceled after totalDuration.
	taskCtx, cancel := context.WithTimeout(ctx, totalDuration)
	defer cancel()

	// Calculate how many records we will log based on the total duration and interval.
	// So we can preallocate the buffer.
	totalRecords := int((totalDuration + interval - 1) / interval) // Round up to ensure we cover the entire duration
	bufferSize := (totalRecords + maxRecordsPerFile - 1) / maxRecordsPerFile // Round up to the nearest maxRecordsPerFile

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	var (
		currentPartMetrics = make([]*MactopMetrics, 0, bufferSize) // Buffer for metrics in the current part file
		fileIndex          int              // Index for the current part file
		recordCount        int              // Count of records in the current buffer
	)

	log.Info().
		Str("duration", totalDuration.String()).
		Str("interval", interval.String()).
		Str("outputDir", fullOutputDir).
		Msg("Starting timed scrape task with incremental saving.")

	// Helper function to write the current part to a file
	writeCurrentPart := func() {
		if len(currentPartMetrics) == 0 {
			return // Nothing to write
		}

		outputFileName := filepath.Join(fullOutputDir, fmt.Sprintf("mactop_metrics_part_%0*d.json", filenameIndexWidth, fileIndex))

		jsonData, err := json.MarshalIndent(currentPartMetrics, "", "  ")
		if err != nil {
			log.Error().Err(err).Msg("Failed to marshal results to JSON for part file")
			return // Don't fatally exit, try to continue scraping
		}

		err = os.WriteFile(outputFileName, jsonData, 0644)
		if err != nil {
			log.Error().Err(err).Msgf("Failed to write metrics to file %s", outputFileName)
			return // Don't fatally exit
		}

		log.Info().Str("filename", outputFileName).Int("records", len(currentPartMetrics)).Msg("Metrics part successfully saved to file.")
		currentPartMetrics = nil // Reset for the next part
		fileIndex++              // Increment for the next file
	}

	// Perform an initial scrape immediately without waiting for the first tick.
	log.Debug().Msg("Performing initial scrape...")
	metrics, err := fetcher.FetchMetrics()
	if err != nil {
		log.Error().Err(err).Msg("Failed initial scrape")
	} else {
		currentPartMetrics = append(currentPartMetrics, metrics)
		recordCount++
		log.Info().Interface("metrics", metrics).Msg("Initial scrape successful")
	}

	// Check if initial scrape filled a part
	if recordCount >= maxRecordsPerFile {
		writeCurrentPart()
		recordCount = 0
	}

	for {
		select {
		case <-taskCtx.Done():
			// The context's deadline was exceeded or the parent context was canceled.
			err := taskCtx.Err()
			log.Info().Int("remaining_records", len(currentPartMetrics)).Msg("Scrape task finished, writing remaining metrics.")
			writeCurrentPart() // Write any remaining buffered metrics

			if errors.Is(err, context.DeadlineExceeded) {
				// This is the expected "success" case: the timer ran out.
				log.Info().Msg("Scrape task finished after specified duration.")
				return nil // Return nil error for successful completion
			}
			// This means the parent context was canceled before the timeout.
			log.Warn().Err(err).Msg("Scrape task was canceled before completion.")
			return err // Return the cancellation error.

		case <-ticker.C:
			log.Debug().Msg("Ticker ticked, fetching new metrics...")
			metrics, err := fetcher.FetchMetrics()
			if err != nil {
				log.Error().Err(err).Msg("Failed to fetch metrics during scrape")
				continue // Don't stop the whole task for one failed scrape.
			}

			currentPartMetrics = append(currentPartMetrics, metrics)
			recordCount++
			log.Info().Interface("metrics", metrics).Msg("Successfully scraped metrics")

			if recordCount >= maxRecordsPerFile {
				writeCurrentPart()
				recordCount = 0 // Reset record count for the new file
			}
		}
	}
}

func main() {
	// Record the start time
	t := time.Now()

	// command-line flags
	var (
		outputDir          string
		scrapeIntervalSec  int
		totalDurationSec   int
		mactopBaseURL      string

		logLevel string
		logOutputDir string
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

	flag.StringVar(&outputDir, "output-dir", "output", "Directory to save collected metrics. Default is 'output'.")
	flag.IntVar(&scrapeIntervalSec, "interval", 2, "Scrape interval in seconds. 2 seconds by default.")
	flag.IntVar(&totalDurationSec, "duration", 60, "Total duration to run the scrape task in seconds. 60 seconds by default.")
	flag.StringVar(&mactopBaseURL, "mactop-url", "http://localhost:2211", "Base URL for the Mactop metrics API. Default is 'http://localhost:2211'.")
	flag.StringVar(&logLevel, "log-level", "warn", "Set the logging level (trace, debug, info, warning, error, fatal, panic). Default is 'warning'.")
	flag.StringVar(&logOutputDir, "log-output-dir", "", "Directory to write logs to. If empty, logs will be written to stderr.")
	flag.Parse()

	// Override log level if specified in the command line arguments
	logLevels := map[string]zerolog.Level{
		"":      zerolog.InfoLevel, // Default log level
		"trace": zerolog.TraceLevel,
		"debug": zerolog.DebugLevel,
		"info":  zerolog.InfoLevel,
		"warn":  zerolog.WarnLevel,
		"error": zerolog.ErrorLevel,
		"fatal": zerolog.FatalLevel,
		"panic": zerolog.PanicLevel,
	}

	if level, exists := logLevels[logLevel]; exists {
		zerolog.SetGlobalLevel(level)
	} else {
		log.Error().Msgf("[main] Unsupported log level: %s", logLevel)
		return
	}

	// used to format the time in the output file name
	// We do not use RFC3339 format because it contains colons, which are not allowed in Windows file names.
	outputFileTimeFormat := "20060102150405"

	// Log to file if specified
	if logOutputDir != "" {
		logFilePath := fmt.Sprintf("%s/log_%s.log", logOutputDir, t.Format(outputFileTimeFormat))
		fileWriter, err := os.Create(logFilePath)
		if err != nil {
			log.Err(err).Msgf("[main] Failed to create log file: %s", logFilePath)
			return
		}
		log.Info().Msgf("[main] Log to file is enabled, I will write logs to %s", logFilePath)
		log.Logger = log.Output(fileWriter)
	}

	log.Info().
		Str("outputDir", outputDir).
		Int("scrapeInterval", scrapeIntervalSec).
		Int("totalDuration", totalDurationSec).
		Str("mactopBaseURL", mactopBaseURL).
		Str("logLevel", logLevel).
		Str("logOutputFile", logOutputDir).
		Msg("Configuration loaded.")

	// Create a fetcher instance.
	fetcher, err := NewMactopMetricsFetcher(mactopBaseURL)
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to create metrics fetcher")
	}

	// Convert seconds to time.Duration
	totalDuration := time.Duration(totalDurationSec) * time.Second
	scrapeInterval := time.Duration(scrapeIntervalSec) * time.Second

	// Create a subdirectory named by running datetime
	// Format:YYYYMMDD_HHMMSS (e.g., 20250622_090908)
	timestampDir := t.Format("20060102_150405")
	fullOutputDir := filepath.Join(outputDir, timestampDir)

	// Create the directory if it doesn't exist
	if err := os.MkdirAll(fullOutputDir, 0755); err != nil { // 0755: rwx for owner, rx for others
		log.Fatal().Err(err).Str("path", fullOutputDir).Msg("Failed to create output directory")
	}
	log.Info().Str("path", fullOutputDir).Msg("Output directory created.")

	// Run the task. This call will block until the task is complete (or canceled).
	// We use context.Background() because we don't need any special cancellation from a parent.
	err = runScrapeTask(context.Background(), totalDuration, scrapeInterval, fetcher, fullOutputDir)
	if err != nil && !errors.Is(err, context.DeadlineExceeded) { // DeadlineExceeded is a "successful" end
		log.Error().Err(err).Msg("Scrape task failed or was canceled prematurely.")
	} else {
		log.Info().Msg("Scrape task completed.")
	}

	log.Info().Msgf("Collected Metrics Saved to directory: %s", fullOutputDir)
}
