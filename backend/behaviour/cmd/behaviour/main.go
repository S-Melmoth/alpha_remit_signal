package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"ai-hack/behaviour/internal/model"
	"ai-hack/behaviour/internal/platform"
	"ai-hack/behaviour/internal/store"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "behaviour:", err)
		os.Exit(1)
	}
}
func run() error {
	if len(os.Args) > 1 && os.Args[1] == "serve" {
		return platform.Serve("behaviour")
	}
	if len(os.Args) < 2 {
		return fmt.Errorf("usage: behaviour <migrate|seed|analyze|demo> [--as-of RFC3339] [--history-months 6] [--horizon-days 30]")
	}
	command := os.Args[1]
	if command != "migrate" && command != "seed" && command != "analyze" && command != "demo" && command != "seed-services" {
		return fmt.Errorf("unknown command %q", command)
	}
	flags := flag.NewFlagSet(command, flag.ContinueOnError)
	asOfText := flags.String("as-of", time.Now().UTC().Format(time.RFC3339), "exclusive knowledge cutoff, RFC3339 with timezone")
	history := flags.Int("history-months", 6, "previous full calendar months plus current month")
	horizon := flags.Int("horizon-days", 30, "future calendar days")
	if err := flags.Parse(os.Args[2:]); err != nil {
		return err
	}
	if flags.NArg() != 0 {
		return fmt.Errorf("unexpected positional arguments: %v", flags.Args())
	}
	asOf, err := time.Parse(time.RFC3339, *asOfText)
	if err != nil {
		return fmt.Errorf("invalid --as-of: %w", err)
	}
	if *history < 1 || *history > 120 || *horizon < 1 || *horizon > 366 {
		return fmt.Errorf("history-months must be 1..120 and horizon-days 1..366")
	}
	url := os.Getenv("DATABASE_URL")
	if url == "" {
		return fmt.Errorf("DATABASE_URL is required (see behaviour/README.md)")
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	ctx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()
	db, err := store.Open(ctx, url)
	if err != nil {
		return err
	}
	defer db.Close()
	if command == "migrate" || command == "demo" || command == "seed-services" {
		if err = store.Migrate(ctx, db); err != nil {
			return err
		}
		fmt.Fprintln(os.Stderr, "schema ready")
	}
	if command == "seed" || command == "demo" {
		if err = store.Seed(ctx, db); err != nil {
			return err
		}
		fmt.Fprintln(os.Stderr, "demo fixture ready")
	}
	if command == "analyze" || command == "demo" {
		windows, err := store.Analyze(ctx, db, model.Config{AsOf: asOf, HistoryMonths: *history, HorizonDays: *horizon})
		if err != nil {
			return err
		}
		encoder := json.NewEncoder(os.Stdout)
		encoder.SetIndent("", "  ")
		if err = encoder.Encode(windows); err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "saved %d candidates as of %s\n", len(windows), asOf.Format(time.RFC3339))
	}
	if command == "seed-services" {
		return store.SeedServices(ctx, db)
	}
	return nil
}
