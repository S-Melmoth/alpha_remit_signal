package store

import (
	"context"
	"database/sql"
	_ "embed"
	"encoding/json"
	"fmt"

	"ai-hack/behaviour/internal/model"

	_ "github.com/lib/pq"
)

//go:embed migrations/001_init.sql
var schema string

//go:embed migrations/002_services.sql
var servicesSchema string

func Open(ctx context.Context, url string) (*sql.DB, error) {
	db, err := sql.Open("postgres", url)
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(4)
	if err := db.PingContext(ctx); err != nil {
		db.Close()
		return nil, fmt.Errorf("connect to PostgreSQL: %w", err)
	}
	return db, nil
}

func Migrate(ctx context.Context, db *sql.DB) error {
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	if _, err = tx.ExecContext(ctx, `SELECT pg_advisory_xact_lock(2026090501)`); err != nil {
		return err
	}
	if _, err = tx.ExecContext(ctx, schema+"\n"+servicesSchema); err != nil {
		return err
	}
	return tx.Commit()
}

// Analyze reads one consistent database snapshot and atomically replaces the
// entire result for this as-of/version. Older as-of snapshots remain available.
func Analyze(ctx context.Context, db *sql.DB, cfg model.Config) ([]model.Window, error) {
	tx, err := db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelSerializable})
	if err != nil {
		return nil, err
	}
	defer tx.Rollback()
	if _, err = tx.ExecContext(ctx, `SELECT pg_advisory_xact_lock(2026090502)`); err != nil {
		return nil, err
	}
	rows, err := tx.QueryContext(ctx, `SELECT user_id, timezone FROM clients ORDER BY user_id`)
	if err != nil {
		return nil, err
	}
	var clients []model.Client
	for rows.Next() {
		var c model.Client
		if err = rows.Scan(&c.UserID, &c.Timezone); err != nil {
			rows.Close()
			return nil, err
		}
		clients = append(clients, c)
	}
	err = rows.Err()
	rows.Close()
	if err != nil {
		return nil, err
	}
	windows := make([]model.Window, 0)
	for _, c := range clients {
		rows, err = tx.QueryContext(ctx, `SELECT id, kind, status, occurred_at, recorded_at, amount_minor, source_currency,
		 COALESCE(destination_currency,''), COALESCE(corridor,'') FROM transfers
		 WHERE user_id=$1 AND occurred_at < $2 AND recorded_at < $2 AND status='completed'
		 AND occurred_at >= $3 ORDER BY occurred_at, id`, c.UserID, cfg.AsOf, cfg.AsOf.AddDate(0, -cfg.HistoryMonths-1, 0))
		if err != nil {
			return nil, err
		}
		for rows.Next() {
			var t model.Transfer
			if err = rows.Scan(&t.ID, &t.Kind, &t.Status, &t.OccurredAt, &t.RecordedAt, &t.AmountMinor, &t.SourceCurrency, &t.DestinationCurrency, &t.Corridor); err != nil {
				rows.Close()
				return nil, err
			}
			c.Transfers = append(c.Transfers, t)
		}
		err = rows.Err()
		rows.Close()
		if err != nil {
			return nil, err
		}
		result, e := model.Analyze(c, cfg)
		if e != nil {
			return nil, e
		}
		windows = append(windows, result...)
	}
	if _, err = tx.ExecContext(ctx, `DELETE FROM behaviour_push WHERE as_of=$1 AND model_version=$2`, cfg.AsOf, model.Version); err != nil {
		return nil, err
	}
	for _, w := range windows {
		metadata, e := json.Marshal(w.Metadata)
		if e != nil {
			return nil, e
		}
		_, err = tx.ExecContext(ctx, `INSERT INTO behaviour_push
		(user_id,as_of,model_version,source_currency,destination_currency,corridor,income_day,cycle_date,
		 window_start,window_end,timezone,typical_amount_minor,amount_p25_minor,amount_p75_minor,sample_count,support_ratio,metadata)
		 VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17::jsonb)`,
			w.UserID, w.AsOf, w.ModelVersion, w.SourceCurrency, w.DestinationCurrency, w.Corridor, w.IncomeDay, w.CycleDate,
			w.WindowStart, w.WindowEnd, w.Timezone, w.TypicalAmountMinor, w.AmountP25Minor, w.AmountP75Minor, w.SampleCount, w.SupportRatio, string(metadata))
		if err != nil {
			return nil, err
		}
	}
	if err = tx.Commit(); err != nil {
		return nil, err
	}
	return windows, nil
}
