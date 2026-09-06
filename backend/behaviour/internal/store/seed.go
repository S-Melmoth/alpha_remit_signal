package store

import (
	"context"
	"database/sql"
	"fmt"
	"time"
)

// Seed adds a deterministic March–August 2026 fixture. Stable IDs make reruns safe.
func Seed(ctx context.Context, db *sql.DB) error {
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	for _, user := range []string{"demo-regular", "demo-sparse", "demo-irregular"} {
		if _, err = tx.ExecContext(ctx, `INSERT INTO clients(user_id,timezone) VALUES($1,'Europe/Moscow') ON CONFLICT DO NOTHING`, user); err != nil {
			return err
		}
	}
	loc, err := time.LoadLocation("Europe/Moscow")
	if err != nil {
		return err
	}
	insert := func(id, user, kind, status string, t time.Time, amount int64) error {
		var dest, corridor any
		if kind == "cross_border" {
			dest = "TJS"
			corridor = "RU-TJ"
		}
		_, e := tx.ExecContext(ctx, `INSERT INTO transfers(id,user_id,kind,status,occurred_at,recorded_at,amount_minor,source_currency,destination_currency,corridor)
		VALUES($1,$2,$3,$4,$5,$5,$6,'RUB',$7,$8) ON CONFLICT(id) DO NOTHING`, id, user, kind, status, t, amount, dest, corridor)
		return e
	}
	for month := 3; month <= 8; month++ {
		for _, day := range []int{5, 20} {
			income := time.Date(2026, time.Month(month), day, 9, 0, 0, 0, loc)
			prefix := fmt.Sprintf("demo-regular-%02d-%02d", month, day)
			if err = insert(prefix+"-income", "demo-regular", "income", "completed", income, 8_000_000); err != nil {
				return err
			}
			lag := month % 2
			hour, minute, amount := 18, 15+(month%3)*15, int64(1_500_000+(month-3)*20_000)
			if day == 20 {
				hour = 12
				minute = 10 + (month%3)*15
				amount = 2_000_000 + int64(month-3)*20_000
			}
			payment := time.Date(2026, time.Month(month), day+lag, hour, minute, 0, 0, loc)
			if err = insert(prefix+"-transfer", "demo-regular", "cross_border", "completed", payment, amount); err != nil {
				return err
			}
			// Failed and domestic events must not change the learned behaviour.
			if err = insert(prefix+"-failed", "demo-regular", "cross_border", "failed", income.Add(time.Hour), 9_900_000); err != nil {
				return err
			}
			if err = insert(prefix+"-domestic", "demo-regular", "domestic", "completed", income.Add(2*time.Hour), 500_000); err != nil {
				return err
			}
		}
		income := time.Date(2026, time.Month(month), 5, 9, 0, 0, 0, loc)
		if err = insert(fmt.Sprintf("demo-irregular-%02d-income", month), "demo-irregular", "income", "completed", income, 7_000_000); err != nil {
			return err
		}
		// Transfers 5–10 days after income: no consistent post-income pattern.
		if err = insert(fmt.Sprintf("demo-irregular-%02d-transfer", month), "demo-irregular", "cross_border", "completed", income.AddDate(0, 0, month+2), 1_000_000); err != nil {
			return err
		}
	}
	for month := 7; month <= 8; month++ {
		income := time.Date(2026, time.Month(month), 5, 9, 0, 0, 0, loc)
		if err = insert(fmt.Sprintf("demo-sparse-%02d-income", month), "demo-sparse", "income", "completed", income, 5_000_000); err != nil {
			return err
		}
		if err = insert(fmt.Sprintf("demo-sparse-%02d-transfer", month), "demo-sparse", "cross_border", "completed", income.Add(10*time.Hour), 1_000_000); err != nil {
			return err
		}
	}
	return tx.Commit()
}
