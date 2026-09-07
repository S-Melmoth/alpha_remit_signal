package platform

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"time"
)

var currencyCode = regexp.MustCompile(`^[A-Z]{3}$`)

type MLSignal struct {
	Date                 string  `json:"date"`
	Corridor             string  `json:"corridor"`
	Indicator            string  `json:"indicator"`
	Direction            string  `json:"direction"`
	Strength             float64 `json:"strength"`
	Speed                float64 `json:"speed"`
	RecommendedScenario  string  `json:"recommended_scenario"`
	SignalSource         string  `json:"signal_source"`
	ExplanationIndicator string  `json:"explanation_indicator"`
	TemplateID           string  `json:"template_id"`
	PushTitle            string  `json:"push_title"`
	PushText             string  `json:"push_text"`
	Currency             string  `json:"currency"`
	RUBPerUnit           float64 `json:"rub_per_unit"`
}

type MLSignalBatch struct {
	ModelVersion string     `json:"model_version"`
	Signals      []MLSignal `json:"signals"`
}

func (s MLSignal) validate() error {
	if _, err := time.Parse("2006-01-02", strings.TrimSuffix(s.Date, "T00:00:00.000")); err != nil {
		return fmt.Errorf("date must be YYYY-MM-DD")
	}
	if !currencyCode.MatchString(s.Currency) || s.Corridor != "RUB→"+s.Currency {
		return fmt.Errorf("currency and corridor do not match")
	}
	if s.SignalSource != "ml" || s.Indicator != "ml_good_push" {
		return fmt.Errorf("only thresholded ml_good_push signals are accepted")
	}
	if s.Strength < 0 || s.Strength > 1 || s.RUBPerUnit <= 0 {
		return fmt.Errorf("invalid strength or rub_per_unit")
	}
	if s.PushTitle == "" || s.PushText == "" || s.TemplateID == "" {
		return fmt.Errorf("push title, text and template_id are required")
	}
	return nil
}

// SaveMLSignals is an idempotent ingestion boundary between the Python model
// and the Go delivery service. The endpoint receives only rows that already
// passed the frozen model's percentile threshold.
func SaveMLSignals(ctx context.Context, db *sql.DB, batch MLSignalBatch, now time.Time) (int, error) {
	if batch.ModelVersion == "" || len(batch.ModelVersion) > 100 {
		return 0, fmt.Errorf("model_version is required and must not exceed 100 characters")
	}
	if len(batch.Signals) > 100 {
		return 0, fmt.Errorf("at most 100 signals per request")
	}
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return 0, err
	}
	defer tx.Rollback()

	saved := 0
	for _, signal := range batch.Signals {
		if err = signal.validate(); err != nil {
			return 0, err
		}
		date := strings.TrimSuffix(signal.Date, "T00:00:00.000")
		metadata, err := json.Marshal(map[string]any{
			"source":                "fx-ml",
			"model_version":         batch.ModelVersion,
			"signal_date":           date,
			"corridor":              signal.Corridor,
			"destination_currency":  signal.Currency,
			"indicator":             signal.Indicator,
			"direction":             signal.Direction,
			"strength":              signal.Strength,
			"speed":                 signal.Speed,
			"recommended_scenario":  signal.RecommendedScenario,
			"explanation_indicator": signal.ExplanationIndicator,
			"template_id":           signal.TemplateID,
			"rub_per_unit":          signal.RUBPerUnit,
		})
		if err != nil {
			return 0, err
		}
		id := fmt.Sprintf("fx-ml:%s:%s:%s", date, signal.Currency, signal.TemplateID)
		body := signal.PushTitle + "\n" + signal.PushText
		result, err := tx.ExecContext(ctx, `
INSERT INTO global_notifications(id,type,body,created_at,expires_at,metadata)
VALUES($1,'history',$2,$3::timestamptz,$3::timestamptz + interval '24 hours',$4::jsonb)
ON CONFLICT (id) DO NOTHING`, id, body, now, string(metadata))
		if err != nil {
			return 0, err
		}
		rows, err := result.RowsAffected()
		if err != nil {
			return 0, err
		}
		saved += int(rows)
	}
	if err = tx.Commit(); err != nil {
		return 0, err
	}
	return saved, nil
}
