package platform

import (
	"ai-hack/behaviour/internal/model"
	"context"
	"database/sql"
	"encoding/json"
	"time"
)

type Recipient struct {
	Name    string `json:"name"`
	Phone   string `json:"phone"`
	Bank    string `json:"bank"`
	Country string `json:"country"`
}
type Payment struct {
	AmountMinor         int64     `json:"amount_minor"`
	SourceCurrency      string    `json:"source_currency"`
	DestinationCurrency string    `json:"destination_currency"`
	Corridor            string    `json:"corridor"`
	Recipient           Recipient `json:"recipient"`
	Purpose             string    `json:"purpose"`
}
type History struct {
	Payment
	TransferID      string    `json:"transfer_id"`
	OccurredAt      time.Time `json:"occurred_at"`
	WeeksAgo        int       `json:"weeks_ago"`
	SimilarPayments int       `json:"similar_payments"`
	CanSchedule     bool      `json:"can_schedule"`
}

// HistoryFor selects an actual historical transfer from the active pattern's
// evidence, including all recipient details needed to populate the transfer form.
func HistoryFor(ctx context.Context, db *sql.DB, user string, now time.Time) (*History, error) {
	var h History
	var recipient []byte
	err := db.QueryRowContext(ctx, `
 WITH evidence AS (
 SELECT DISTINCT t.* FROM behaviour_push b JOIN transfers t
 ON b.metadata->'evidence_transfer_ids' ? t.id
 WHERE b.user_id=$1 AND t.user_id=$1 AND b.model_version=$3
 AND b.as_of=(SELECT max(as_of) FROM behaviour_push WHERE model_version=$3 AND as_of<=$2)
 AND b.window_start<=$2 AND $2<b.window_end
 AND t.kind='cross_border' AND t.status='completed' AND t.occurred_at<$2 AND t.recorded_at<=$2
 AND t.recipient IS NOT NULL AND t.purpose IS NOT NULL
 ), latest AS (SELECT * FROM evidence ORDER BY occurred_at DESC,id LIMIT 1)
 SELECT l.id,l.occurred_at,l.amount_minor,l.source_currency,l.destination_currency,l.corridor,l.recipient,l.purpose,
 (SELECT count(*) FROM evidence e WHERE e.recipient=l.recipient AND e.corridor=l.corridor
 AND e.amount_minor BETWEEN l.amount_minor*0.8 AND l.amount_minor*1.2)
 FROM latest l`, user, now, model.Version).Scan(&h.TransferID, &h.OccurredAt, &h.AmountMinor, &h.SourceCurrency, &h.DestinationCurrency, &h.Corridor, &recipient, &h.Purpose, &h.SimilarPayments)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	if err = json.Unmarshal(recipient, &h.Recipient); err != nil {
		return nil, err
	}
	h.WeeksAgo = int(now.Sub(h.OccurredAt).Hours() / 168)
	h.CanSchedule = h.SimilarPayments >= 2
	return &h, nil
}
