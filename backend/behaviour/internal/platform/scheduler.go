package platform

import (
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"time"
)

type ScheduleInput struct {
	Payment
	StartDate string `json:"start_date"`
	LocalTime string `json:"local_time"`
	Timezone  string `json:"timezone"`
	Confirmed bool   `json:"confirmed"`
}
type Schedule struct {
	ScheduleInput
	ID         string    `json:"id"`
	UserID     string    `json:"user_id"`
	DayOfMonth int       `json:"day_of_month"`
	NextRunAt  time.Time `json:"next_run_at"`
	Status     string    `json:"status"`
}

var phoneRE = regexp.MustCompile(`^\+[1-9][0-9]{9,14}$`)
var ErrConflict = errors.New("idempotency key already used with different data")

func ValidatePayment(p Payment) error {
	routes := map[string]string{"TJS": "RU-TJ", "UZS": "RU-UZ", "KGS": "RU-KG", "BYN": "RU-BY"}
	if p.AmountMinor < 10000 || p.AmountMinor > 10000000 || p.SourceCurrency != "RUB" || routes[p.DestinationCurrency] == "" || routes[p.DestinationCurrency] != p.Corridor {
		return fmt.Errorf("amount must be 100..100000 RUB and corridor must match currency")
	}
	if !phoneRE.MatchString(p.Recipient.Phone) || len(p.Recipient.Name) < 2 || len(p.Recipient.Name) > 150 || len(p.Recipient.Bank) < 2 || len(p.Recipient.Bank) > 150 || len(p.Recipient.Country) < 2 || len(p.Recipient.Country) > 100 || len(p.Purpose) < 2 || len(p.Purpose) > 300 {
		return fmt.Errorf("valid recipient name, international phone, bank, country and purpose required")
	}
	return nil
}
func (s ScheduleInput) Validate(now time.Time) (time.Time, error) {
	if err := ValidatePayment(s.Payment); err != nil {
		return time.Time{}, err
	}
	loc, err := time.LoadLocation(s.Timezone)
	if err != nil || s.Timezone == "" {
		return time.Time{}, fmt.Errorf("valid IANA timezone required")
	}
	first, err := time.ParseInLocation("2006-01-02 15:04", s.StartDate+" "+s.LocalTime, loc)
	if err != nil || first.Format("2006-01-02 15:04") != s.StartDate+" "+s.LocalTime {
		return time.Time{}, fmt.Errorf("invalid or nonexistent local date/time")
	}
	if !first.After(now) {
		return time.Time{}, fmt.Errorf("first payment must be in the future")
	}
	return first, nil
}

// NextMonthly keeps the original calendar day, clamping 29..31 in short months.
// Missing DST times move forward to the first valid minute; ambiguous times use
// Go's IANA timezone resolution. The timezone is persisted, never the UTC offset.
func NextMonthly(after time.Time, day int, clock, zone string) time.Time {
	loc, _ := time.LoadLocation(zone)
	local := after.In(loc)
	var hour, minute int
	fmt.Sscanf(clock, "%d:%d", &hour, &minute)
	for offset := 0; ; offset++ {
		month := time.Date(local.Year(), local.Month()+time.Month(offset), 1, 0, 0, 0, 0, loc)
		last := time.Date(month.Year(), month.Month()+1, 0, 12, 0, 0, 0, loc).Day()
		d := day
		if d > last {
			d = last
		}
		candidate := time.Date(month.Year(), month.Month(), d, hour, minute, 0, 0, loc)
		// Advance normalized nonexistent clock times that Go placed before the gap.
		for candidate.Day() == d && candidate.Hour()*60+candidate.Minute() < hour*60+minute {
			candidate = candidate.Add(time.Minute)
		}
		if candidate.After(after) {
			return candidate
		}
	}
}
func NewID(prefix string) string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		panic(err)
	}
	return prefix + hex.EncodeToString(b[:])
}

const scheduleColumns = `id,user_id,amount_minor,source_currency,destination_currency,corridor,recipient,purpose,start_date::text,day_of_month,local_time,timezone,next_run_at,status`

type scanner interface{ Scan(...any) error }

func scanSchedule(row scanner) (Schedule, error) {
	var s Schedule
	var recipient []byte
	err := row.Scan(&s.ID, &s.UserID, &s.AmountMinor, &s.SourceCurrency, &s.DestinationCurrency, &s.Corridor, &recipient, &s.Purpose, &s.StartDate, &s.DayOfMonth, &s.LocalTime, &s.Timezone, &s.NextRunAt, &s.Status)
	if err != nil {
		return s, err
	}
	err = json.Unmarshal(recipient, &s.Recipient)
	s.Confirmed = true
	return s, err
}
func SaveSchedule(ctx context.Context, db *sql.DB, user, id, key string, input ScheduleInput, now time.Time) (Schedule, error) {
	first, err := input.Validate(now)
	if err != nil {
		return Schedule{}, err
	}
	if !input.Confirmed {
		return Schedule{}, fmt.Errorf("explicit confirmation required")
	}
	recipient, _ := json.Marshal(input.Recipient)
	if id != "" {
		return scanSchedule(db.QueryRowContext(ctx, `UPDATE scheduled_cis_payments SET amount_minor=$3,source_currency=$4,destination_currency=$5,corridor=$6,recipient=$7,purpose=$8,start_date=$9,day_of_month=$10,local_time=$11,timezone=$12,next_run_at=$13,updated_at=$14 WHERE user_id=$1 AND id=$2 AND status='active' RETURNING `+scheduleColumns, user, id, input.AmountMinor, input.SourceCurrency, input.DestinationCurrency, input.Corridor, string(recipient), input.Purpose, input.StartDate, first.Day(), input.LocalTime, input.Timezone, first, now))
	}
	if len(key) < 8 || len(key) > 128 {
		return Schedule{}, fmt.Errorf("Idempotency-Key of 8..128 characters required")
	}
	s, err := scanSchedule(db.QueryRowContext(ctx, `INSERT INTO scheduled_cis_payments(id,user_id,amount_minor,source_currency,destination_currency,corridor,recipient,purpose,start_date,day_of_month,local_time,timezone,next_run_at,created_at,updated_at,confirmed_at,request_key)
 VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$14,$14,$15)
 ON CONFLICT(user_id,request_key) DO UPDATE SET request_key=EXCLUDED.request_key RETURNING `+scheduleColumns, NewID("scp-"), user, input.AmountMinor, input.SourceCurrency, input.DestinationCurrency, input.Corridor, string(recipient), input.Purpose, input.StartDate, first.Day(), input.LocalTime, input.Timezone, first, now, key))
	if err == nil && s.ScheduleInput != input {
		return Schedule{}, ErrConflict
	}
	return s, err
}

// ExecuteDue is a demo ledger adapter. Each occurrence, ledger entry and next
// date commit together. Missed months are skipped to avoid a burst of debits.
func ExecuteDue(ctx context.Context, db *sql.DB, now time.Time) (int, error) {
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return 0, err
	}
	defer tx.Rollback()
	rows, err := tx.QueryContext(ctx, `SELECT `+scheduleColumns+` FROM scheduled_cis_payments WHERE status='active' AND next_run_at<=$1 ORDER BY next_run_at LIMIT 100 FOR UPDATE SKIP LOCKED`, now)
	if err != nil {
		return 0, err
	}
	var due []Schedule
	for rows.Next() {
		s, e := scanSchedule(rows)
		if e != nil {
			rows.Close()
			return 0, e
		}
		due = append(due, s)
	}
	err = rows.Err()
	rows.Close()
	if err != nil {
		return 0, err
	}
	for _, s := range due {
		recipient, _ := json.Marshal(s.Recipient)
		id := NewID("scheduled-demo-")
		_, err = tx.ExecContext(ctx, `INSERT INTO transfers(id,user_id,kind,status,occurred_at,recorded_at,amount_minor,source_currency,destination_currency,corridor,recipient,purpose) VALUES($1,$2,'cross_border','completed',$3,$3,$4,$5,$6,$7,$8,$9)`, id, s.UserID, now, s.AmountMinor, s.SourceCurrency, s.DestinationCurrency, s.Corridor, string(recipient), s.Purpose)
		if err != nil {
			return 0, err
		}
		_, err = tx.ExecContext(ctx, `INSERT INTO scheduled_payment_runs(schedule_id,due_at,transfer_id,executed_at) VALUES($1,$2,$3,$4)`, s.ID, s.NextRunAt, id, now)
		if err != nil {
			return 0, err
		}
		_, err = tx.ExecContext(ctx, `UPDATE scheduled_cis_payments SET next_run_at=$2,updated_at=$3 WHERE id=$1`, s.ID, NextMonthly(now, s.DayOfMonth, s.LocalTime, s.Timezone), now)
		if err != nil {
			return 0, err
		}
	}
	return len(due), tx.Commit()
}
