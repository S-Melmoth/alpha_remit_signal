// Package model learns an explainable income-relative timing baseline.
package model

import (
	"fmt"
	"math"
	"sort"
	"time"
	_ "time/tzdata"
)

const Version = "income-window-v1"

type Transfer struct {
	ID, Kind, Status, SourceCurrency, DestinationCurrency, Corridor string
	OccurredAt, RecordedAt                                          time.Time
	AmountMinor                                                     int64
}

type Client struct {
	UserID, Timezone string
	Transfers        []Transfer
}

type Config struct {
	AsOf                       time.Time
	HistoryMonths, HorizonDays int
}

type Metadata struct {
	Algorithm           string    `json:"algorithm"`
	HistoryStart        time.Time `json:"history_start"`
	HistoryMonths       int       `json:"history_months"`
	HorizonDays         int       `json:"horizon_days"`
	IncomeOpportunities int       `json:"income_opportunities"`
	MatchedMonths       int       `json:"matched_months"`
	LagDaysMin          int       `json:"lag_days_min"`
	LagDaysMax          int       `json:"lag_days_max"`
	WindowStartLagMin   int       `json:"window_start_lag_days_min"`
	WindowStartLagMax   int       `json:"window_start_lag_days_max"`
	LocalTimeStart      string    `json:"local_time_start"`
	LocalTimeEnd        string    `json:"local_time_end"`
	TimeEndDayOffset    int       `json:"time_end_day_offset"`
	TimeCoverage        float64   `json:"time_coverage"`
	EvidenceTransferIDs []string  `json:"evidence_transfer_ids"`
	Explanation         string    `json:"explanation"`
}

type Window struct {
	UserID              string    `json:"user_id"`
	AsOf                time.Time `json:"as_of"`
	ModelVersion        string    `json:"model_version"`
	SourceCurrency      string    `json:"source_currency"`
	DestinationCurrency string    `json:"destination_currency"`
	Corridor            string    `json:"corridor"`
	IncomeDay           int       `json:"income_day"`
	CycleDate           string    `json:"cycle_date"`
	WindowStart         time.Time `json:"window_start"`
	WindowEnd           time.Time `json:"window_end"`
	Timezone            string    `json:"timezone"`
	TypicalAmountMinor  int64     `json:"typical_amount_minor"`
	AmountP25Minor      int64     `json:"amount_p25_minor"`
	AmountP75Minor      int64     `json:"amount_p75_minor"`
	SampleCount         int       `json:"sample_count"`
	SupportRatio        float64   `json:"support_ratio"`
	Status              string    `json:"status"`
	Metadata            Metadata  `json:"metadata"`
}

type anchor struct {
	currency string
	day      int
}
type route struct{ source, destination, corridor string }
type sample struct {
	transfer    Transfer
	lag, minute int
	month       string
}

func Analyze(client Client, cfg Config) ([]Window, error) {
	if cfg.AsOf.IsZero() || cfg.HistoryMonths < 1 || cfg.HistoryMonths > 120 || cfg.HorizonDays < 1 || cfg.HorizonDays > 366 {
		return nil, fmt.Errorf("as-of is required; history-months must be 1..120 and horizon-days 1..366")
	}
	loc, err := time.LoadLocation(client.Timezone)
	if err != nil {
		return nil, fmt.Errorf("client %s timezone: %w", client.UserID, err)
	}
	asOf := cfg.AsOf.In(loc)
	// Whole calendar months plus the current partial month; no AddDate month overflow.
	monthStart := time.Date(asOf.Year(), asOf.Month(), 1, 0, 0, 0, 0, loc)
	historyStart := monthStart.AddDate(0, -cfg.HistoryMonths, 0)
	horizon := asOf.AddDate(0, 0, cfg.HorizonDays)
	var events []Transfer
	for _, t := range client.Transfers {
		if t.Status == "completed" && !t.OccurredAt.Before(historyStart) && t.OccurredAt.Before(asOf) && t.RecordedAt.Before(asOf) {
			events = append(events, t)
		}
	}
	sort.Slice(events, func(i, j int) bool {
		if events[i].OccurredAt.Equal(events[j].OccurredAt) {
			return events[i].ID < events[j].ID
		}
		return events[i].OccurredAt.Before(events[j].OccurredAt)
	})
	// Multiple credits on the same local date are one income opportunity.
	incomes := map[anchor][]Transfer{}
	seenIncome := map[string]bool{}
	var allIncomes []Transfer
	for _, t := range events {
		if t.Kind != "income" {
			continue
		}
		local := t.OccurredAt.In(loc)
		key := t.SourceCurrency + local.Format("2006-01-02")
		if seenIncome[key] {
			continue
		}
		seenIncome[key] = true
		allIncomes = append(allIncomes, t)
		a := anchor{t.SourceCurrency, local.Day()}
		// Observe all of days 0..3 before counting a missed transfer.
		if !dayStart(local).AddDate(0, 0, 4).After(asOf) {
			incomes[a] = append(incomes[a], t)
		}
	}
	var windows []Window
	for a, credits := range incomes {
		if len(credits) < 3 {
			continue
		}
		// Require recurring credits in >=60% of fully observable calendar opportunities.
		exposures := 0
		for m := historyStart; !m.After(asOf); m = m.AddDate(0, 1, 0) {
			d := time.Date(m.Year(), m.Month(), a.day, 0, 0, 0, 0, loc)
			if d.Month() == m.Month() && !d.AddDate(0, 0, 4).After(asOf) {
				exposures++
			}
		}
		if exposures == 0 || float64(len(credits))/float64(exposures) < .6 {
			continue
		}
		samples := map[route][]sample{}
		for _, credit := range credits {
			matched := map[route]bool{}
			for _, t := range events {
				if t.Kind != "cross_border" || t.SourceCurrency != a.currency || t.OccurredAt.Before(credit.OccurredAt) {
					continue
				}
				lag := calendarDays(credit.OccurredAt.In(loc), t.OccurredAt.In(loc))
				if lag > 3 {
					break
				}
				// A transfer belongs to the latest credit date, never two income cycles.
				latest := true
				for _, other := range allIncomes {
					if other.SourceCurrency == a.currency && other.OccurredAt.After(credit.OccurredAt) && !other.OccurredAt.After(t.OccurredAt) {
						latest = false
						break
					}
				}
				if !latest {
					continue
				}
				r := route{t.SourceCurrency, t.DestinationCurrency, t.Corridor}
				if matched[r] {
					continue
				}
				matched[r] = true
				local := t.OccurredAt.In(loc)
				samples[r] = append(samples[r], sample{t, lag, local.Hour()*60 + local.Minute(), credit.OccurredAt.In(loc).Format("2006-01")})
			}
		}
		for r, observations := range samples {
			support := float64(len(observations)) / float64(len(credits))
			months := map[string]bool{}
			var amounts []int64
			var minutes []int
			var ids []string
			lagMin, lagMax := 3, 0
			for _, s := range observations {
				months[s.month] = true
				amounts = append(amounts, s.transfer.AmountMinor)
				minutes = append(minutes, s.minute)
				ids = append(ids, s.transfer.ID)
				lagMin = min(lagMin, s.lag)
				lagMax = max(lagMax, s.lag)
			}
			if len(months) < 3 || support < .6 {
				continue
			}
			startMinute, width, coverage := clockWindow(minutes)
			if width > 6*60 {
				continue
			} // No narrow, useful daily window.
			// The date of a midnight-spanning window is the date it starts.
			// A 00:10 transfer on day +1 belongs to the 23:xx window on day 0.
			windowLagMin, windowLagMax := 3, -1
			for _, s := range observations {
				if (s.minute-startMinute+1440)%1440 >= width {
					continue
				}
				lag := s.lag
				if s.minute < startMinute {
					lag--
				}
				windowLagMin = min(windowLagMin, lag)
				windowLagMax = max(windowLagMax, lag)
			}
			sort.Slice(amounts, func(i, j int) bool { return amounts[i] < amounts[j] })
			meta := Metadata{
				Algorithm: Version, HistoryStart: historyStart, HistoryMonths: cfg.HistoryMonths, HorizonDays: cfg.HorizonDays,
				IncomeOpportunities: len(credits), MatchedMonths: len(months), LagDaysMin: lagMin, LagDaysMax: lagMax,
				WindowStartLagMin: windowLagMin, WindowStartLagMax: windowLagMax,
				LocalTimeStart: clockText(startMinute), LocalTimeEnd: clockText(startMinute + width), TimeEndDayOffset: (startMinute + width) / 1440,
				TimeCoverage: coverage, EvidenceTransferIDs: ids,
				Explanation: fmt.Sprintf("Поступление %d-го числа; первый перевод по коридору через %d–%d календарных дней в %d из %d циклов. Сумма — медиана списания в %s.", a.day, lagMin, lagMax, len(observations), len(credits), r.source),
			}
			// Include the prior month for cycles crossing a month boundary.
			for m := monthStart.AddDate(0, -1, 0); m.Before(horizon); m = m.AddDate(0, 1, 0) {
				cycle := time.Date(m.Year(), m.Month(), a.day, 0, 0, 0, 0, loc)
				if cycle.Month() != m.Month() {
					continue
				} // No 31st in a 30-day month.
				// Suppress a cycle already fulfilled, including before the learned clock window.
				fulfilled := false
				cycleCredit := cycle
				for _, credit := range allIncomes {
					if credit.SourceCurrency == r.source && dayStart(credit.OccurredAt.In(loc)).Equal(cycle) {
						cycleCredit = credit.OccurredAt
						break
					}
				}
				for _, t := range events {
					if t.Kind == "cross_border" && t.SourceCurrency == r.source && t.DestinationCurrency == r.destination && t.Corridor == r.corridor && !t.OccurredAt.Before(cycleCredit) && t.OccurredAt.Before(cycle.AddDate(0, 0, 4)) {
						fulfilled = true
						break
					}
				}
				if fulfilled {
					continue
				}
				for lag := windowLagMin; lag <= windowLagMax; lag++ {
					date := cycle.AddDate(0, 0, lag)
					start := time.Date(date.Year(), date.Month(), date.Day(), startMinute/60, startMinute%60, 0, 0, loc)
					end := time.Date(date.Year(), date.Month(), date.Day(), (startMinute+width)/60, (startMinute+width)%60, 0, 0, loc)
					if !end.After(asOf) || !start.Before(horizon) || !end.After(start) {
						continue
					}
					windows = append(windows, Window{
						UserID: client.UserID, AsOf: cfg.AsOf, ModelVersion: Version,
						SourceCurrency: r.source, DestinationCurrency: r.destination, Corridor: r.corridor,
						IncomeDay: a.day, CycleDate: cycle.Format("2006-01-02"), WindowStart: start, WindowEnd: end, Timezone: client.Timezone,
						TypicalAmountMinor: median(amounts), AmountP25Minor: quantile(amounts, .25), AmountP75Minor: quantile(amounts, .75),
						SampleCount: len(observations), SupportRatio: support, Status: "candidate", Metadata: meta,
					})
				}
			}
		}
	}
	sort.Slice(windows, func(i, j int) bool {
		a, b := windows[i], windows[j]
		if !a.WindowStart.Equal(b.WindowStart) {
			return a.WindowStart.Before(b.WindowStart)
		}
		if a.SourceCurrency != b.SourceCurrency {
			return a.SourceCurrency < b.SourceCurrency
		}
		if a.DestinationCurrency != b.DestinationCurrency {
			return a.DestinationCurrency < b.DestinationCurrency
		}
		if a.Corridor != b.Corridor {
			return a.Corridor < b.Corridor
		}
		return a.IncomeDay < b.IncomeDay
	})
	return windows, nil
}

func dayStart(t time.Time) time.Time {
	return time.Date(t.Year(), t.Month(), t.Day(), 0, 0, 0, 0, t.Location())
}
func calendarDays(a, b time.Time) int {
	x := time.Date(a.Year(), a.Month(), a.Day(), 0, 0, 0, 0, time.UTC)
	y := time.Date(b.Year(), b.Month(), b.Day(), 0, 0, 0, 0, time.UTC)
	return int(y.Sub(x) / (24 * time.Hour))
}
func clockText(m int) string { m %= 1440; return fmt.Sprintf("%02d:%02d", m/60, m%60) }
func median(v []int64) int64 {
	n := len(v)
	if n%2 == 1 {
		return v[n/2]
	}
	a, b := v[n/2-1], v[n/2]
	return a + (b-a)/2
}
func quantile(v []int64, q float64) int64 { return v[max(0, int(math.Ceil(q*float64(len(v))))-1)] }

// Shortest circular interval containing >=80% of observed minutes, padded by
// 30 minutes on either side. Handles 23:50 and 00:10 without a 24-hour window.
func clockWindow(minutes []int) (start, width int, coverage float64) {
	sort.Ints(minutes)
	n := len(minutes)
	count := int(math.Ceil(.8 * float64(n)))
	extended := append([]int(nil), minutes...)
	for _, m := range minutes {
		extended = append(extended, m+1440)
	}
	best, span := 0, 1441
	for i := 0; i < n; i++ {
		if s := extended[i+count-1] - extended[i]; s < span {
			best, span = i, s
		}
	}
	start = (extended[best] - 30 + 1440) % 1440
	width = span + 60
	inside := 0
	for _, m := range minutes {
		if (m-start+1440)%1440 < width {
			inside++
		}
	}
	return start, width, float64(inside) / float64(n)
}
