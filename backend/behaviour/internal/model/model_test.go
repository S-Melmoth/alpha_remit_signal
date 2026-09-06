package model

import (
	"fmt"
	"reflect"
	"testing"
	"time"
)

func timestamp(s string) time.Time {
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		panic(err)
	}
	return t
}
func fixture() (Client, Config) {
	c := Client{UserID: "regular", Timezone: "Europe/Moscow"}
	loc, _ := time.LoadLocation(c.Timezone)
	for month := 3; month <= 8; month++ {
		for _, day := range []int{5, 20} {
			income := time.Date(2026, time.Month(month), day, 9, 0, 0, 0, loc)
			hour := 18
			if day == 20 {
				hour = 12
			}
			payment := time.Date(2026, time.Month(month), day+month%2, hour, 30, 0, 0, loc)
			c.Transfers = append(c.Transfers,
				Transfer{ID: fmt.Sprintf("income-%d-%d", month, day), Kind: "income", Status: "completed", OccurredAt: income, RecordedAt: income, AmountMinor: 8_000_000, SourceCurrency: "RUB"},
				Transfer{ID: fmt.Sprintf("transfer-%d-%d", month, day), Kind: "cross_border", Status: "completed", OccurredAt: payment, RecordedAt: payment, AmountMinor: int64(1_000_000 + month*10_000), SourceCurrency: "RUB", DestinationCurrency: "TJS", Corridor: "RU-TJ"})
		}
	}
	return c, Config{AsOf: timestamp("2026-09-01T00:00:00+03:00"), HistoryMonths: 6, HorizonDays: 30}
}
func analyze(t *testing.T, c Client, cfg Config) []Window {
	t.Helper()
	w, err := Analyze(c, cfg)
	if err != nil {
		t.Fatal(err)
	}
	return w
}

func TestTwoIncomeCycles(t *testing.T) {
	c, cfg := fixture()
	windows := analyze(t, c, cfg)
	if len(windows) != 4 {
		t.Fatalf("want 4 windows, got %d: %+v", len(windows), windows)
	}
	for i, day := range []int{5, 6, 20, 21} {
		w := windows[i]
		hour := 18
		if day >= 20 {
			hour = 12
		}
		if w.WindowStart.Day() != day || w.WindowStart.Hour() != hour || w.WindowStart.Minute() != 0 || w.WindowEnd.Sub(w.WindowStart) != time.Hour {
			t.Errorf("wrong window: %+v", w)
		}
		if w.SampleCount != 6 || w.SupportRatio != 1 || w.TypicalAmountMinor != 1_055_000 || w.Metadata.MatchedMonths != 6 {
			t.Errorf("wrong evidence: %+v", w)
		}
	}
	// Input order must not change ties, evidence ordering, or output ordering.
	for i, j := 0, len(c.Transfers)-1; i < j; i, j = i+1, j-1 {
		c.Transfers[i], c.Transfers[j] = c.Transfers[j], c.Transfers[i]
	}
	if got := analyze(t, c, cfg); !reflect.DeepEqual(windows, got) {
		t.Fatal("input order changed result")
	}
}

func TestIgnoresFailedDomesticFutureAndLateRecorded(t *testing.T) {
	c, cfg := fixture()
	want := analyze(t, c, cfg)
	base := c.Transfers[1]
	for _, kind := range []string{"failed", "pending", "domestic", "future", "late"} {
		extra := base
		extra.ID = kind
		extra.AmountMinor = 99_000_000
		switch kind {
		case "failed", "pending":
			extra.Status = kind
		case "domestic":
			extra.Kind = "domestic"
		case "future":
			extra.OccurredAt = cfg.AsOf
		case "late":
			extra.RecordedAt = cfg.AsOf
		}
		c.Transfers = append(c.Transfers, extra)
	}
	if got := analyze(t, c, cfg); !reflect.DeepEqual(want, got) {
		t.Fatal("ineligible events changed result")
	}
}

func TestInsufficientOrInconsistentHistory(t *testing.T) {
	for _, scenario := range []string{"two-months", "rare-transfers", "no-income", "scattered-hours"} {
		t.Run(scenario, func(t *testing.T) {
			c, cfg := fixture()
			var filtered []Transfer
			for _, e := range c.Transfers {
				switch scenario {
				case "two-months":
					if e.OccurredAt.Month() < 7 {
						continue
					}
				case "rare-transfers":
					if e.Kind == "cross_border" && e.OccurredAt.Month() < 7 {
						continue
					}
				case "no-income":
					if e.Kind == "income" {
						continue
					}
				case "scattered-hours":
					if e.Kind == "cross_border" {
						e.OccurredAt = dayStart(e.OccurredAt).AddDate(0, 0, 1).Add(time.Duration(int(e.OccurredAt.Month())%6*4) * time.Hour)
						e.RecordedAt = e.OccurredAt
					}
				}
				filtered = append(filtered, e)
			}
			c.Transfers = filtered
			if got := analyze(t, c, cfg); len(got) != 0 {
				t.Fatalf("want no candidates, got %d", len(got))
			}
		})
	}
}

func TestAlreadyTransferredSuppressesWholeCycle(t *testing.T) {
	c, cfg := fixture()
	cfg.AsOf = timestamp("2026-09-05T11:00:00+03:00")
	cfg.HorizonDays = 26
	for _, kind := range []string{"income", "cross_border"} {
		at := timestamp("2026-09-05T09:00:00+03:00")
		if kind == "cross_border" {
			at = at.Add(time.Hour)
		}
		c.Transfers = append(c.Transfers, Transfer{ID: kind, Kind: kind, Status: "completed", OccurredAt: at, RecordedAt: at, AmountMinor: 1_000_000, SourceCurrency: "RUB", DestinationCurrency: "TJS", Corridor: "RU-TJ"})
	}
	w := analyze(t, c, cfg)
	if len(w) != 2 || w[0].WindowStart.Day() != 20 || w[1].WindowStart.Day() != 21 {
		t.Fatalf("fulfilled cycle still present: %+v", w)
	}
}

func TestCurrenciesAndCorridorsStaySeparate(t *testing.T) {
	c, cfg := fixture()
	for _, e := range append([]Transfer(nil), c.Transfers...) {
		if e.Kind != "cross_border" {
			continue
		}
		e.ID += "-uz"
		e.Corridor = "RU-UZ"
		e.DestinationCurrency = "UZS"
		e.AmountMinor = 9_000_000
		c.Transfers = append(c.Transfers, e)
		e.ID += "-usd"
		e.SourceCurrency = "USD"
		c.Transfers = append(c.Transfers, e) // no matching USD credits
	}
	w := analyze(t, c, cfg)
	if len(w) != 8 {
		t.Fatalf("want 8, got %d", len(w))
	}
	for _, e := range w {
		if e.SourceCurrency != "RUB" {
			t.Fatal("mixed income currencies")
		}
		if e.Corridor == "RU-UZ" && e.TypicalAmountMinor != 9_000_000 {
			t.Fatal("mixed route amounts")
		}
		if e.Corridor == "RU-TJ" && e.TypicalAmountMinor != 1_055_000 {
			t.Fatal("mixed route amounts")
		}
	}
}

func TestMidnightWindowUsesStartDate(t *testing.T) {
	c, cfg := fixture()
	for i := range c.Transfers {
		e := &c.Transfers[i]
		if e.Kind != "cross_border" {
			continue
		}
		day := 5
		if e.OccurredAt.Day() > 15 {
			day = 20
		}
		if e.OccurredAt.Month()%2 == 0 {
			e.OccurredAt = time.Date(2026, e.OccurredAt.Month(), day, 23, 50, 0, 0, e.OccurredAt.Location())
		} else {
			e.OccurredAt = time.Date(2026, e.OccurredAt.Month(), day+1, 0, 10, 0, 0, e.OccurredAt.Location())
		}
		e.RecordedAt = e.OccurredAt
	}
	w := analyze(t, c, cfg)
	if len(w) != 2 {
		t.Fatalf("want 2 midnight windows, got %+v", w)
	}
	for i, day := range []int{5, 20} {
		if w[i].WindowStart.Day() != day || w[i].WindowStart.Hour() != 23 || w[i].WindowStart.Minute() != 20 || w[i].WindowEnd.Day() != day+1 || w[i].WindowEnd.Minute() != 40 {
			t.Fatalf("incorrect midnight window: %+v", w[i])
		}
	}
}

func TestCalendarLagAcrossDST(t *testing.T) {
	loc, _ := time.LoadLocation("America/New_York")
	a := time.Date(2026, 3, 7, 23, 0, 0, 0, loc)
	b := time.Date(2026, 3, 8, 3, 0, 0, 0, loc)
	if calendarDays(a, b) != 1 {
		t.Fatal("lag must count local dates, not elapsed 24h")
	}
}

func TestMonthEndCycleContinuesIntoNextMonth(t *testing.T) {
	loc, _ := time.LoadLocation("Europe/Moscow")
	c := Client{UserID: "month-end", Timezone: "Europe/Moscow"}
	// Day 30 exists in each training month. Only next-day transfers are observed.
	for month := 3; month <= 8; month++ {
		income := time.Date(2026, time.Month(month), 30, 9, 0, 0, 0, loc)
		payment := income.AddDate(0, 0, 1).Add(9 * time.Hour)
		c.Transfers = append(c.Transfers,
			Transfer{ID: fmt.Sprint("income", month), Kind: "income", Status: "completed", OccurredAt: income, RecordedAt: income, AmountMinor: 4_000_000, SourceCurrency: "RUB"},
			Transfer{ID: fmt.Sprint("transfer", month), Kind: "cross_border", Status: "completed", OccurredAt: payment, RecordedAt: payment, AmountMinor: 1_000_000, SourceCurrency: "RUB", DestinationCurrency: "TJS", Corridor: "RU-TJ"})
	}
	cfg := Config{AsOf: timestamp("2026-10-01T00:00:00+03:00"), HistoryMonths: 7, HorizonDays: 2}
	w := analyze(t, c, cfg)
	if len(w) != 1 || w[0].CycleDate != "2026-09-30" || w[0].WindowStart.Format("2006-01-02") != "2026-10-01" {
		t.Fatalf("lost prior-month cycle: %+v", w)
	}
}

func TestUnfinishedCycleDoesNotCountAsMiss(t *testing.T) {
	c, cfg := fixture()
	cfg.AsOf = timestamp("2026-09-06T10:00:00+03:00")
	cfg.HorizonDays = 24
	credit := timestamp("2026-09-05T09:00:00+03:00")
	c.Transfers = append(c.Transfers, Transfer{ID: "new-credit", Kind: "income", Status: "completed", OccurredAt: credit, RecordedAt: credit, AmountMinor: 8_000_000, SourceCurrency: "RUB"})
	w := analyze(t, c, cfg)
	if len(w) != 3 {
		t.Fatalf("want active/upcoming windows, got %d", len(w))
	}
	for _, e := range w {
		if e.SupportRatio != 1 || e.Metadata.IncomeOpportunities != 6 {
			t.Fatalf("unfinished cycle counted: %+v", e)
		}
	}
}

func TestInvalidConfigAndTimezone(t *testing.T) {
	c, cfg := fixture()
	c.Timezone = "bad/timezone"
	if _, err := Analyze(c, cfg); err == nil {
		t.Fatal("expected timezone error")
	}
	c, cfg = fixture()
	cfg.HorizonDays = 0
	if _, err := Analyze(c, cfg); err == nil {
		t.Fatal("expected config error")
	}
}
