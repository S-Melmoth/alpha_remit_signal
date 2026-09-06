package platform

import (
	"context"
	"fmt"
	"net/url"
	"os"
	"testing"
	"time"

	"ai-hack/behaviour/internal/model"
	"ai-hack/behaviour/internal/store"
)

func TestNextMonthlyClampsShortMonths(t *testing.T) {
	got := NextMonthly(time.Date(2027, 1, 31, 19, 0, 0, 0, time.UTC), 31, "18:30", "UTC")
	if want := "2027-02-28T18:30:00Z"; got.Format(time.RFC3339) != want {
		t.Fatalf("next monthly = %s, want %s", got.Format(time.RFC3339), want)
	}
}

func TestServicesRoundTrip(t *testing.T) {
	dsn := os.Getenv("TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("set TEST_DATABASE_URL to run PostgreSQL integration test")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	admin, err := store.Open(ctx, dsn)
	if err != nil {
		t.Fatal(err)
	}
	defer admin.Close()
	schema := fmt.Sprintf("services_test_%d", time.Now().UnixNano())
	if _, err = admin.ExecContext(ctx, "CREATE SCHEMA "+schema); err != nil {
		t.Fatal(err)
	}
	defer admin.ExecContext(context.Background(), "DROP SCHEMA "+schema+" CASCADE")
	u, err := url.Parse(dsn)
	if err != nil {
		t.Fatal(err)
	}
	q := u.Query()
	q.Set("search_path", schema)
	u.RawQuery = q.Encode()
	db, err := store.Open(ctx, u.String())
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if err = store.Migrate(ctx, db); err != nil {
		t.Fatal(err)
	}
	if err = store.Seed(ctx, db); err != nil {
		t.Fatal(err)
	}
	asOf := mustTime(t, "2026-09-01T00:00:00+03:00")
	if _, err = store.Analyze(ctx, db, model.Config{AsOf: asOf, HistoryMonths: 6, HorizonDays: 30}); err != nil {
		t.Fatal(err)
	}
	if err = store.SeedServices(ctx, db); err != nil {
		t.Fatal(err)
	}
	now := mustTime(t, "2026-09-05T18:00:00+03:00")

	history, err := HistoryFor(ctx, db, "demo-regular", now)
	if err != nil || history == nil || history.Recipient.Phone != "+79265775128" || !history.CanSchedule {
		t.Fatalf("history = %+v, err=%v", history, err)
	}
	delivered, err := Deliver(ctx, db, now)
	if err != nil || delivered != 6 {
		t.Fatalf("delivered=%d err=%v", delivered, err)
	}
	if delivered, err = Deliver(ctx, db, now); err != nil || delivered != 0 {
		t.Fatalf("repeat delivered=%d err=%v", delivered, err)
	}
	var first, second string
	if err = db.QueryRowContext(ctx, `SELECT min(type) FILTER (WHERE slot=1), min(type) FILTER (WHERE slot=2) FROM completed_push WHERE user_id='demo-regular'`).Scan(&first, &second); err != nil {
		t.Fatal(err)
	}
	if first != "behaviour" || second != "history" {
		t.Fatalf("priority order = %s, %s", first, second)
	}

	input := ScheduleInput{Payment: Payment{AmountMinor: 1_700_000, SourceCurrency: "RUB", DestinationCurrency: "TJS", Corridor: "RU-TJ", Recipient: Recipient{Name: "Фаррух Саидов", Phone: "+79265775128", Bank: "Душанбе Сити Банк", Country: "Таджикистан"}, Purpose: "Безвозмездный перевод на текущие расходы"}, StartDate: "2026-10-05", LocalTime: "18:30", Timezone: "Europe/Moscow", Confirmed: true}
	schedule, err := SaveSchedule(ctx, db, "demo-regular", "", "services-test-request", input, now)
	if err != nil {
		t.Fatal(err)
	}
	repeat, err := SaveSchedule(ctx, db, "demo-regular", "", "services-test-request", input, now)
	if err != nil || repeat.ID != schedule.ID {
		t.Fatalf("idempotent save = %+v err=%v", repeat, err)
	}
	due := mustTime(t, "2026-10-05T18:31:00+03:00")
	count, err := ExecuteDue(ctx, db, due)
	if err != nil || count != 2 {
		t.Fatalf("execute due=%d err=%v", count, err)
	}
	var runs int
	if err = db.QueryRowContext(ctx, `SELECT count(*) FROM scheduled_payment_runs`).Scan(&runs); err != nil || runs != 2 {
		t.Fatalf("runs=%d err=%v", runs, err)
	}
}

func mustTime(t *testing.T, value string) time.Time {
	t.Helper()
	parsed, err := time.Parse(time.RFC3339, value)
	if err != nil {
		t.Fatal(err)
	}
	return parsed
}
