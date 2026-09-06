package store

import (
	"context"
	"fmt"
	"net/url"
	"os"
	"reflect"
	"testing"
	"time"

	"ai-hack/behaviour/internal/model"
)

func TestPostgresRoundTrip(t *testing.T) {
	dsn := os.Getenv("TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("set TEST_DATABASE_URL to run PostgreSQL integration test")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	admin, err := Open(ctx, dsn)
	if err != nil {
		t.Fatal(err)
	}
	defer admin.Close()
	schema := fmt.Sprintf("behaviour_test_%d", time.Now().UnixNano())
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
	db, err := Open(ctx, u.String())
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	for i := 0; i < 2; i++ {
		if err = Migrate(ctx, db); err != nil {
			t.Fatal(err)
		}
		if err = Seed(ctx, db); err != nil {
			t.Fatal(err)
		}
	}
	var count int
	if err = db.QueryRowContext(ctx, `SELECT count(*) FROM transfers`).Scan(&count); err != nil || count != 64 {
		t.Fatalf("seed count=%d err=%v", count, err)
	}
	asOf, _ := time.Parse(time.RFC3339, "2026-09-01T00:00:00+03:00")
	cfg := model.Config{AsOf: asOf, HistoryMonths: 6, HorizonDays: 30}
	w, err := Analyze(ctx, db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(w) != 4 {
		t.Fatalf("want 4 candidates, got %+v", w)
	}
	if w[0].TypicalAmountMinor != 1_550_000 || w[2].TypicalAmountMinor != 2_050_000 {
		t.Fatalf("wrong median amounts: %+v", w)
	}
	w2, err := Analyze(ctx, db, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(w, w2) {
		t.Fatal("rerun changed result")
	}
	if err = db.QueryRowContext(ctx, `SELECT count(*) FROM behaviour_push`).Scan(&count); err != nil || count != 4 {
		t.Fatalf("rerun count=%d err=%v", count, err)
	}
	var evidence int
	if err = db.QueryRowContext(ctx, `SELECT jsonb_array_length(metadata->'evidence_transfer_ids') FROM behaviour_push LIMIT 1`).Scan(&evidence); err != nil || evidence != 6 {
		t.Fatalf("metadata evidence=%d err=%v", evidence, err)
	}
	// Recomputing an empty snapshot must remove its stale candidates atomically.
	if _, err = db.ExecContext(ctx, `UPDATE transfers SET status='failed' WHERE kind='cross_border'`); err != nil {
		t.Fatal(err)
	}
	w, err = Analyze(ctx, db, cfg)
	if err != nil || len(w) != 0 {
		t.Fatalf("empty recompute: %v %v", w, err)
	}
	if err = db.QueryRowContext(ctx, `SELECT count(*) FROM behaviour_push`).Scan(&count); err != nil || count != 0 {
		t.Fatalf("stale results count=%d err=%v", count, err)
	}
}
