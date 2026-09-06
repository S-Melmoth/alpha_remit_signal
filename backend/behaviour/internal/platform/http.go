package platform

import (
	"ai-hack/behaviour/internal/store"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"
)

type API struct {
	DB   *sql.DB
	Now  func() time.Time
	Demo bool
}

func reply(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	if v != nil {
		json.NewEncoder(w).Encode(v)
	}
}
func failure(w http.ResponseWriter, status int, msg string) {
	reply(w, status, map[string]string{"error": msg})
}
func decode(w http.ResponseWriter, r *http.Request, v any) bool {
	r.Body = http.MaxBytesReader(w, r.Body, 16*1024)
	d := json.NewDecoder(r.Body)
	d.DisallowUnknownFields()
	if err := d.Decode(v); err != nil {
		failure(w, 400, "invalid JSON body")
		return false
	}
	if d.Decode(&struct{}{}) != io.EOF {
		failure(w, 400, "one JSON object required")
		return false
	}
	return true
}
func (a API) Handler(service string) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		if err := a.DB.PingContext(r.Context()); err != nil {
			failure(w, 503, "database unavailable")
			return
		}
		reply(w, 200, map[string]string{"service": service, "transport": "demo-inbox", "payments": "demo-ledger"})
	})
	if service == "behaviour" {
		mux.HandleFunc("GET /api/behaviour/history", func(w http.ResponseWriter, r *http.Request) {
			h, err := HistoryFor(r.Context(), a.DB, r.Header.Get("X-User-ID"), a.Now())
			if err != nil {
				a.dbError(w, err)
				return
			}
			reply(w, 200, map[string]any{"history": h})
		})
	}
	if service == "pusher" {
		mux.HandleFunc("GET /api/notifications", func(w http.ResponseWriter, r *http.Request) { a.table(w, r, "completed_push") })
		if a.Demo {
			mux.HandleFunc("POST /api/pusher/tick", func(w http.ResponseWriter, r *http.Request) {
				n, err := Deliver(r.Context(), a.DB, a.Now())
				if err != nil {
					a.dbError(w, err)
					return
				}
				reply(w, 200, map[string]int{"delivered": n})
			})
		}
	}
	if service == "payment-scheduler" {
		mux.HandleFunc("GET /api/schedules", func(w http.ResponseWriter, r *http.Request) {
			rows, err := a.DB.QueryContext(r.Context(), `SELECT `+scheduleColumns+` FROM scheduled_cis_payments WHERE user_id=$1 AND status='active' ORDER BY next_run_at,id`, r.Header.Get("X-User-ID"))
			if err != nil {
				a.dbError(w, err)
				return
			}
			defer rows.Close()
			out := []Schedule{}
			for rows.Next() {
				s, e := scanSchedule(rows)
				if e != nil {
					a.dbError(w, e)
					return
				}
				out = append(out, s)
			}
			if err = rows.Err(); err != nil {
				a.dbError(w, err)
				return
			}
			reply(w, 200, out)
		})
		mux.HandleFunc("GET /api/schedules/{id}", func(w http.ResponseWriter, r *http.Request) {
			s, err := scanSchedule(a.DB.QueryRowContext(r.Context(), `SELECT `+scheduleColumns+` FROM scheduled_cis_payments WHERE user_id=$1 AND id=$2 AND status='active'`, r.Header.Get("X-User-ID"), r.PathValue("id")))
			if err != nil {
				a.dbError(w, err)
				return
			}
			reply(w, 200, s)
		})
		save := func(w http.ResponseWriter, r *http.Request) {
			var input ScheduleInput
			if !decode(w, r, &input) {
				return
			}
			if _, err := input.Validate(a.Now()); err != nil {
				failure(w, 400, err.Error())
				return
			}
			if !input.Confirmed {
				failure(w, 400, "explicit confirmation required")
				return
			}
			key := r.Header.Get("Idempotency-Key")
			if r.Method == "POST" && (len(key) < 8 || len(key) > 128) {
				failure(w, 400, "Idempotency-Key of 8..128 characters required")
				return
			}
			s, err := SaveSchedule(r.Context(), a.DB, r.Header.Get("X-User-ID"), r.PathValue("id"), key, input, a.Now())
			if err != nil {
				a.dbError(w, err)
				return
			}
			status := 200
			if r.Method == "POST" {
				status = 201
			}
			reply(w, status, s)
		}
		mux.HandleFunc("POST /api/schedules", save)
		mux.HandleFunc("PUT /api/schedules/{id}", save)
		mux.HandleFunc("DELETE /api/schedules/{id}", func(w http.ResponseWriter, r *http.Request) {
			result, err := a.DB.ExecContext(r.Context(), `UPDATE scheduled_cis_payments SET status='cancelled',updated_at=$3 WHERE user_id=$1 AND id=$2`, r.Header.Get("X-User-ID"), r.PathValue("id"), a.Now())
			if err != nil {
				a.dbError(w, err)
				return
			}
			n, _ := result.RowsAffected()
			if n == 0 {
				failure(w, 404, "not found")
				return
			}
			reply(w, 204, nil)
		})
	}
	if a.Demo {
		mux.HandleFunc("GET /api/tables/{table}", func(w http.ResponseWriter, r *http.Request) { a.table(w, r, r.PathValue("table")) })
	}
	// This is a local demo identity boundary. Production must replace this header
	// with a trusted authenticated gateway; clients must not set their own identity.
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "no-store")
		ctx, cancel := context.WithTimeout(r.Context(), 15*time.Second)
		defer cancel()
		r = r.WithContext(ctx)
		if r.URL.Path != "/health" {
			user := r.Header.Get("X-User-ID")
			if user == "" {
				failure(w, 401, "X-User-ID demo identity required")
				return
			}
			var exists bool
			if err := a.DB.QueryRowContext(ctx, `SELECT EXISTS(SELECT 1 FROM clients WHERE user_id=$1)`, user).Scan(&exists); err != nil {
				a.dbError(w, err)
				return
			}
			if !exists {
				failure(w, 401, "unknown demo user")
				return
			}
		}
		mux.ServeHTTP(w, r)
	})
}
func (a API) dbError(w http.ResponseWriter, err error) {
	if errors.Is(err, sql.ErrNoRows) {
		failure(w, 404, "not found")
	} else if errors.Is(err, ErrConflict) {
		failure(w, 409, err.Error())
	} else {
		log.Printf("database: %v", err)
		failure(w, 500, "database operation failed")
	}
}
func (a API) table(w http.ResponseWriter, r *http.Request, name string) {
	allowed := map[string]bool{"clients": true, "transfers": true, "behaviour_push": true, "completed_push": true, "scheduled_cis_payments": true, "global_notifications": true, "scheduled_payment_runs": true}
	if !allowed[name] {
		failure(w, 404, "unknown table")
		return
	}
	where := ` WHERE user_id=$1`
	args := []any{r.Header.Get("X-User-ID")}
	if name == "global_notifications" {
		where = ""
		args = nil
	}
	if name == "scheduled_payment_runs" {
		where = ` WHERE schedule_id IN (SELECT id FROM scheduled_cis_payments WHERE user_id=$1)`
	}
	rows, err := a.DB.QueryContext(r.Context(), `SELECT row_to_json(t) FROM (SELECT * FROM `+name+where+` LIMIT 100) t`, args...)
	if err != nil {
		a.dbError(w, err)
		return
	}
	defer rows.Close()
	out := []json.RawMessage{}
	for rows.Next() {
		var data json.RawMessage
		if err = rows.Scan(&data); err != nil {
			a.dbError(w, err)
			return
		}
		out = append(out, data)
	}
	if err = rows.Err(); err != nil {
		a.dbError(w, err)
		return
	}
	reply(w, 200, out)
}
func Serve(service string) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	db, err := store.Open(ctx, os.Getenv("DATABASE_URL"))
	if err != nil {
		return err
	}
	defer db.Close()
	if err = store.Migrate(ctx, db); err != nil {
		return err
	}
	now := time.Now
	demo := os.Getenv("DEMO_MODE") == "true"
	if value := os.Getenv("DEMO_NOW"); value != "" {
		if !demo {
			return errors.New("DEMO_NOW requires DEMO_MODE=true")
		}
		fixed, e := time.Parse(time.RFC3339, value)
		if e != nil {
			return e
		}
		now = func() time.Time { return fixed }
	}
	interval := 30
	if value := os.Getenv("POLL_SECONDS"); value != "" {
		interval, err = strconv.Atoi(value)
		if err != nil || interval < 1 {
			return errors.New("POLL_SECONDS must be a positive integer")
		}
	}
	addr := os.Getenv("HTTP_ADDR")
	if addr == "" {
		addr = "127.0.0.1:8080"
	}
	server := &http.Server{Addr: addr, Handler: (API{db, now, demo}).Handler(service), ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 20 * time.Second, WriteTimeout: 20 * time.Second, IdleTimeout: 60 * time.Second}
	done := make(chan struct{})
	go func() {
		defer close(done)
		tick := time.NewTicker(time.Duration(interval) * time.Second)
		defer tick.Stop()
		for {
			iteration, cancel := context.WithTimeout(ctx, 20*time.Second)
			var e error
			if service == "pusher" {
				_, e = Deliver(iteration, db, now())
			}
			if service == "payment-scheduler" && demo {
				_, e = ExecuteDue(iteration, db, now())
			}
			cancel()
			if e != nil && ctx.Err() == nil {
				log.Printf("%s worker: %v", service, e)
			}
			select {
			case <-ctx.Done():
				return
			case <-tick.C:
			}
		}
	}()
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		server.Shutdown(shutdown)
	}()
	log.Printf("%s listening on %s (demo=%t)", service, addr, demo)
	err = server.ListenAndServe()
	stop()
	<-done
	if err == http.ErrServerClosed {
		return nil
	}
	return err
}
