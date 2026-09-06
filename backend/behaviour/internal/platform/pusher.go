// Package platform implements local demo services sharing the behaviour database.
package platform

import (
	"ai-hack/behaviour/internal/model"
	"context"
	"database/sql"
	"time"
)

const BehaviourMessage = "Пришло время отправить деньги за границу, сейчас отличное время"

// Deliver serializes decisions per client, across every pusher instance. Delivery
// to the demo inbox and accounting are the SAME commit; there is no external send.
func Deliver(ctx context.Context, db *sql.DB, now time.Time) (int, error) {
	rows, err := db.QueryContext(ctx, `SELECT user_id FROM clients ORDER BY user_id`)
	if err != nil {
		return 0, err
	}
	var users []string
	for rows.Next() {
		var u string
		if err = rows.Scan(&u); err != nil {
			rows.Close()
			return 0, err
		}
		users = append(users, u)
	}
	err = rows.Err()
	rows.Close()
	if err != nil {
		return 0, err
	}
	total := 0
	for _, user := range users {
		n, e := deliverUser(ctx, db, user, now)
		if e != nil {
			return total, e
		}
		total += n
	}
	return total, nil
}

func deliverUser(ctx context.Context, db *sql.DB, user string, now time.Time) (int, error) {
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return 0, err
	}
	defer tx.Rollback()
	var timezone string
	err = tx.QueryRowContext(ctx, `SELECT timezone FROM clients WHERE user_id=$1 FOR UPDATE SKIP LOCKED`, user).Scan(&timezone)
	if err == sql.ErrNoRows {
		return 0, nil
	}
	if err != nil {
		return 0, err
	}
	var week string
	var used int
	err = tx.QueryRowContext(ctx, `SELECT date_trunc('week',$1::timestamptz AT TIME ZONE $2)::date::text`, now, timezone).Scan(&week)
	if err != nil {
		return 0, err
	}
	err = tx.QueryRowContext(ctx, `SELECT count(*) FROM completed_push WHERE user_id=$1 AND week_start=$2::date`, user, week).Scan(&used)
	if err != nil {
		return 0, err
	}
	if used >= 2 {
		return 0, nil
	}
	// Only the latest supported snapshot, a real credit on the predicted cycle day,
	// no completed transfer since that credit, and half-open delivery windows.
	rows, err := tx.QueryContext(ctx, `
 WITH candidates AS (
 SELECT DISTINCT 'behaviour:'||b.cycle_date||':'||b.source_currency||':'||b.destination_currency||':'||b.corridor AS key,
 'behaviour' AS type, $4::text AS body, 0 AS priority, b.cycle_date::timestamptz AS created
 FROM behaviour_push b
 WHERE b.user_id=$1 AND b.model_version=$3 AND b.as_of=(SELECT max(as_of) FROM behaviour_push WHERE model_version=$3 AND as_of<=$2)
 AND b.window_start<=$2 AND $2<b.window_end
 AND EXISTS (SELECT 1 FROM transfers i WHERE i.user_id=b.user_id AND i.kind='income' AND i.status='completed'
	   AND i.source_currency=b.source_currency
	   AND extract(day FROM i.occurred_at AT TIME ZONE b.timezone)=b.income_day
	   AND (i.occurred_at AT TIME ZONE b.timezone)::date BETWEEN b.cycle_date - 3 AND b.cycle_date
   AND i.occurred_at<=$2 AND i.recorded_at<=$2
   AND NOT EXISTS (SELECT 1 FROM transfers t WHERE t.user_id=b.user_id AND t.kind='cross_border' AND t.status='completed'
    AND t.source_currency=b.source_currency AND t.destination_currency=b.destination_currency AND t.corridor=b.corridor
    AND t.occurred_at>=i.occurred_at AND t.occurred_at<=$2 AND t.recorded_at<=$2))
 UNION ALL
 SELECT 'global:'||id,type,body,CASE type WHEN 'history' THEN 1 ELSE 2 END,created_at
 FROM global_notifications WHERE created_at<=$2 AND expires_at>$2
 ) SELECT key,type,body FROM candidates c
 WHERE NOT EXISTS (SELECT 1 FROM completed_push p WHERE p.user_id=$1 AND p.source_key=c.key)
 ORDER BY priority,created,key LIMIT $5`, user, now, model.Version, BehaviourMessage, 2-used)
	if err != nil {
		return 0, err
	}
	type push struct{ key, kind, body string }
	var pending []push
	for rows.Next() {
		var p push
		if err = rows.Scan(&p.key, &p.kind, &p.body); err != nil {
			rows.Close()
			return 0, err
		}
		pending = append(pending, p)
	}
	err = rows.Err()
	rows.Close()
	if err != nil {
		return 0, err
	}
	for i, p := range pending {
		_, err = tx.ExecContext(ctx, `INSERT INTO completed_push(user_id,source_key,type,body,sent_at,week_start,slot) VALUES($1,$2,$3,$4,$5,$6::date,$7)`, user, p.key, p.kind, p.body, now, week, used+i+1)
		if err != nil {
			return 0, err
		}
	}
	return len(pending), tx.Commit()
}
