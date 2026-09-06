package store

import (
	"context"
	"database/sql"
)

// SeedServices extends the existing fixture without changing baseline test counts.
func SeedServices(ctx context.Context, db *sql.DB) error {
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	_, err = tx.ExecContext(ctx, `
 UPDATE transfers SET recipient='{"name":"Фаррух Саидов","phone":"+79265775128","bank":"Душанбе Сити Банк","country":"Таджикистан"}',purpose='Безвозмездный перевод на текущие расходы'
 WHERE id LIKE 'demo-%' AND kind='cross_border' AND recipient IS NULL;
 INSERT INTO transfers(id,user_id,kind,status,occurred_at,recorded_at,amount_minor,source_currency)
 VALUES('demo-september-income','demo-regular','income','completed','2026-09-05T09:00:00+03:00','2026-09-05T09:00:00+03:00',8000000,'RUB') ON CONFLICT DO NOTHING;
 INSERT INTO global_notifications(id,type,body,created_at,expires_at,metadata) VALUES
 ('demo-history','history','По историческим данным сейчас подходящий момент для перевода близким. Проверьте курс в приложении.','2026-09-05T17:58:00+03:00','2026-09-06T18:00:00+03:00','{"synthetic":true,"source":"Open Data","model":"history-ml-demo"}'),
 ('demo-news','news','Новости указывают на благоприятный момент для перевода. Посмотрите актуальные условия.','2026-09-05T17:59:00+03:00','2026-09-06T18:00:00+03:00','{"synthetic":true,"source":"Open Data","model":"news-ml-demo"}') ON CONFLICT DO NOTHING;
 INSERT INTO scheduled_cis_payments(id,user_id,amount_minor,source_currency,destination_currency,corridor,recipient,purpose,start_date,day_of_month,local_time,timezone,next_run_at,created_at,updated_at,confirmed_at,request_key)
 VALUES('demo-schedule','demo-regular',1600000,'RUB','TJS','RU-TJ','{"name":"Фаррух Саидов","phone":"+79265775128","bank":"Душанбе Сити Банк","country":"Таджикистан"}','Безвозмездный перевод на текущие расходы','2026-10-05',5,'18:30','Europe/Moscow','2026-10-05T18:30:00+03:00','2026-09-05T18:00:00+03:00','2026-09-05T18:00:00+03:00','2026-09-05T18:00:00+03:00','demo-seed-schedule') ON CONFLICT DO NOTHING;`)
	if err != nil {
		return err
	}
	return tx.Commit()
}
