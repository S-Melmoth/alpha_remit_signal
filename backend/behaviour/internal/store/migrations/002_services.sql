ALTER TABLE transfers ADD COLUMN IF NOT EXISTS recipient jsonb;
ALTER TABLE transfers ADD COLUMN IF NOT EXISTS purpose text;
CREATE TABLE IF NOT EXISTS global_notifications (
 id text PRIMARY KEY,
 type text NOT NULL CHECK (type IN ('news','history')),
 body text NOT NULL CHECK (length(body) BETWEEN 1 AND 1000),
 created_at timestamptz NOT NULL DEFAULT now(),
 expires_at timestamptz NOT NULL DEFAULT now() + interval '1 day',
 metadata jsonb NOT NULL DEFAULT '{}',
 CHECK (expires_at > created_at)
);
CREATE TABLE IF NOT EXISTS completed_push (
 id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
 user_id text NOT NULL REFERENCES clients(user_id),
 source_key text NOT NULL,
 type text NOT NULL CHECK (type IN ('behaviour','history','news')),
 body text NOT NULL,
 sent_at timestamptz NOT NULL,
 week_start date NOT NULL,
 slot integer NOT NULL CHECK (slot BETWEEN 1 AND 2),
 transport text NOT NULL DEFAULT 'demo-inbox' CHECK (transport = 'demo-inbox'),
 UNIQUE(user_id,source_key),
 UNIQUE(user_id,week_start,slot)
);
CREATE INDEX IF NOT EXISTS completed_push_user_time ON completed_push(user_id,sent_at);
CREATE TABLE IF NOT EXISTS scheduled_cis_payments (
 id text PRIMARY KEY,
 user_id text NOT NULL REFERENCES clients(user_id),
 amount_minor bigint NOT NULL CHECK (amount_minor BETWEEN 10000 AND 10000000),
 source_currency text NOT NULL CHECK (source_currency = 'RUB'),
 destination_currency text NOT NULL CHECK (destination_currency IN ('TJS','UZS','KGS','BYN')),
 corridor text NOT NULL,
 recipient jsonb NOT NULL,
 purpose text NOT NULL,
 start_date date NOT NULL,
 day_of_month integer NOT NULL CHECK (day_of_month BETWEEN 1 AND 31),
 local_time text NOT NULL,
 timezone text NOT NULL,
 next_run_at timestamptz NOT NULL,
 status text NOT NULL DEFAULT 'active' CHECK (status IN ('active','cancelled')),
 created_at timestamptz NOT NULL,
 updated_at timestamptz NOT NULL,
 confirmed_at timestamptz NOT NULL,
 request_key text NOT NULL,
 UNIQUE(user_id, request_key)
);
CREATE INDEX IF NOT EXISTS scheduled_cis_due ON scheduled_cis_payments(next_run_at) WHERE status='active';
CREATE TABLE IF NOT EXISTS scheduled_payment_runs (
 schedule_id text NOT NULL REFERENCES scheduled_cis_payments(id),
 due_at timestamptz NOT NULL,
 transfer_id text NOT NULL UNIQUE REFERENCES transfers(id),
 executed_at timestamptz NOT NULL,
 PRIMARY KEY(schedule_id,due_at)
);
