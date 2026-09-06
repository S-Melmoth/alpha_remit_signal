CREATE TABLE IF NOT EXISTS clients (
    user_id text PRIMARY KEY,
    timezone text NOT NULL
);

-- amount_minor is the amount debited/credited in source_currency, without FX conversion.
CREATE TABLE IF NOT EXISTS transfers (
    id text PRIMARY KEY,
    user_id text NOT NULL REFERENCES clients(user_id),
    kind text NOT NULL CHECK (kind IN ('income', 'cross_border', 'domestic')),
    status text NOT NULL CHECK (status IN ('completed', 'pending', 'failed')),
    occurred_at timestamptz NOT NULL,
    recorded_at timestamptz NOT NULL DEFAULT now(),
    amount_minor bigint NOT NULL CHECK (amount_minor > 0),
    source_currency text NOT NULL CHECK (source_currency ~ '^[A-Z]{3}$'),
    destination_currency text,
    corridor text,
    CHECK (kind <> 'cross_border' OR
           (destination_currency IS NOT NULL AND destination_currency ~ '^[A-Z]{3}$'
            AND corridor IS NOT NULL AND corridor <> ''))
);
CREATE INDEX IF NOT EXISTS transfers_user_time_idx ON transfers(user_id, occurred_at);

-- Recomputable as-of snapshots of candidates; this is NOT a push delivery queue.
CREATE TABLE IF NOT EXISTS behaviour_push (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id text NOT NULL REFERENCES clients(user_id),
    as_of timestamptz NOT NULL,
    model_version text NOT NULL,
    source_currency text NOT NULL,
    destination_currency text NOT NULL,
    corridor text NOT NULL,
    income_day integer NOT NULL CHECK (income_day BETWEEN 1 AND 31),
    cycle_date date NOT NULL,
    window_start timestamptz NOT NULL,
    window_end timestamptz NOT NULL CHECK (window_end > window_start),
    timezone text NOT NULL,
    typical_amount_minor bigint NOT NULL CHECK (typical_amount_minor > 0),
    amount_p25_minor bigint NOT NULL,
    amount_p75_minor bigint NOT NULL,
    sample_count integer NOT NULL CHECK (sample_count >= 3),
    support_ratio double precision NOT NULL CHECK (support_ratio BETWEEN 0 AND 1),
    status text NOT NULL DEFAULT 'candidate' CHECK (status = 'candidate'),
    metadata jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (user_id, as_of, model_version, source_currency, destination_currency,
            corridor, income_day, cycle_date, window_start)
);
CREATE INDEX IF NOT EXISTS behaviour_push_window_idx ON behaviour_push(window_start, window_end);
