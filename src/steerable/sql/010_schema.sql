-- steerable: schema, tables, immutability. Runs as steerable_admin (SET ROLE in db init).
-- Stock PostgreSQL 14+; no extensions.

CREATE SCHEMA steerable;
REVOKE ALL ON SCHEMA steerable FROM PUBLIC;

CREATE TYPE steerable.run_status AS ENUM ('created', 'running', 'finished', 'aborted');
CREATE TYPE steerable.boundary AS ENUM ('step', 'validation', 'frozen');
CREATE TYPE steerable.policy_mode AS ENUM ('auto', 'review', 'frozen');
CREATE TYPE steerable.actor_kind AS ENUM ('agent', 'human', 'system');
CREATE TYPE steerable.event_kind AS ENUM ('approved', 'rejected', 'applied', 'superseded', 'trainer_rejected');

CREATE TABLE steerable.runs (
    id                     bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name                   text NOT NULL UNIQUE CHECK (name ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$'),
    status                 steerable.run_status NOT NULL DEFAULT 'created',
    agent_enabled          boolean NOT NULL DEFAULT true,
    launch_config          jsonb NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(launch_config) = 'object'),
    max_proposals_per_hour integer NOT NULL DEFAULT 30 CHECK (max_proposals_per_hour > 0),
    created_at             timestamptz NOT NULL DEFAULT now(),
    updated_at             timestamptz NOT NULL DEFAULT now()
);

CREATE FUNCTION steerable.schema_is_valid(s jsonb) RETURNS boolean
LANGUAGE plpgsql IMMUTABLE STRICT SET search_path = pg_catalog, pg_temp AS $$
DECLARE
    t text;
    v jsonb;
BEGIN
    IF jsonb_typeof(s) <> 'object' THEN RETURN false; END IF;
    t := s ->> 'type';
    IF t IN ('number', 'integer') THEN
        IF s ? 'min' AND jsonb_typeof(s -> 'min') <> 'number' THEN RETURN false; END IF;
        IF s ? 'max' AND jsonb_typeof(s -> 'max') <> 'number' THEN RETURN false; END IF;
        IF s ? 'min' AND s ? 'max' AND (s ->> 'min')::numeric > (s ->> 'max')::numeric THEN RETURN false; END IF;
        RETURN true;
    ELSIF t = 'boolean' THEN
        RETURN true;
    ELSIF t = 'enum' THEN
        IF coalesce(jsonb_typeof(s -> 'values'), '') <> 'array' OR jsonb_array_length(s -> 'values') = 0 THEN RETURN false; END IF;
        FOR v IN SELECT * FROM jsonb_array_elements(s -> 'values') LOOP
            IF jsonb_typeof(v) NOT IN ('string', 'number') THEN RETURN false; END IF;
        END LOOP;
        RETURN true;
    END IF;
    RETURN false;
END $$;

CREATE FUNCTION steerable.value_matches(s jsonb, v jsonb) RETURNS boolean
LANGUAGE plpgsql IMMUTABLE STRICT SET search_path = pg_catalog, pg_temp AS $$
DECLARE
    t text := s ->> 'type';
    n numeric;
BEGIN
    IF t IN ('number', 'integer') THEN
        IF jsonb_typeof(v) <> 'number' THEN RETURN false; END IF;
        n := (v #>> '{}')::numeric;
        IF t = 'integer' AND n <> trunc(n) THEN RETURN false; END IF;
        IF s ? 'min' AND n < (s ->> 'min')::numeric THEN RETURN false; END IF;
        IF s ? 'max' AND n > (s ->> 'max')::numeric THEN RETURN false; END IF;
        RETURN true;
    ELSIF t = 'boolean' THEN
        RETURN jsonb_typeof(v) = 'boolean';
    ELSIF t = 'enum' THEN
        RETURN jsonb_typeof(v) IN ('string', 'number') AND (s -> 'values') @> jsonb_build_array(v);
    END IF;
    RETURN false;
END $$;

CREATE TABLE steerable.policies (
    run_id            bigint NOT NULL REFERENCES steerable.runs (id),
    key               text NOT NULL CHECK (key ~ '^[a-z][a-z0-9_.]{0,63}$'),
    schema            jsonb NOT NULL CHECK (steerable.schema_is_valid(schema)),
    boundary          steerable.boundary NOT NULL DEFAULT 'step',
    mode              steerable.policy_mode NOT NULL DEFAULT 'review',
    max_rel_change    numeric CHECK (max_rel_change >= 1),
    min_steps_between integer NOT NULL DEFAULT 0 CHECK (min_steps_between >= 0),
    max_per_hour      integer NOT NULL DEFAULT 6 CHECK (max_per_hour > 0),
    note              text,
    PRIMARY KEY (run_id, key)
);

CREATE TABLE steerable.proposals (
    id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id     bigint NOT NULL,
    key        text NOT NULL,
    value      jsonb NOT NULL,
    previous   jsonb,
    actor      text NOT NULL,
    actor_kind steerable.actor_kind NOT NULL,
    reason     text NOT NULL CHECK (length(reason) BETWEEN 1 AND 1000),
    created_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (run_id, key) REFERENCES steerable.policies (run_id, key)
);
CREATE INDEX proposals_run_key_created ON steerable.proposals (run_id, key, created_at DESC);
CREATE INDEX proposals_run_created ON steerable.proposals (run_id, created_at DESC);

CREATE TABLE steerable.events (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    proposal_id bigint NOT NULL REFERENCES steerable.proposals (id),
    kind        steerable.event_kind NOT NULL,
    step        integer CHECK (step IS NULL OR step >= 0),
    note        text CHECK (note IS NULL OR length(note) <= 2000),
    actor       text NOT NULL,
    at          timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX events_proposal ON steerable.events (proposal_id, id DESC);

CREATE TABLE steerable.metrics (
    run_id bigint NOT NULL REFERENCES steerable.runs (id),
    step   integer NOT NULL CHECK (step >= 0),
    name   text NOT NULL CHECK (name ~ '^[A-Za-z0-9][A-Za-z0-9_./:@-]{0,63}$'),
    value  double precision NOT NULL,
    at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX metrics_run_name_step ON steerable.metrics (run_id, name, step DESC);
CREATE INDEX metrics_run_step ON steerable.metrics (run_id, step DESC);

-- Proposals and events are append-only for every role, the owner included.
CREATE FUNCTION steerable.refuse_mutation() RETURNS trigger
LANGUAGE plpgsql SET search_path = pg_catalog, pg_temp AS $$
BEGIN
    RAISE EXCEPTION USING
        ERRCODE = 'SE000',
        MESSAGE = format('%s.%s is append-only: %s is not allowed', TG_TABLE_SCHEMA, TG_TABLE_NAME, TG_OP);
END $$;

CREATE TRIGGER proposals_append_only BEFORE UPDATE OR DELETE ON steerable.proposals
    FOR EACH ROW EXECUTE FUNCTION steerable.refuse_mutation();
CREATE TRIGGER proposals_no_truncate BEFORE TRUNCATE ON steerable.proposals
    FOR EACH STATEMENT EXECUTE FUNCTION steerable.refuse_mutation();
CREATE TRIGGER events_append_only BEFORE UPDATE OR DELETE ON steerable.events
    FOR EACH ROW EXECUTE FUNCTION steerable.refuse_mutation();
CREATE TRIGGER events_no_truncate BEFORE TRUNCATE ON steerable.events
    FOR EACH STATEMENT EXECUTE FUNCTION steerable.refuse_mutation();

CREATE FUNCTION steerable.touch_run() RETURNS trigger
LANGUAGE plpgsql SET search_path = pg_catalog, pg_temp AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END $$;
CREATE TRIGGER runs_touch BEFORE UPDATE ON steerable.runs
    FOR EACH ROW EXECUTE FUNCTION steerable.touch_run();
