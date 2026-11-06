-- SECURITY DEFINER entry points. Owned by steerable_admin, fixed search_path, every
-- argument validated, no dynamic SQL. Rejections raise with a stable SQLSTATE in class
-- 'SE' and a plain message the caller may return to an LLM verbatim, as data.
--
--   SE000  append-only table mutation refused
--   SE001  run missing or not running
--   SE002  run frozen (agent_enabled = false)
--   SE003  key has no policy
--   SE004  key frozen by policy
--   SE005  value does not match the key's schema
--   SE006  relative change exceeds max_rel_change
--   SE007  too few steps since the key last changed
--   SE008  rate limit (per key or per run)
--   SE009  bad argument
--   SE010  proposal not in a state that allows this transition

CREATE FUNCTION steerable.caller_kind() RETURNS steerable.actor_kind
LANGUAGE sql STABLE SET search_path = pg_catalog, pg_temp AS $$
    SELECT CASE
        WHEN pg_has_role(session_user, 'steerable_agent', 'MEMBER') THEN 'agent'::steerable.actor_kind
        WHEN pg_has_role(session_user, 'steerable_trainer', 'MEMBER') THEN 'system'::steerable.actor_kind
        ELSE 'human'::steerable.actor_kind
    END
$$;

CREATE FUNCTION steerable.propose(p_run_id bigint, p_key text, p_value jsonb, p_reason text)
RETURNS TABLE (proposal_id bigint, status text)
LANGUAGE plpgsql SECURITY DEFINER SET search_path = steerable, pg_temp AS $$
DECLARE
    r         steerable.runs%ROWTYPE;
    pol       steerable.policies%ROWTYPE;
    prev      jsonb;
    pv        numeric;
    nv        numeric;
    ratio     numeric;
    cur_step  integer;
    last_step integer;
    n_key     integer;
    n_run     integer;
    new_id    bigint;
    st        text;
BEGIN
    IF p_run_id IS NULL OR p_key IS NULL OR p_value IS NULL OR p_reason IS NULL THEN
        RAISE EXCEPTION USING ERRCODE = 'SE009', MESSAGE = 'run_id, key, value and reason are all required';
    END IF;
    IF length(p_reason) < 1 OR length(p_reason) > 1000 THEN
        RAISE EXCEPTION USING ERRCODE = 'SE009', MESSAGE = 'reason must be 1 to 1000 characters';
    END IF;
    IF length(p_key) > 64 THEN
        RAISE EXCEPTION USING ERRCODE = 'SE009', MESSAGE = 'key is longer than 64 characters';
    END IF;
    IF pg_column_size(p_value) > 4096 THEN
        RAISE EXCEPTION USING ERRCODE = 'SE009', MESSAGE = 'value is larger than 4096 bytes';
    END IF;

    SELECT * INTO r FROM steerable.runs WHERE id = p_run_id FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION USING ERRCODE = 'SE001', MESSAGE = format('run %s does not exist', p_run_id);
    END IF;
    IF r.status <> 'running' THEN
        RAISE EXCEPTION USING ERRCODE = 'SE001',
            MESSAGE = format('run %s is %s, not running; nothing can be proposed', p_run_id, r.status);
    END IF;
    IF NOT r.agent_enabled THEN
        RAISE EXCEPTION USING ERRCODE = 'SE002',
            MESSAGE = format('run %s is frozen: proposals are disabled until an admin re-enables them', p_run_id);
    END IF;

    SELECT * INTO pol FROM steerable.policies WHERE run_id = p_run_id AND key = p_key;
    IF NOT FOUND THEN
        RAISE EXCEPTION USING ERRCODE = 'SE003',
            MESSAGE = format('key %s has no policy for run %s and cannot be proposed', p_key, p_run_id);
    END IF;
    IF pol.mode = 'frozen' OR pol.boundary = 'frozen' THEN
        RAISE EXCEPTION USING ERRCODE = 'SE004',
            MESSAGE = format('key %s is frozen for run %s%s', p_key, p_run_id,
                             coalesce(': ' || pol.note, ''));
    END IF;
    IF NOT steerable.value_matches(pol.schema, p_value) THEN
        RAISE EXCEPTION USING ERRCODE = 'SE005',
            MESSAGE = format('value %s does not match the schema %s for key %s', p_value, pol.schema, p_key);
    END IF;

    SELECT c.value INTO prev FROM steerable.current_config c WHERE c.run_id = p_run_id AND c.key = p_key;
    IF pol.max_rel_change IS NOT NULL AND prev IS NOT NULL
       AND jsonb_typeof(prev) = 'number' AND jsonb_typeof(p_value) = 'number' THEN
        pv := (prev #>> '{}')::numeric;
        nv := (p_value #>> '{}')::numeric;
        IF pv <> 0 THEN
            IF nv = 0 OR sign(nv) <> sign(pv) THEN
                RAISE EXCEPTION USING ERRCODE = 'SE006',
                    MESSAGE = format('change of %s from %s to %s has no finite ratio; max_rel_change is %sx',
                                     p_key, prev, p_value, rtrim(rtrim(pol.max_rel_change::text, '0'), '.'));
            END IF;
            ratio := greatest(nv / pv, pv / nv);
            IF ratio > pol.max_rel_change THEN
                RAISE EXCEPTION USING ERRCODE = 'SE006',
                    MESSAGE = format('change of %s from %s to %s is %sx; max_rel_change is %sx',
                                     p_key, prev, p_value, rtrim(rtrim(round(ratio, 3)::text, '0'), '.'), rtrim(rtrim(pol.max_rel_change::text, '0'), '.'));
            END IF;
        END IF;
    END IF;

    SELECT greatest(
        (SELECT max(m.step) FROM steerable.metrics m WHERE m.run_id = p_run_id),
        (SELECT max(e.step) FROM steerable.events e JOIN steerable.proposals p ON p.id = e.proposal_id
         WHERE p.run_id = p_run_id)
    ) INTO cur_step;
    SELECT max(e.step) INTO last_step
    FROM steerable.events e JOIN steerable.proposals p ON p.id = e.proposal_id
    WHERE p.run_id = p_run_id AND p.key = p_key AND e.kind = 'applied';
    IF pol.min_steps_between > 0 AND last_step IS NOT NULL
       AND coalesce(cur_step, last_step) - last_step < pol.min_steps_between THEN
        RAISE EXCEPTION USING ERRCODE = 'SE007',
            MESSAGE = format('key %s was last applied at step %s; it may change again at step %s (current step %s)',
                             p_key, last_step, last_step + pol.min_steps_between, coalesce(cur_step, last_step));
    END IF;

    SELECT count(*) INTO n_key FROM steerable.proposals
    WHERE run_id = p_run_id AND key = p_key AND created_at > now() - interval '1 hour';
    IF n_key >= pol.max_per_hour THEN
        RAISE EXCEPTION USING ERRCODE = 'SE008',
            MESSAGE = format('key %s already has %s proposals in the last hour; max_per_hour is %s',
                             p_key, n_key, pol.max_per_hour);
    END IF;
    SELECT count(*) INTO n_run FROM steerable.proposals
    WHERE run_id = p_run_id AND created_at > now() - interval '1 hour';
    IF n_run >= r.max_proposals_per_hour THEN
        RAISE EXCEPTION USING ERRCODE = 'SE008',
            MESSAGE = format('run %s already has %s proposals in the last hour; the run limit is %s',
                             p_run_id, n_run, r.max_proposals_per_hour);
    END IF;

    INSERT INTO steerable.events (proposal_id, kind, note, actor)
    SELECT ps.proposal_id, 'superseded', format('superseded by a newer proposal for %s', p_key), session_user
    FROM steerable.proposal_status ps
    WHERE ps.run_id = p_run_id AND ps.key = p_key AND ps.status IN ('pending', 'approved');

    INSERT INTO steerable.proposals (run_id, key, value, previous, actor, actor_kind, reason)
    VALUES (p_run_id, p_key, p_value, prev, session_user, steerable.caller_kind(), p_reason)
    RETURNING id INTO new_id;

    IF pol.mode = 'auto' THEN
        INSERT INTO steerable.events (proposal_id, kind, note, actor)
        VALUES (new_id, 'approved', 'auto-approved by policy', session_user);
        st := 'approved';
    ELSE
        st := 'pending';
    END IF;

    RETURN QUERY SELECT new_id, st;
END $$;

CREATE FUNCTION steerable.ack(p_proposal_id bigint, p_step integer, p_status text, p_note text DEFAULT NULL)
RETURNS void
LANGUAGE plpgsql SECURITY DEFINER SET search_path = steerable, pg_temp AS $$
DECLARE
    cur text;
BEGIN
    IF p_proposal_id IS NULL OR p_step IS NULL OR p_status IS NULL THEN
        RAISE EXCEPTION USING ERRCODE = 'SE009', MESSAGE = 'proposal_id, step and status are required';
    END IF;
    IF p_status NOT IN ('applied', 'trainer_rejected') THEN
        RAISE EXCEPTION USING ERRCODE = 'SE009',
            MESSAGE = format('ack status must be applied or trainer_rejected, not %s', p_status);
    END IF;
    IF p_step < 0 THEN
        RAISE EXCEPTION USING ERRCODE = 'SE009', MESSAGE = 'step must be >= 0';
    END IF;
    IF p_note IS NOT NULL AND length(p_note) > 2000 THEN
        RAISE EXCEPTION USING ERRCODE = 'SE009', MESSAGE = 'note is longer than 2000 characters';
    END IF;

    PERFORM 1 FROM steerable.proposals WHERE id = p_proposal_id FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION USING ERRCODE = 'SE010', MESSAGE = format('proposal %s does not exist', p_proposal_id);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM steerable.events WHERE proposal_id = p_proposal_id AND kind = 'approved') THEN
        RAISE EXCEPTION USING ERRCODE = 'SE010',
            MESSAGE = format('proposal %s has not been approved; it cannot be acknowledged', p_proposal_id);
    END IF;
    SELECT kind::text INTO cur FROM steerable.events
    WHERE proposal_id = p_proposal_id AND kind IN ('applied', 'trainer_rejected', 'rejected')
    ORDER BY id DESC LIMIT 1;
    IF cur IS NOT NULL THEN
        RAISE EXCEPTION USING ERRCODE = 'SE010',
            MESSAGE = format('proposal %s is already %s', p_proposal_id, cur);
    END IF;

    INSERT INTO steerable.events (proposal_id, kind, step, note, actor)
    VALUES (p_proposal_id, p_status::steerable.event_kind, p_step, p_note, session_user);
END $$;

CREATE FUNCTION steerable.decide(p_proposal_id bigint, p_status text, p_note text DEFAULT NULL)
RETURNS void
LANGUAGE plpgsql SECURITY DEFINER SET search_path = steerable, pg_temp AS $$
DECLARE
    cur text;
BEGIN
    IF p_proposal_id IS NULL OR p_status IS NULL THEN
        RAISE EXCEPTION USING ERRCODE = 'SE009', MESSAGE = 'proposal_id and status are required';
    END IF;
    IF p_status NOT IN ('approved', 'rejected') THEN
        RAISE EXCEPTION USING ERRCODE = 'SE009',
            MESSAGE = format('decision must be approved or rejected, not %s', p_status);
    END IF;
    IF p_note IS NOT NULL AND length(p_note) > 2000 THEN
        RAISE EXCEPTION USING ERRCODE = 'SE009', MESSAGE = 'note is longer than 2000 characters';
    END IF;

    PERFORM 1 FROM steerable.proposals WHERE id = p_proposal_id FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION USING ERRCODE = 'SE010', MESSAGE = format('proposal %s does not exist', p_proposal_id);
    END IF;
    SELECT status INTO cur FROM steerable.proposal_status WHERE proposal_id = p_proposal_id;
    IF cur <> 'pending' THEN
        RAISE EXCEPTION USING ERRCODE = 'SE010',
            MESSAGE = format('proposal %s is %s, not pending; only pending proposals can be decided',
                             p_proposal_id, cur);
    END IF;

    INSERT INTO steerable.events (proposal_id, kind, note, actor)
    VALUES (p_proposal_id, p_status::steerable.event_kind, p_note, session_user);
END $$;

-- The trainer records its launch config and default policies exactly once, while the run
-- is still 'created'. Policy rows an admin seeded beforehand are kept as they are.
CREATE FUNCTION steerable.launch(p_run_id bigint, p_config jsonb, p_policies jsonb DEFAULT '[]'::jsonb)
RETURNS void
LANGUAGE plpgsql SECURITY DEFINER SET search_path = steerable, pg_temp AS $$
DECLARE
    r steerable.runs%ROWTYPE;
    pol jsonb;
BEGIN
    IF p_run_id IS NULL OR p_config IS NULL OR jsonb_typeof(p_config) <> 'object' THEN
        RAISE EXCEPTION USING ERRCODE = 'SE009', MESSAGE = 'run_id and an object-valued config are required';
    END IF;
    IF p_policies IS NULL OR jsonb_typeof(p_policies) <> 'array' THEN
        RAISE EXCEPTION USING ERRCODE = 'SE009', MESSAGE = 'policies must be an array';
    END IF;
    IF pg_column_size(p_config) > 1048576 THEN
        RAISE EXCEPTION USING ERRCODE = 'SE009', MESSAGE = 'launch config is larger than 1 MiB';
    END IF;

    SELECT * INTO r FROM steerable.runs WHERE id = p_run_id FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION USING ERRCODE = 'SE001', MESSAGE = format('run %s does not exist', p_run_id);
    END IF;
    IF r.status <> 'created' THEN
        RAISE EXCEPTION USING ERRCODE = 'SE001',
            MESSAGE = format('run %s is %s; launch() is only allowed once, on a created run', p_run_id, r.status);
    END IF;

    FOR pol IN SELECT * FROM jsonb_array_elements(p_policies) LOOP
        INSERT INTO steerable.policies (run_id, key, schema, boundary, mode, max_rel_change,
                                        min_steps_between, max_per_hour, note)
        VALUES (p_run_id,
                pol ->> 'key',
                pol -> 'schema',
                coalesce((pol ->> 'boundary')::steerable.boundary, 'step'),
                coalesce((pol ->> 'mode')::steerable.policy_mode, 'review'),
                (pol ->> 'max_rel_change')::numeric,
                coalesce((pol ->> 'min_steps_between')::integer, 0),
                coalesce((pol ->> 'max_per_hour')::integer, 6),
                pol ->> 'note')
        ON CONFLICT (run_id, key) DO NOTHING;
    END LOOP;

    UPDATE steerable.runs SET launch_config = p_config, status = 'running' WHERE id = p_run_id;
END $$;

CREATE FUNCTION steerable.finish(p_run_id bigint, p_status text)
RETURNS void
LANGUAGE plpgsql SECURITY DEFINER SET search_path = steerable, pg_temp AS $$
BEGIN
    IF p_status NOT IN ('finished', 'aborted') THEN
        RAISE EXCEPTION USING ERRCODE = 'SE009',
            MESSAGE = format('finish status must be finished or aborted, not %s', p_status);
    END IF;
    UPDATE steerable.runs SET status = p_status::steerable.run_status
    WHERE id = p_run_id AND status = 'running';
    IF NOT FOUND THEN
        RAISE EXCEPTION USING ERRCODE = 'SE001', MESSAGE = format('run %s is not running', p_run_id);
    END IF;
END $$;
