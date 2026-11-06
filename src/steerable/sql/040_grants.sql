-- Grants. The three roles already exist (db init creates them). Nothing here is reachable
-- by PUBLIC; each role gets exactly the surface it needs and nothing else.

REVOKE ALL ON ALL TABLES IN SCHEMA steerable FROM PUBLIC;
REVOKE ALL ON ALL FUNCTIONS IN SCHEMA steerable FROM PUBLIC;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA steerable FROM PUBLIC;

-- steerable_agent: one function, two read views.
GRANT USAGE ON SCHEMA steerable TO steerable_agent;
GRANT EXECUTE ON FUNCTION steerable.propose(bigint, text, jsonb, text) TO steerable_agent;
GRANT SELECT ON steerable.run_state, steerable.history TO steerable_agent;

-- steerable_trainer: read what it must apply, write metrics, acknowledge, start and finish.
GRANT USAGE ON SCHEMA steerable TO steerable_trainer;
GRANT SELECT ON steerable.runs, steerable.policies, steerable.proposals, steerable.events,
                steerable.proposal_status, steerable.current_config, steerable.pending
    TO steerable_trainer;
GRANT SELECT, INSERT ON steerable.metrics TO steerable_trainer;
GRANT EXECUTE ON FUNCTION steerable.ack(bigint, integer, text, text) TO steerable_trainer;
GRANT EXECUTE ON FUNCTION steerable.launch(bigint, jsonb, jsonb) TO steerable_trainer;
GRANT EXECUTE ON FUNCTION steerable.finish(bigint, text) TO steerable_trainer;

-- steerable_admin owns everything; it additionally gets the human entry points explicitly
-- so the intent is visible even though ownership would already allow them.
GRANT EXECUTE ON FUNCTION steerable.propose(bigint, text, jsonb, text) TO steerable_admin;
GRANT EXECUTE ON FUNCTION steerable.decide(bigint, text, text) TO steerable_admin;
