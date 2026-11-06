-- Views. Status of a proposal is its latest event; effective config is the fold of
-- applied events per key over the run's launch config.

CREATE VIEW steerable.proposal_status AS
SELECT p.id AS proposal_id,
       p.run_id,
       p.key,
       p.value,
       p.previous,
       p.actor,
       p.actor_kind,
       p.reason,
       p.created_at,
       coalesce(e.kind::text, 'pending') AS status,
       e.step AS status_step,
       e.note AS status_note,
       e.at AS status_at
FROM steerable.proposals p
LEFT JOIN LATERAL (
    SELECT kind, step, note, at
    FROM steerable.events
    WHERE proposal_id = p.id
    ORDER BY id DESC
    LIMIT 1
) e ON true;

CREATE VIEW steerable.current_config AS
WITH applied AS (
    SELECT DISTINCT ON (p.run_id, p.key)
           p.run_id, p.key, p.value, p.id AS proposal_id, e.step AS applied_step, e.at AS applied_at
    FROM steerable.events e
    JOIN steerable.proposals p ON p.id = e.proposal_id
    WHERE e.kind = 'applied'
    ORDER BY p.run_id, p.key, e.id DESC
),
launch AS (
    SELECT r.id AS run_id, kv.key, kv.value
    FROM steerable.runs r, jsonb_each(r.launch_config) kv
)
SELECT coalesce(a.run_id, l.run_id) AS run_id,
       coalesce(a.key, l.key) AS key,
       coalesce(a.value, l.value) AS value,
       a.proposal_id,
       a.applied_step,
       a.applied_at
FROM launch l
FULL JOIN applied a ON a.run_id = l.run_id AND a.key = l.key;

CREATE VIEW steerable.pending AS
SELECT ps.*, pl.boundary
FROM steerable.proposal_status ps
JOIN steerable.policies pl ON pl.run_id = ps.run_id AND pl.key = ps.key
WHERE ps.status IN ('pending', 'approved');

CREATE VIEW steerable.history AS
SELECT ps.proposal_id,
       ps.run_id,
       ps.key,
       ps.value,
       ps.previous,
       ps.actor_kind,
       ps.reason,
       ps.created_at,
       ps.status,
       ps.status_step,
       ps.status_note,
       ps.status_at,
       coalesce((
           SELECT jsonb_agg(jsonb_build_object('kind', e.kind, 'step', e.step, 'note', e.note, 'at', e.at)
                            ORDER BY e.id)
           FROM steerable.events e WHERE e.proposal_id = ps.proposal_id
       ), '[]'::jsonb) AS events
FROM steerable.proposal_status ps;

CREATE VIEW steerable.run_state AS
SELECT r.id AS run_id,
       r.name,
       r.status,
       r.agent_enabled,
       greatest(
           (SELECT max(m.step) FROM steerable.metrics m WHERE m.run_id = r.id),
           (SELECT max(e.step) FROM steerable.events e JOIN steerable.proposals p ON p.id = e.proposal_id
            WHERE p.run_id = r.id)
       ) AS step,
       (SELECT coalesce(jsonb_object_agg(c.key, c.value), '{}'::jsonb)
        FROM steerable.current_config c WHERE c.run_id = r.id) AS config,
       (SELECT coalesce(jsonb_object_agg(pl.key, jsonb_build_object(
            'schema', pl.schema,
            'boundary', pl.boundary,
            'mode', pl.mode,
            'max_rel_change', pl.max_rel_change,
            'min_steps_between', pl.min_steps_between,
            'max_per_hour', pl.max_per_hour)), '{}'::jsonb)
        FROM steerable.policies pl WHERE pl.run_id = r.id) AS policies,
       (SELECT coalesce(jsonb_object_agg(s.name, s.points), '{}'::jsonb)
        FROM (
            SELECT t.name, jsonb_agg(jsonb_build_array(t.step, t.value) ORDER BY t.step) AS points
            FROM (
                SELECT m.name, m.step, m.value,
                       row_number() OVER (PARTITION BY m.name ORDER BY m.step DESC, m.at DESC) AS rn
                FROM steerable.metrics m WHERE m.run_id = r.id
            ) t
            WHERE t.rn <= 25
            GROUP BY t.name
        ) s) AS metrics,
       (SELECT coalesce(jsonb_agg(jsonb_build_object(
            'proposal_id', ps.proposal_id,
            'key', ps.key,
            'value', ps.value,
            'previous', ps.previous,
            'status', ps.status,
            'actor_kind', ps.actor_kind,
            'reason', ps.reason,
            'created_at', ps.created_at) ORDER BY ps.proposal_id), '[]'::jsonb)
        FROM steerable.proposal_status ps
        WHERE ps.run_id = r.id AND ps.status IN ('pending', 'approved')) AS open_proposals
FROM steerable.runs r;
