"""Admin-side database setup: roles, schema, grants. Needs a superuser (or CREATEROLE) DSN."""

from __future__ import annotations

from importlib import resources

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

ADMIN = "steerable_admin"
TRAINER = "steerable_trainer"
AGENT = "steerable_agent"
ROLES = (ADMIN, TRAINER, AGENT)

_SQL_FILES = ("010_schema.sql", "020_views.sql", "030_functions.sql", "040_grants.sql")


def role_dsn(base_dsn: str, role: str, password: str) -> str:
    """The same server and database as ``base_dsn``, logged in as ``role``."""
    params = conninfo_to_dict(base_dsn)
    params["user"] = role
    params["password"] = password
    return make_conninfo("", **params)


def schema_exists(conn: psycopg.Connection[tuple[object, ...]]) -> bool:
    row = conn.execute("SELECT 1 FROM pg_namespace WHERE nspname = 'steerable'").fetchone()
    return row is not None


def init(
    superuser_dsn: str,
    *,
    admin_password: str,
    trainer_password: str,
    agent_password: str,
    reset: bool = False,
) -> bool:
    """Create the three roles (idempotent) and the schema (once). Returns True if the schema was created."""
    passwords = {ADMIN: admin_password, TRAINER: trainer_password, AGENT: agent_password}
    with psycopg.connect(superuser_dsn, autocommit=True) as conn:
        dbname = conn.info.dbname
        for role, password in passwords.items():
            conn.execute(
                sql.SQL(
                    "DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = {name}) "
                    "THEN CREATE ROLE {role} NOLOGIN; END IF; END $$"
                ).format(name=sql.Literal(role), role=sql.Identifier(role))
            )
            conn.execute(
                sql.SQL(
                    "ALTER ROLE {role} WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT PASSWORD {pw}"
                ).format(role=sql.Identifier(role), pw=sql.Literal(password))
            )
            conn.execute(
                sql.SQL("GRANT CONNECT ON DATABASE {db} TO {role}").format(
                    db=sql.Identifier(dbname), role=sql.Identifier(role)
                )
            )
        conn.execute(sql.SQL("ALTER ROLE {role} CONNECTION LIMIT 8").format(role=sql.Identifier(AGENT)))
        conn.execute(
            sql.SQL("GRANT CREATE ON DATABASE {db} TO {role}").format(
                db=sql.Identifier(dbname), role=sql.Identifier(ADMIN)
            )
        )

        if reset:
            conn.execute("DROP SCHEMA IF EXISTS steerable CASCADE")
        if schema_exists(conn):
            return False

        conn.execute(sql.SQL("SET ROLE {role}").format(role=sql.Identifier(ADMIN)))
        try:
            for name in _SQL_FILES:
                conn.execute(resources.files("steerable.sql").joinpath(name).read_text())
        finally:
            conn.execute("RESET ROLE")
    return True
