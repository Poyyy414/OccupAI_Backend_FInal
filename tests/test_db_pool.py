import psycopg2.pool
import pytest

from backend import db


class FakeCursor:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.executed = None
        self.closed = False
        self.rowcount = 1

    def execute(self, sql, params):
        self.executed = (sql, params)

    def fetchall(self):
        return self.rows

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self, rows=None):
        self.cursor_instance = FakeCursor(rows)
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def test_borrow_waits_for_a_connection_during_a_short_burst(monkeypatch):
    connection = object()

    class BusyPool:
        attempts = 0

        def getconn(self):
            self.attempts += 1
            if self.attempts < 3:
                raise psycopg2.pool.PoolError("connection pool exhausted")
            return connection

    pool = BusyPool()
    clock = iter((0.0, 0.1, 0.2, 0.3))
    monkeypatch.setattr(db, "_get_pool", lambda: pool)
    monkeypatch.setattr(db.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(db.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(db, "DB_POOL_WAIT_SECONDS", 1.0)

    assert db._borrow() is connection
    assert pool.attempts == 3


def test_query_ends_read_transaction_before_releasing_connection(monkeypatch):
    rows = [{"value": 1}]
    connection = FakeConnection(rows)
    released = []
    monkeypatch.setattr(db, "_borrow", lambda: connection)
    monkeypatch.setattr(db, "_release", released.append)

    assert db.query("SELECT %s AS value", (1,)) == rows
    assert connection.cursor_instance.executed == ("SELECT %s AS value", (1,))
    assert connection.rollbacks == 1
    assert connection.cursor_instance.closed is True
    assert released == [connection]


def test_execute_closes_cursor_and_releases_connection(monkeypatch):
    connection = FakeConnection()
    released = []
    monkeypatch.setattr(db, "_borrow", lambda: connection)
    monkeypatch.setattr(db, "_release", released.append)

    affected = db.execute("UPDATE users SET is_active = %s", (True,))

    assert affected == 1
    assert connection.commits == 1
    assert connection.cursor_instance.closed is True
    assert released == [connection]


def test_transaction_handle_returns_connection_to_pool_once(monkeypatch):
    connection = FakeConnection()
    released = []
    monkeypatch.setattr(db, "_borrow", lambda: connection)
    monkeypatch.setattr(db, "_release", released.append)

    handle = db.get_db()
    assert handle.cursor() is connection.cursor_instance
    handle.close()
    handle.close()

    assert released == [connection]


def test_query_rolls_back_and_releases_connection_after_database_error(monkeypatch):
    connection = FakeConnection()
    released = []

    def fail_execute(_sql, _params):
        raise psycopg2.DatabaseError("synthetic failure")

    connection.cursor_instance.execute = fail_execute
    monkeypatch.setattr(db, "_borrow", lambda: connection)
    monkeypatch.setattr(db, "_release", released.append)

    with pytest.raises(Exception) as error:
        db.query("SELECT 1")

    assert getattr(error.value, "status_code", None) == 500
    assert connection.rollbacks == 1
    assert connection.cursor_instance.closed is True
    assert released == [connection]
