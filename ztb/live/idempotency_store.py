"""
Durable idempotency store using SQLite.

Ensures client_order_id uniqueness across sessions with persistent storage.
"""

import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional


class IdempotencyStore:
    """SQLite-based idempotency store for order IDs."""


class IdempotencyStore:
    """SQLite-based idempotency store for order IDs."""

    def __init__(
        self, db_path: str = ":memory:", table_name: str = "order_idempotency"
    ):
        """
        Initialize idempotency store.

        Args:
            db_path: SQLite database path (:memory: for in-memory)
            table_name: Table name for idempotency records
        """
        self.db_path = db_path
        self.table_name = table_name
        self._local = threading.local()
        self._lock_file = None

        # Set up process-level locking for file-based databases
        if db_path != ":memory:":
            db_file = Path(db_path)
            self._lock_file = db_file.with_suffix(".lock")
            self._lock_fd = None

        # Create table if it doesn't exist
        self._ensure_table()

    @contextmanager
    def _process_lock(self):
        """Process-level file locking for cross-process synchronization."""
        if self._lock_file is None:
            # In-memory database, no process locking needed
            yield
            return

        lock_acquired = False
        try:
            # Simple file-based locking (works on Windows)
            while not lock_acquired:
                try:
                    with open(self._lock_file, "w") as f:
                        f.write(str(os.getpid()))
                    lock_acquired = True
                except (OSError, IOError):
                    # Lock file exists, wait and retry
                    time.sleep(0.01)

            yield

        finally:
            if lock_acquired and self._lock_file.exists():
                try:
                    self._lock_file.unlink()
                except OSError:
                    pass  # Ignore cleanup errors

    @contextmanager
    def _get_connection(self):
        """Get thread-local database connection with process-level locking."""
        with self._process_lock():
            if not hasattr(self._local, "connection"):
                # Each thread gets its own connection; do not share connections between threads.
                self._local.connection = sqlite3.connect(
                    self.db_path, check_same_thread=True
                )
                # Enable WAL mode for better concurrency
                self._local.connection.execute("PRAGMA journal_mode=WAL")
                self._local.connection.execute("PRAGMA synchronous=NORMAL")
                self._local.connection.execute("PRAGMA wal_autocheckpoint=1000")

            try:
                yield self._local.connection
            except Exception:
                # Reset connection on error
                if hasattr(self._local, "connection"):
                    self._local.connection.close()
                    delattr(self._local, "connection")
                raise

    def _ensure_table(self):
        """Ensure the idempotency table exists."""
        with self._get_connection() as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    client_order_id TEXT PRIMARY KEY,
                    order_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'pending'
                )
            """)
            # Create index for faster lookups
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_client_order_id
                ON {self.table_name}(client_order_id)
            """)
            conn.commit()

    def check_and_store(self, client_order_id: str, order_data: Dict[str, Any]) -> bool:
        """
        Check if order_id exists, store if not.

        Args:
            client_order_id: Client-provided order ID
            order_data: Order data dict

        Returns:
            True if stored (new order), False if already exists
        """
        import json

        order_json = json.dumps(order_data, sort_keys=True, default=str)

        with self._get_connection() as conn:
            try:
                # Try to insert
                conn.execute(
                    f"""
                    INSERT INTO {self.table_name} (client_order_id, order_data, status)
                    VALUES (?, ?, 'pending')
                """,
                    (client_order_id, order_json),
                )

                conn.commit()
                return True  # Successfully stored

            except sqlite3.IntegrityError:
                # Order ID already exists
                return False

    def get_order_data(self, client_order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get stored order data for client_order_id.

        Args:
            client_order_id: Client order ID

        Returns:
            Order data dict or None if not found
        """
        import json

        with self._get_connection() as conn:
            cursor = conn.execute(
                f"""
                SELECT order_data FROM {self.table_name}
                WHERE client_order_id = ?
            """,
                (client_order_id,),
            )

            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
            return None

    def update_status(self, client_order_id: str, status: str) -> bool:
        """
        Update order status.

        Args:
            client_order_id: Client order ID
            status: New status ('pending', 'filled', 'cancelled', 'rejected')

        Returns:
            True if updated, False if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                f"""
                UPDATE {self.table_name}
                SET status = ?
                WHERE client_order_id = ?
            """,
                (status, client_order_id),
            )

            conn.commit()
            return cursor.rowcount > 0

    def list_orders(
        self, status: Optional[str] = None, limit: int = 100
    ) -> list[Dict[str, Any]]:
        """
        List stored orders.

        Args:
            status: Filter by status (optional)
            limit: Maximum number of records

        Returns:
            List of order records
        """
        import json

        with self._get_connection() as conn:
            if status:
                cursor = conn.execute(
                    f"""
                    SELECT client_order_id, order_data, status, created_at
                    FROM {self.table_name}
                    WHERE status = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """,
                    (status, limit),
                )
            else:
                cursor = conn.execute(
                    f"""
                    SELECT client_order_id, order_data, status, created_at
                    FROM {self.table_name}
                    ORDER BY created_at DESC
                    LIMIT ?
                """,
                    (limit,),
                )

            results = []
            for row in cursor.fetchall():
                order_data = json.loads(row[1])
                results.append(
                    {
                        "client_order_id": row[0],
                        "order_data": order_data,
                        "status": row[2],
                        "created_at": row[3],
                    }
                )

            return results

    def cleanup_old_orders(self, days_old: int = 30) -> int:
        """
        Clean up old orders.

        Args:
            days_old: Remove orders older than this many days

        Returns:
            Number of records deleted
        """
        with self._get_connection() as conn:
            query = f"""
                DELETE FROM {self.table_name}
                WHERE created_at < datetime('now', ?)
            """
            param = f"-{int(days_old)} days"
            cursor = conn.execute(query, (param,))

            conn.commit()
            return cursor.rowcount

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        with self._get_connection() as conn:
            # Total orders
            cursor = conn.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            total_orders = cursor.fetchone()[0]

            # Status breakdown
            cursor = conn.execute(f"""
                SELECT status, COUNT(*) FROM {self.table_name}
                GROUP BY status
            """)
            status_counts = dict(cursor.fetchall())

            return {"total_orders": total_orders, "status_breakdown": status_counts}


import threading

# Global instance and lock for thread safety
_idempotency_store = IdempotencyStore()
_idempotency_store_lock = threading.Lock()


def get_idempotency_store() -> IdempotencyStore:
    """Get global idempotency store instance."""
    return _idempotency_store


def set_global_store_path(db_path: str):
    """Set global store database path in a thread-safe manner."""
    global _idempotency_store
    with _idempotency_store_lock:
        _idempotency_store = IdempotencyStore(db_path)
