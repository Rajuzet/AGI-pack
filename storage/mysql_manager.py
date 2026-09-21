"""Production MySQL database persistence and relational audit logging layer.

Implements MySQLAuditManager with connection pooling, auto-reconnect,
non-blocking offline fallback, and structured telemetry recording for:
- Research papers and technical reports
- QLoRA fine-tuning training runs
- Autonomous ReAct agent sessions and granular step traces
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MySQLAuditManager:
    """Production relational database audit manager with connection pooling and graceful offline fallback."""

    _instance: Optional["MySQLAuditManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(
        cls,
        *args: Any,
        force_new: bool = False,
        **kwargs: Any,
    ) -> "MySQLAuditManager":
        """Instantiate or retrieve the singleton MySQLAuditManager instance."""
        if force_new:
            return super().__new__(cls)

        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset the singleton instance (primarily for testing and reconfiguration)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._pool = None
                cls._instance._initialized = False
                if hasattr(cls._instance, "_buffer_lock"):
                    with cls._instance._buffer_lock:
                        cls._instance._buffered_papers.clear()
                        cls._instance._buffered_sessions.clear()
                        cls._instance._buffered_training_runs.clear()
            cls._instance = None

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        pool_size: int = 5,
        pool_name: str = "agi_audit_pool",
        settings: Optional[Any] = None,
        force_new: bool = False,
    ) -> None:
        """Initialize connection pool parameters from arguments, settings, or environment."""
        if getattr(self, "_initialized", False) and not force_new:
            return

        with self._lock:
            if getattr(self, "_initialized", False) and not force_new:
                return

            if settings is not None:
                self.host = host or getattr(settings, "mysql_host", "localhost")
                self.port = int(port or getattr(settings, "mysql_port", 3306))
                self.user = user or getattr(settings, "mysql_user", "root")
                self.password = password or getattr(settings, "mysql_password", "")
                self.database = database or getattr(settings, "mysql_database", "agi_memory")
                self.pool_size = int(pool_size or getattr(settings, "mysql_pool_size", 5))
            else:
                self.host = host or os.environ.get("MYSQL_HOST", "localhost")
                self.port = int(port or os.environ.get("MYSQL_PORT", 3306))
                self.user = user or os.environ.get("MYSQL_USER", "root")
                self.password = password or os.environ.get("MYSQL_PASSWORD", "")
                self.database = database or os.environ.get("MYSQL_DATABASE", "agi_memory")
                self.pool_size = int(pool_size or os.environ.get("MYSQL_POOL_SIZE", 5))

            self.pool_name = pool_name
            self._pool = None
            self._is_available: Optional[bool] = None
            self._last_error: Optional[str] = None
            self._last_attempt_time: float = 0.0
            self._reconnect_cooldown_seconds: float = 5.0
            self._buffer_lock = threading.RLock()
            self._buffered_papers: List[Dict[str, Any]] = []
            self._buffered_sessions: List[Tuple[Dict[str, Any], Optional[List[Dict[str, Any]]]]] = []
            self._buffered_training_runs: List[Dict[str, Any]] = []
            self._initialized = True

    def _initialize_pool(self) -> bool:
        """Initialize the MySQL connection pool lazily with reconnect cooldown."""
        if self._pool is not None:
            return True

        # Non-blocking circuit-breaker: do not retry within cooldown window if previously unavailable
        now = time.time()
        if self._is_available is False and (now - self._last_attempt_time) < self._reconnect_cooldown_seconds:
            return False

        self._last_attempt_time = now
        try:
            import mysql.connector
            from mysql.connector import pooling

            pool_config = {
                "pool_name": self.pool_name,
                "pool_size": self.pool_size,
                "pool_reset_session": True,
                "host": self.host,
                "port": self.port,
                "user": self.user,
                "password": self.password,
                "database": self.database,
                "charset": "utf8mb4",
                "collation": "utf8mb4_unicode_ci",
                "connection_timeout": 3,  # Short timeout prevents blocking agent
            }
            self._pool = pooling.MySQLConnectionPool(**pool_config)
            self._is_available = True
            self._last_error = None
            logger.info("MySQL connection pool '%s' initialized for '%s@%s:%d'", self.pool_name, self.user, self.host, self.port)
            return True
        except Exception as exc:
            self._is_available = False
            self._last_error = str(exc)
            logger.warning(
                "MySQL database is unreachable at %s:%d (database: '%s'). Operating in non-blocking offline mode: %s",
                self.host,
                self.port,
                self.database,
                exc,
            )
            return False

    def get_connection(self):
        """Borrow an active connection from the pool with auto-reconnect validation."""
        if not self._initialize_pool() or self._pool is None:
            return None

        try:
            conn = self._pool.get_connection()
            if not conn.is_connected():
                conn.reconnect(attempts=2, delay=1)
            return conn
        except Exception as exc:
            self._is_available = False
            self._last_error = str(exc)
            logger.warning("Failed to borrow active connection from MySQL pool: %s", exc)
            return None

    def health_check(self) -> Dict[str, Any]:
        """Perform non-blocking connection diagnostics and latency measurement.

        Returns:
            Dictionary containing connectivity status, latency in milliseconds,
            target endpoint, and error details if offline.
        """
        start_time = time.perf_counter()
        conn = self.get_connection()
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        if not conn:
            return {
                "status": "unreachable",
                "healthy": False,
                "host": self.host,
                "port": self.port,
                "database": self.database,
                "latency_ms": latency_ms,
                "error": self._last_error or "Connection pool failed to establish socket connection.",
            }

        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1;")
            cursor.fetchone()
            cursor.close()
            conn.close()  # Return connection to pool

            return {
                "status": "healthy",
                "healthy": True,
                "host": self.host,
                "port": self.port,
                "database": self.database,
                "latency_ms": latency_ms,
                "pool_size": self.pool_size,
                "error": None,
            }
        except Exception as exc:
            try:
                conn.close()
            except Exception:
                pass
            return {
                "status": "degraded",
                "healthy": False,
                "host": self.host,
                "port": self.port,
                "database": self.database,
                "latency_ms": latency_ms,
                "error": str(exc),
            }

    def init_schema(self, schema_path: Optional[Path] = None) -> bool:
        """Execute DDL schema to provision database tables if online."""
        conn = self.get_connection()
        if not conn:
            return False

        path = schema_path or Path(__file__).parent / "schema.sql"
        if not path.exists():
            logger.warning("Schema file '%s' not found.", path)
            conn.close()
            return False

        try:
            with path.open("r", encoding="utf-8") as f:
                sql_script = f.read()

            cursor = conn.cursor()
            for statement in sql_script.split(";"):
                stmt = statement.strip()
                if stmt:
                    cursor.execute(stmt)
            conn.commit()
            cursor.close()
            conn.close()
            logger.info("Successfully provisioned MySQL audit schema from %s", path)
            return True
        except Exception as exc:
            logger.error("Failed to execute schema provisioning: %s", exc)
            try:
                conn.close()
            except Exception:
                pass
            return False

    def get_existing_paper_ids(self, paper_ids: List[str]) -> set:
        """Query MySQL for paper IDs that already exist in research_papers.

        Args:
            paper_ids: List of paper or article IDs to check.

        Returns:
            Set of IDs that already exist in the database (empty set if offline or none exist).
        """
        if not paper_ids:
            return set()

        conn = self.get_connection()
        if not conn:
            return set()

        try:
            cursor = conn.cursor()
            format_strings = ",".join(["%s"] * len(paper_ids))
            query = f"SELECT id FROM research_papers WHERE id IN ({format_strings})"
            cursor.execute(query, tuple(paper_ids))
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            return {row[0] for row in rows}
        except Exception as exc:
            logger.warning("Error checking existing paper IDs in MySQL: %s", exc)
            try:
                conn.close()
            except Exception:
                pass
            return set()

    def paper_exists(self, paper_id: str) -> bool:
        """Check if a specific paper ID is already persisted in MySQL.

        Args:
            paper_id: Unique paper or article ID.

        Returns:
            True if exists, False if not or if database is offline.
        """
        return paper_id in self.get_existing_paper_ids([paper_id])

    @property
    def buffered_count(self) -> int:
        """Return the count of all unflushed buffered audit items."""
        with self._buffer_lock:
            return len(self._buffered_papers) + len(self._buffered_sessions) + len(self._buffered_training_runs)

    def _buffer_paper(self, paper_data: Dict[str, Any]) -> None:
        with self._buffer_lock:
            self._buffered_papers.append(paper_data)
            logger.info("Buffered paper audit record in memory (unflushed: %d)", self.buffered_count)

    def _buffer_training_run(self, run_summary: Dict[str, Any]) -> None:
        with self._buffer_lock:
            self._buffered_training_runs.append(run_summary)
            logger.info("Buffered training run audit record in memory (unflushed: %d)", self.buffered_count)

    def _buffer_agent_session(
        self,
        session_data: Dict[str, Any],
        steps: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        with self._buffer_lock:
            self._buffered_sessions.append((session_data, steps))
            logger.info("Buffered agent session audit record in memory (unflushed: %d)", self.buffered_count)

    def flush_buffer(self) -> Dict[str, int]:
        """Attempt to flush in-memory buffered records to MySQL upon reconnect.

        Returns:
            Dictionary indicating count of flushed records per category.
        """
        with self._buffer_lock:
            if not self._buffered_papers and not self._buffered_sessions and not self._buffered_training_runs:
                return {"flushed_papers": 0, "flushed_sessions": 0, "flushed_training_runs": 0}

        conn = self.get_connection()
        if not conn:
            return {"flushed_papers": 0, "flushed_sessions": 0, "flushed_training_runs": 0}

        flushed = {"flushed_papers": 0, "flushed_sessions": 0, "flushed_training_runs": 0}
        with self._buffer_lock:
            # 1. Flush papers
            remaining_papers = []
            for p in self._buffered_papers:
                if self._execute_record_paper(conn, p):
                    flushed["flushed_papers"] += 1
                else:
                    remaining_papers.append(p)
            self._buffered_papers = remaining_papers

            # 2. Flush training runs
            remaining_runs = []
            for r in self._buffered_training_runs:
                if self._execute_record_training_run(conn, r):
                    flushed["flushed_training_runs"] += 1
                else:
                    remaining_runs.append(r)
            self._buffered_training_runs = remaining_runs

            # 3. Flush agent sessions
            remaining_sessions = []
            for s_data, st_list in self._buffered_sessions:
                if self._execute_record_agent_session(conn, s_data, st_list):
                    flushed["flushed_sessions"] += 1
                else:
                    remaining_sessions.append((s_data, st_list))
            self._buffered_sessions = remaining_sessions

        try:
            conn.close()
        except Exception:
            pass

        logger.info("Flushed buffered audit telemetry to MySQL: %s (remaining: %d)", flushed, self.buffered_count)
        return flushed

    def _execute_record_paper(self, conn: Any, paper_data: Dict[str, Any]) -> bool:
        """Execute paper insertion SQL on an active connection."""
        paper_id = str(paper_data.get("id") or paper_data.get("paper_id") or "")
        title = str(paper_data.get("title") or "Untitled")[:512]
        abstract = str(paper_data.get("content") or paper_data.get("abstract") or paper_data.get("text") or "")
        source = str(paper_data.get("source") or "unknown")[:64]

        meta = paper_data.get("metadata") or {}
        authors_val = meta.get("authors") or paper_data.get("authors") or []
        categories_val = meta.get("categories") or paper_data.get("categories") or []
        pdf_url = meta.get("pdf_url") or meta.get("link") or paper_data.get("pdf_url") or paper_data.get("link")

        authors_json = json.dumps(authors_val if isinstance(authors_val, list) else [authors_val])
        categories_json = json.dumps(categories_val if isinstance(categories_val, list) else [categories_val])

        sql = """
        INSERT INTO research_papers (
            id, title, abstract, source, authors, categories, pdf_url, ingested_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
        ON DUPLICATE KEY UPDATE
            title = VALUES(title),
            abstract = VALUES(abstract),
            authors = VALUES(authors),
            categories = VALUES(categories),
            pdf_url = VALUES(pdf_url);
        """
        params = (paper_id, title, abstract, source, authors_json, categories_json, pdf_url)

        try:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            conn.commit()
            cursor.close()
            logger.debug("Audited paper '%s' in MySQL.", paper_id)
            return True
        except Exception as exc:
            logger.warning("Failed to audit paper '%s' in MySQL: %s", paper_id, exc)
            return False

    def record_paper(self, paper_data: Dict[str, Any]) -> bool:
        """Audit log an ingested research paper or news article.

        If MySQL is offline or disconnected, buffers in memory and retries upon reconnect.
        """
        conn = self.get_connection()
        if not conn:
            self._buffer_paper(paper_data)
            return False

        success = self._execute_record_paper(conn, paper_data)
        try:
            conn.close()
        except Exception:
            pass

        if not success:
            self._buffer_paper(paper_data)
        elif self.buffered_count > 0:
            self.flush_buffer()

        return success

    def _execute_record_training_run(self, conn: Any, run_summary: Dict[str, Any]) -> bool:
        """Execute training run insertion SQL on an active connection."""
        run_id = str(run_summary.get("run_id") or f"run_{int(time.time())}")[:128]
        base_model = str(run_summary.get("base_model") or "unknown")[:255]
        quantization = str(run_summary.get("quantization") or "4-bit NF4")[:64]
        samples = int(run_summary.get("dataset_sample_count") or run_summary.get("training_samples") or 0)
        loss = float(run_summary.get("training_loss") or 0.0)
        duration = float(run_summary.get("duration_seconds") or 0.0)

        gcs_sync = run_summary.get("gcs_sync") or {}
        gcs_uri = (
            gcs_sync.get("target_uri")
            or gcs_sync.get("simulated_uri")
            or run_summary.get("gcs_checkpoint_uri")
            or "N/A"
        )[:1024]

        sql = """
        INSERT INTO training_runs (
            run_id, base_model, quantization, dataset_sample_count,
            training_loss, duration_seconds, gcs_checkpoint_uri, executed_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
        ON DUPLICATE KEY UPDATE
            training_loss = VALUES(training_loss),
            duration_seconds = VALUES(duration_seconds),
            gcs_checkpoint_uri = VALUES(gcs_checkpoint_uri);
        """
        params = (run_id, base_model, quantization, samples, loss, duration, gcs_uri)

        try:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            conn.commit()
            cursor.close()
            logger.info("Audited training run '%s' in MySQL.", run_id)
            return True
        except Exception as exc:
            logger.warning("Failed to audit training run '%s' in MySQL: %s", run_id, exc)
            return False

    def record_training_run(self, run_summary: Dict[str, Any]) -> bool:
        """Audit log a QLoRA fine-tuning execution run and checkpoint URI.

        If MySQL is offline or disconnected, buffers in memory and retries upon reconnect.
        """
        conn = self.get_connection()
        if not conn:
            self._buffer_training_run(run_summary)
            return False

        success = self._execute_record_training_run(conn, run_summary)
        try:
            conn.close()
        except Exception:
            pass

        if not success:
            self._buffer_training_run(run_summary)
        elif self.buffered_count > 0:
            self.flush_buffer()

        return success

    def _execute_record_agent_session(
        self,
        conn: Any,
        session_data: Dict[str, Any],
        steps: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Execute agent session and granular steps SQL on an active connection."""
        session_id = str(
            session_data.get("session_id")
            or f"sess_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 10000}"
        )[:128]

        input_objective = str(session_data.get("input_objective") or session_data.get("query") or "")
        is_success = bool(session_data.get("success", True))
        status = "success" if is_success else "failed"
        total_steps = int(session_data.get("total_steps") or 0)
        retry_count = int(session_data.get("retry_count") or session_data.get("retries_used") or 0)
        latency = float(session_data.get("execution_latency") or session_data.get("execution_time_seconds") or 0.0)

        step_list = steps or session_data.get("steps") or []

        session_sql = """
        INSERT INTO agent_sessions (
            session_id, input_objective, status, total_steps,
            retry_count, execution_latency, session_timestamp
        ) VALUES (%s, %s, %s, %s, %s, %s, NOW());
        """
        session_params = (session_id, input_objective, status, total_steps, retry_count, latency)

        step_sql = """
        INSERT INTO agent_steps (
            session_id, step_index, thought_trace, action_name,
            action_input, observation_output, step_latency, created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW());
        """

        try:
            cursor = conn.cursor()
            cursor.execute(session_sql, session_params)

            for step in step_list:
                step_idx = int(step.get("step_index") or step.get("step_number") or 1)
                thought = str(step.get("thought_trace") or step.get("thought") or "")
                action = step.get("action_name") or step.get("action")
                action_str = str(action)[:128] if action else None
                act_input = json.dumps(step.get("action_input") or {})
                observation = step.get("observation_output") or step.get("observation")
                obs_str = str(observation) if observation is not None else None
                step_lat = float(step.get("step_latency") or step.get("duration_seconds") or 0.0)

                cursor.execute(step_sql, (session_id, step_idx, thought, action_str, act_input, obs_str, step_lat))

            conn.commit()
            cursor.close()
            logger.info("Audited agent session '%s' with %d steps in MySQL.", session_id, len(step_list))
            return True
        except Exception as exc:
            logger.warning("Failed to audit agent session '%s' in MySQL: %s", session_id, exc)
            return False

    def record_agent_session(
        self,
        session_data: Dict[str, Any],
        steps: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Audit log an autonomous ReAct reasoning session and all granular steps.

        If MySQL is offline or disconnected, buffers in memory and retries upon reconnect.
        """
        step_list = steps or session_data.get("steps") or []
        conn = self.get_connection()
        if not conn:
            self._buffer_agent_session(session_data, step_list)
            return False

        success = self._execute_record_agent_session(conn, session_data, step_list)
        try:
            conn.close()
        except Exception:
            pass

        if not success:
            self._buffer_agent_session(session_data, step_list)
        elif self.buffered_count > 0:
            self.flush_buffer()

        return success


def get_mysql_manager(settings: Optional[Any] = None) -> MySQLAuditManager:
    """Obtain or initialize the singleton MySQLAuditManager instance."""
    return MySQLAuditManager(settings=settings)
