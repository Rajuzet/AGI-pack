"""Continuous data harvesting and vectorization daemon package."""

from daemon.continuous_ingest import ContinuousIngestionDaemon, run_daemon

__all__ = ["ContinuousIngestionDaemon", "run_daemon"]
