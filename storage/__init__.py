"""Storage subsystem package initialization."""

from storage.gcs_manager import GCSManager, get_gcs_manager
from storage.mysql_manager import MySQLAuditManager, get_mysql_manager

# Backward-compatibility alias
MySQLManager = MySQLAuditManager

__all__ = ["GCSManager", "get_gcs_manager", "MySQLAuditManager", "get_mysql_manager", "MySQLManager"]
