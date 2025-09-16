"""
Async Database Manager for Multi-Agent Venture Studio

This module provides:
- Async SQLite database connection management
- Database initialization and migrations
- Connection pooling and transaction management
- Type-safe async operations with SQLAlchemy 2.0
- Health checks and monitoring

Design principles:
- Async-first for high performance
- Type safety with modern SQLAlchemy patterns
- Graceful error handling and retries
- Comprehensive logging and monitoring
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, AsyncIterator, Dict, Any, List
from datetime import datetime

from sqlalchemy import create_engine, text, event
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker
)
from sqlalchemy.pool import StaticPool
from sqlalchemy.exc import SQLAlchemyError

from .models import Base

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Configuration for database connections"""

    def __init__(self):
        # Database file location
        self.db_path = os.getenv('DATABASE_PATH', 'data/venture_studio.db')

        # Connection settings
        self.pool_size = int(os.getenv('DB_POOL_SIZE', '10'))
        self.max_overflow = int(os.getenv('DB_MAX_OVERFLOW', '20'))
        self.pool_timeout = int(os.getenv('DB_POOL_TIMEOUT', '30'))

        # Performance settings
        self.echo = os.getenv('DB_ECHO', 'false').lower() == 'true'
        self.echo_pool = os.getenv('DB_ECHO_POOL', 'false').lower() == 'true'

        # Backup settings
        self.auto_backup = os.getenv('DB_AUTO_BACKUP', 'true').lower() == 'true'
        self.backup_interval_hours = int(os.getenv('DB_BACKUP_INTERVAL', '24'))


class DatabaseManager:
    """
    Async database manager with connection pooling and health monitoring

    Features:
    - Async SQLite with WAL mode for better concurrency
    - Connection pooling for performance
    - Automatic migrations and schema updates
    - Health checks and connection monitoring
    - Backup and recovery utilities
    """

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker] = None
        self._initialized = False

        # Health monitoring
        self._connection_count = 0
        self._error_count = 0
        self._last_health_check: Optional[datetime] = None

    async def initialize(self) -> None:
        """Initialize database connection and create tables if needed"""
        if self._initialized:
            logger.warning("Database already initialized")
            return

        try:
            # Ensure data directory exists
            db_path = Path(self.config.db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)

            # Create async engine with optimized settings
            database_url = f"sqlite+aiosqlite:///{self.config.db_path}"

            self.engine = create_async_engine(
                database_url,
                echo=self.config.echo,
                echo_pool=self.config.echo_pool,
                poolclass=StaticPool,
                connect_args={
                    "check_same_thread": False,
                    "timeout": self.config.pool_timeout,
                },
                # SQLite-specific optimizations
                execution_options={
                    "isolation_level": "AUTOCOMMIT"
                }
            )

            # Configure SQLite for better performance and concurrency
            @event.listens_for(self.engine.sync_engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                """Configure SQLite for optimal performance"""
                cursor = dbapi_connection.cursor()

                # Enable WAL mode for better concurrency
                cursor.execute("PRAGMA journal_mode=WAL")

                # Performance optimizations
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA cache_size=10000")
                cursor.execute("PRAGMA temp_store=MEMORY")
                cursor.execute("PRAGMA mmap_size=268435456")  # 256MB

                # Foreign key support
                cursor.execute("PRAGMA foreign_keys=ON")

                cursor.close()

            # Create session factory
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )

            # Create tables if they don't exist
            await self._create_tables()

            # Verify connection
            await self._health_check()

            self._initialized = True
            logger.info(f"✅ Database initialized: {self.config.db_path}")

        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise

    async def _create_tables(self) -> None:
        """Create database tables if they don't exist"""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("📊 Database tables created/verified")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """
        Async context manager for database sessions

        Usage:
            async with db.session() as session:
                result = await session.execute(...)
                await session.commit()
        """
        if not self._initialized:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        session = self.session_factory()
        self._connection_count += 1

        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            self._error_count += 1
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()
            self._connection_count -= 1

    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a raw SQL query and return results"""
        async with self.session() as session:
            result = await session.execute(text(query), params or {})
            return [dict(row._mapping) for row in result.fetchall()]

    async def _health_check(self) -> Dict[str, Any]:
        """Perform database health check"""
        try:
            start_time = datetime.now()

            async with self.session() as session:
                # Simple query to test connection
                result = await session.execute(text("SELECT 1 as health_check"))
                health_result = result.scalar()

            response_time = (datetime.now() - start_time).total_seconds() * 1000

            health_status = {
                "status": "healthy" if health_result == 1 else "unhealthy",
                "response_time_ms": response_time,
                "connection_count": self._connection_count,
                "error_count": self._error_count,
                "last_check": datetime.now().isoformat(),
                "database_path": self.config.db_path
            }

            self._last_health_check = datetime.now()
            return health_status

        except Exception as e:
            self._error_count += 1
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }

    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            async with self.session() as session:
                # Get table counts
                stats = {}

                table_queries = {
                    "signals": "SELECT COUNT(*) as count FROM signals",
                    "theses": "SELECT COUNT(*) as count FROM theses",
                    "experiments": "SELECT COUNT(*) as count FROM experiments",
                    "decisions": "SELECT COUNT(*) as count FROM decisions"
                }

                for table, query in table_queries.items():
                    result = await session.execute(text(query))
                    stats[f"{table}_count"] = result.scalar()

                # Get recent activity
                recent_signals = await session.execute(text(
                    "SELECT COUNT(*) FROM signals WHERE discovered_at > datetime('now', '-24 hours')"
                ))
                stats["recent_signals_24h"] = recent_signals.scalar()

                # Add connection stats
                stats.update({
                    "connection_count": self._connection_count,
                    "error_count": self._error_count,
                    "last_health_check": self._last_health_check.isoformat() if self._last_health_check else None
                })

                return stats

        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {"error": str(e)}

    async def backup_database(self, backup_path: Optional[str] = None) -> str:
        """Create a backup of the database"""
        if not backup_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"data/backups/venture_studio_{timestamp}.db"

        # Ensure backup directory exists
        Path(backup_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            # Use SQLite backup API for consistent backup
            async with self.session() as session:
                await session.execute(text(f"VACUUM INTO '{backup_path}'"))

            logger.info(f"✅ Database backed up to: {backup_path}")
            return backup_path

        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up database connections"""
        if self.engine:
            await self.engine.dispose()
            logger.info("🧹 Database connections cleaned up")

        self._initialized = False

    async def reset_database(self) -> None:
        """WARNING: Completely reset the database (deletes all data)"""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
                await conn.run_sync(Base.metadata.create_all)

            logger.warning("⚠️  Database reset completed - all data deleted")

        except Exception as e:
            logger.error(f"Failed to reset database: {e}")
            raise


# Global database instance
db_manager = DatabaseManager()


# Convenience functions for common operations
async def init_database(config: Optional[DatabaseConfig] = None) -> DatabaseManager:
    """Initialize the global database manager"""
    global db_manager
    if config:
        db_manager = DatabaseManager(config)
    await db_manager.initialize()
    return db_manager


async def get_db_session():
    """Get a database session (for dependency injection)"""
    if not db_manager._initialized:
        await db_manager.initialize()
    return db_manager.session()


# Example usage and testing
async def main():
    """Example usage of DatabaseManager"""

    # Initialize database
    db = await init_database()

    try:
        # Health check
        health = await db._health_check()
        print(f"Database health: {health}")

        # Get stats
        stats = await db.get_stats()
        print(f"Database stats: {stats}")

        # Test raw query
        result = await db.execute_query("SELECT name FROM sqlite_master WHERE type='table'")
        print(f"Tables: {[row['name'] for row in result]}")

    finally:
        await db.cleanup()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())