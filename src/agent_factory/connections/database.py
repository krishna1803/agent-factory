"""
Database connectivity module for Oracle 23ai and PostgreSQL.

This module provides robust database connections and operations
for the Agent Factory, supporting Oracle 23ai vector search
and PostgreSQL with pgvector.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID

import oracledb
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from agent_factory.core.models import ConnectionSpec

logger = logging.getLogger(__name__)


class DatabaseConnection(ABC):
    """Abstract base class for database connections."""
    
    def __init__(self, connection_spec: ConnectionSpec):
        """Initialize database connection."""
        self.connection_spec = connection_spec
        self.connection_pool = None
        self.is_connected = False
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish database connection."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close database connection."""
        pass
    
    @abstractmethod
    async def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Execute a query and return results."""
        pass
    
    @abstractmethod
    async def execute_vector_search(self, query_vector: List[float], table: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Execute vector similarity search."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check database health."""
        pass


class Oracle23aiConnection(DatabaseConnection):
    """Oracle 23ai database connection with AI Vector Search support."""
    
    def __init__(self, connection_spec: ConnectionSpec):
        """Initialize Oracle 23ai connection."""
        super().__init__(connection_spec)
        self.pool = None
        self.engine = None
    
    async def connect(self) -> None:
        """Establish connection to Oracle 23ai."""
        try:
            config = self.connection_spec.configuration
            credentials = self.connection_spec.credentials
            
            # Create connection string
            dsn = f"{config.get('host', 'localhost')}:{config.get('port', 1521)}/{config.get('service_name', 'XEPDB1')}"
            
            # Configure Oracle client for vector operations
            oracledb.init_oracle_client()
            
            # Create connection pool
            self.pool = oracledb.create_pool(
                user=credentials.get('username'),
                password=credentials.get('password'),
                dsn=dsn,
                min=config.get('min_connections', 1),
                max=config.get('max_connections', 10),
                increment=config.get('connection_increment', 1)
            )
            
            # Create SQLAlchemy engine for advanced operations
            connection_string = f"oracle+oracledb://{credentials.get('username')}:{credentials.get('password')}@{dsn}"
            self.engine = create_async_engine(connection_string, echo=config.get('echo', False))
            
            self.is_connected = True
            logger.info("Successfully connected to Oracle 23ai")
            
        except Exception as e:
            logger.error(f"Failed to connect to Oracle 23ai: {str(e)}")
            raise
    
    async def disconnect(self) -> None:
        """Close Oracle connection."""
        try:
            if self.pool:
                self.pool.close()
            if self.engine:
                await self.engine.dispose()
            self.is_connected = False
            logger.info("Disconnected from Oracle 23ai")
        except Exception as e:
            logger.error(f"Error disconnecting from Oracle 23ai: {str(e)}")
    
    async def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Execute SQL query."""
        if not self.is_connected:
            await self.connect()
        
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(text(query), params or {})
                rows = result.fetchall()
                
                # Convert to list of dictionaries
                columns = result.keys()
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise
    
    async def execute_vector_search(self, query_vector: List[float], table: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Execute vector similarity search using Oracle AI Vector Search."""
        if not self.is_connected:
            await self.connect()
        
        try:
            # Convert vector to Oracle VECTOR format
            vector_str = f"VECTOR({query_vector})"
            
            # Oracle AI Vector Search query
            query = f"""
            SELECT id, content, metadata, 
                   VECTOR_DISTANCE(embedding, {vector_str}, COSINE) as distance
            FROM {table}
            ORDER BY VECTOR_DISTANCE(embedding, {vector_str}, COSINE)
            FETCH FIRST {limit} ROWS ONLY
            """
            
            return await self.execute_query(query)
            
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            raise
    
    async def create_vector_table(self, table_name: str, dimension: int = 1536) -> None:
        """Create a table with vector column for embeddings."""
        try:
            create_query = f"""
            CREATE TABLE {table_name} (
                id VARCHAR2(100) PRIMARY KEY,
                content CLOB,
                metadata JSON,
                embedding VECTOR({dimension}, FLOAT32)
            )
            """
            
            await self.execute_query(create_query)
            
            # Create vector index for performance
            index_query = f"""
            CREATE VECTOR INDEX {table_name}_vec_idx ON {table_name} (embedding)
            ORGANIZATION NEIGHBOR PARTITIONS
            WITH TARGET ACCURACY 95
            """
            
            await self.execute_query(index_query)
            logger.info(f"Created vector table and index: {table_name}")
            
        except Exception as e:
            logger.error(f"Failed to create vector table: {str(e)}")
            raise
    
    async def insert_vector_data(self, table: str, data: List[Dict[str, Any]]) -> None:
        """Insert data with vectors into Oracle table."""
        try:
            for item in data:
                query = f"""
                INSERT INTO {table} (id, content, metadata, embedding)
                VALUES (:id, :content, :metadata, VECTOR(:embedding))
                """
                
                params = {
                    'id': item['id'],
                    'content': item['content'],
                    'metadata': item.get('metadata', {}),
                    'embedding': item['embedding']
                }
                
                await self.execute_query(query, params)
            
            logger.info(f"Inserted {len(data)} records into {table}")
            
        except Exception as e:
            logger.error(f"Failed to insert vector data: {str(e)}")
            raise
    
    async def health_check(self) -> bool:
        """Check Oracle database health."""
        try:
            result = await self.execute_query("SELECT 1 FROM DUAL")
            return len(result) > 0
        except Exception:
            return False


class PostgreSQLConnection(DatabaseConnection):
    """PostgreSQL connection with pgvector support."""
    
    def __init__(self, connection_spec: ConnectionSpec):
        """Initialize PostgreSQL connection."""
        super().__init__(connection_spec)
        self.pool = None
        self.engine = None
    
    async def connect(self) -> None:
        """Establish connection to PostgreSQL."""
        try:
            config = self.connection_spec.configuration
            credentials = self.connection_spec.credentials
            
            # Create connection string
            connection_string = (
                f"postgresql+asyncpg://"
                f"{credentials.get('username')}:{credentials.get('password')}@"
                f"{config.get('host', 'localhost')}:{config.get('port', 5432)}/"
                f"{config.get('database', 'postgres')}"
            )
            
            # Create SQLAlchemy async engine
            self.engine = create_async_engine(
                connection_string,
                echo=config.get('echo', False),
                pool_size=config.get('pool_size', 10),
                max_overflow=config.get('max_overflow', 20)
            )
            
            # Test connection and enable pgvector
            async with self.engine.begin() as conn:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            
            self.is_connected = True
            logger.info("Successfully connected to PostgreSQL with pgvector")
            
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
            raise
    
    async def disconnect(self) -> None:
        """Close PostgreSQL connection."""
        try:
            if self.engine:
                await self.engine.dispose()
            self.is_connected = False
            logger.info("Disconnected from PostgreSQL")
        except Exception as e:
            logger.error(f"Error disconnecting from PostgreSQL: {str(e)}")
    
    async def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Execute SQL query."""
        if not self.is_connected:
            await self.connect()
        
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(text(query), params or {})
                rows = result.fetchall()
                
                # Convert to list of dictionaries
                columns = result.keys()
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise
    
    async def execute_vector_search(self, query_vector: List[float], table: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Execute vector similarity search using pgvector."""
        if not self.is_connected:
            await self.connect()
        
        try:
            # Convert vector to pgvector format
            vector_str = f"[{','.join(map(str, query_vector))}]"
            
            # pgvector similarity search query
            query = f"""
            SELECT id, content, metadata, 
                   embedding <-> '{vector_str}'::vector as distance
            FROM {table}
            ORDER BY embedding <-> '{vector_str}'::vector
            LIMIT {limit}
            """
            
            return await self.execute_query(query)
            
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            raise
    
    async def create_vector_table(self, table_name: str, dimension: int = 1536) -> None:
        """Create a table with vector column for embeddings."""
        try:
            create_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id VARCHAR(100) PRIMARY KEY,
                content TEXT,
                metadata JSONB,
                embedding vector({dimension})
            )
            """
            
            await self.execute_query(create_query)
            
            # Create vector index for performance
            index_query = f"""
            CREATE INDEX IF NOT EXISTS {table_name}_vec_idx 
            ON {table_name} USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
            """
            
            await self.execute_query(index_query)
            logger.info(f"Created vector table and index: {table_name}")
            
        except Exception as e:
            logger.error(f"Failed to create vector table: {str(e)}")
            raise
    
    async def insert_vector_data(self, table: str, data: List[Dict[str, Any]]) -> None:
        """Insert data with vectors into PostgreSQL table."""
        try:
            for item in data:
                vector_str = f"[{','.join(map(str, item['embedding']))}]"
                
                query = f"""
                INSERT INTO {table} (id, content, metadata, embedding)
                VALUES (:id, :content, :metadata, :embedding::vector)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """
                
                params = {
                    'id': item['id'],
                    'content': item['content'],
                    'metadata': item.get('metadata', {}),
                    'embedding': vector_str
                }
                
                await self.execute_query(query, params)
            
            logger.info(f"Inserted {len(data)} records into {table}")
            
        except Exception as e:
            logger.error(f"Failed to insert vector data: {str(e)}")
            raise
    
    async def health_check(self) -> bool:
        """Check PostgreSQL database health."""
        try:
            result = await self.execute_query("SELECT 1")
            return len(result) > 0
        except Exception:
            return False


class DatabaseManager:
    """Manager for database connections and operations."""
    
    def __init__(self):
        """Initialize database manager."""
        self.connections: Dict[str, DatabaseConnection] = {}
    
    def register_connection(self, name: str, connection_spec: ConnectionSpec) -> None:
        """Register a database connection."""
        if connection_spec.connection_type.value == "database":
            config = connection_spec.configuration
            db_type = config.get('database_type', '').lower()
            
            if db_type == 'oracle' or 'oracle' in connection_spec.endpoint.lower():
                self.connections[name] = Oracle23aiConnection(connection_spec)
            elif db_type == 'postgresql' or 'postgres' in connection_spec.endpoint.lower():
                self.connections[name] = PostgreSQLConnection(connection_spec)
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
        else:
            raise ValueError("Connection spec must be of type 'database'")
    
    async def get_connection(self, name: str) -> DatabaseConnection:
        """Get a database connection by name."""
        if name not in self.connections:
            raise ValueError(f"Database connection '{name}' not found")
        
        connection = self.connections[name]
        if not connection.is_connected:
            await connection.connect()
        
        return connection
    
    async def execute_query(self, connection_name: str, query: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Execute query on specified connection."""
        connection = await self.get_connection(connection_name)
        return await connection.execute_query(query, params)
    
    async def execute_vector_search(
        self, 
        connection_name: str, 
        query_vector: List[float], 
        table: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Execute vector search on specified connection."""
        connection = await self.get_connection(connection_name)
        return await connection.execute_vector_search(query_vector, table, limit)
    
    async def close_all_connections(self) -> None:
        """Close all database connections."""
        for connection in self.connections.values():
            if connection.is_connected:
                await connection.disconnect()
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all connections."""
        health_status = {}
        for name, connection in self.connections.items():
            try:
                health_status[name] = await connection.health_check()
            except Exception as e:
                logger.error(f"Health check failed for {name}: {str(e)}")
                health_status[name] = False
        
        return health_status


# Global database manager instance
db_manager = DatabaseManager()
