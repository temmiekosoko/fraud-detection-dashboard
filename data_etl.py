"""
Data ETL module for loading and preprocessing model data.

This module provides abstract and concrete implementations for loading
data from various database sources into pandas DataFrames.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd
import sqlite3
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataLoader(ABC):
    """Abstract base class for loading data from databases."""
    
    @abstractmethod
    def load(self, target: str, features: List[str]) -> pd.DataFrame:
        """
        Load data from database into a pandas DataFrame.
        
        Args:
            target: Name of the target variable column
            features: List of feature column names to load
            
        Returns:
            pd.DataFrame: DataFrame containing target and features
        """
        pass


class SQLiteDataLoader(DataLoader):
    """DataLoader implementation for SQLite databases."""
    
    def __init__(self, connection: sqlite3.Connection):
        """
        Initialize SQLite data loader.
        
        Args:
            connection: SQLite database connection object
        """
        self.connection = connection
        logger.info("SQLiteDataLoader initialized")
    
    def load(self, target: str, features: List[str], table_name: str = "model_data_new") -> pd.DataFrame:
        """
        Load data from SQLite database into a pandas DataFrame.
        
        Args:
            target: Name of the target variable column
            features: List of feature column names to load
            table_name: Name of the table to query (default: model_data_new)
            
        Returns:
            pd.DataFrame: DataFrame containing target and features
            
        Raises:
            ValueError: If target or features are missing from the table
        """
        # Build column list
        columns = [target] + features
        columns_str = ", ".join(columns)
        
        # Construct query
        query = f"SELECT {columns_str} FROM {table_name}"
        
        logger.info(f"Loading data from table '{table_name}'")
        logger.info(f"Columns: {columns}")
        
        # Load data
        df = pd.read_sql_query(query, self.connection)
        
        # Validate data types and missing values
        logger.info(f"Loaded {len(df)} rows")
        logger.info(f"Missing values:\n{df.isnull().sum()}")
        logger.info(f"Data types:\n{df.dtypes}")
        
        return df