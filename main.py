"""
Main entry point for project

This script orchestrates the entire workflow including:
- Feature engineering
- Data loading
- Model training and evaluation
"""

import argparse
import sqlite3
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Appriss Return Fraud Detection"
    )
    parser.add_argument(
        "--db-type",
        type=str,
        choices=["sqlite", "postgresql"],
        default="sqlite",
        help="Type of database to use (default: sqlite)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="sample.db",
        help="Path to the database file (for SQLite)"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Starting workflow with {args.db_type} database")
    
    if args.db_type == "sqlite":
        # Check if database exists
        db_path = Path(args.db_path)
        if not db_path.exists():
            logger.error(f"Database file not found: {args.db_path}")
            return
        
        # Connect to SQLite database
        conn = sqlite3.connect(args.db_path)
        logger.info(f"Connected to SQLite database: {args.db_path}")
        
        # Import and run feature engineering
        logger.info("Step 1: Running feature engineering...")
        import create_dataset
        create_dataset.create_new_features(conn)
        
        # Import and run model training
        logger.info("Step 2: Running model training and evaluation...")
        import train_model
        train_model.run_training(conn)
        
        conn.close()
        logger.info("Workflow completed successfully!")
        
    elif args.db_type == "postgresql":
        # PostgreSQL implementation placeholder
        logger.warning("PostgreSQL support not yet implemented")
        # Future: Add PostgreSQL connection and workflow
    
    else:
        logger.error(f"Unsupported database type: {args.db_type}")


if __name__ == "__main__":
    main()