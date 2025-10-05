"""
Main Feature Engineering Script

Creates new features for the return fraud detection model, including:
- num_true_ret: Returns excluding quick voids (< 5 minutes)
- Additional engineered features for improved model performance 
"""

import sqlite3
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_new_features(conn: sqlite3.Connection) -> None:
    """
    Create model_data_new table with additional engineered features.
    
    Args:
        conn: SQLite database connection
    """
    cursor = conn.cursor()
    
    # SQL query to calculate num_true_ret
    # This counts returns in the past 365 days, excluding returns made within 5 minutes of purchase
    sql_num_true_ret = """
    CREATE TABLE IF NOT EXISTS model_data_new AS
    WITH return_transactions AS (
        -- Get all return transactions with their details
        SELECT 
            t_ret.retailer_id,
            t_ret.transaction_id,
            t_ret.item_id,
            t_ret.transaction_time as return_time,
            t_ret.orig_transaction_id,
            c.customer_id
        FROM transactions t_ret
        JOIN customers c 
            ON t_ret.retailer_id = c.retailer_id 
            AND t_ret.transaction_id = c.transaction_id
        WHERE t_ret.sale = 'N'  -- Returns only
    ),
    purchase_transactions AS (
        -- Get all purchase transactions with their times
        SELECT 
            retailer_id,
            transaction_id,
            transaction_time as purchase_time
        FROM transactions
        WHERE sale = 'Y'  -- Purchases only
    ),
    true_returns AS (
        -- Calculate num_true_ret for each return
        SELECT 
            rt_current.retailer_id,
            rt_current.transaction_id,
            rt_current.item_id,
            rt_current.customer_id,
            COUNT(DISTINCT CASE 
                WHEN rt_past.transaction_id IS NOT NULL 
                     AND (pt.purchase_time IS NULL OR 
                          (julianday(rt_past.return_time) - julianday(pt.purchase_time)) * 24 * 60 >= 5)
                THEN rt_past.transaction_id || '-' || rt_past.item_id
                ELSE NULL
            END) as num_true_ret
        FROM return_transactions rt_current
        LEFT JOIN return_transactions rt_past
            ON rt_current.retailer_id = rt_past.retailer_id
            AND rt_current.customer_id = rt_past.customer_id
            AND rt_past.return_time < rt_current.return_time
            AND julianday(rt_current.return_time) - julianday(rt_past.return_time) <= 365
        LEFT JOIN purchase_transactions pt
            ON rt_past.retailer_id = pt.retailer_id
            AND rt_past.orig_transaction_id = pt.transaction_id
        GROUP BY 
            rt_current.retailer_id,
            rt_current.transaction_id,
            rt_current.item_id,
            rt_current.customer_id
    ),
    last_return_dates AS (
        -- Calculate days since last return for each current return
        SELECT 
            rt_current.retailer_id,
            rt_current.transaction_id,
            rt_current.item_id,
            rt_current.customer_id,
            MAX(julianday(rt_current.return_time) - julianday(rt_past.return_time)) as days_since_last_return
        FROM return_transactions rt_current
        LEFT JOIN return_transactions rt_past
            ON rt_current.retailer_id = rt_past.retailer_id
            AND rt_current.customer_id = rt_past.customer_id
            AND rt_past.return_time < rt_current.return_time
            AND julianday(rt_current.return_time) - julianday(rt_past.return_time) <= 365
        LEFT JOIN purchase_transactions pt
            ON rt_past.retailer_id = pt.retailer_id
            AND rt_past.orig_transaction_id = pt.transaction_id
        WHERE rt_past.transaction_id IS NULL 
              OR pt.purchase_time IS NULL 
              OR (julianday(rt_past.return_time) - julianday(pt.purchase_time)) * 24 * 60 >= 5
        GROUP BY 
            rt_current.retailer_id,
            rt_current.transaction_id,
            rt_current.item_id,
            rt_current.customer_id
    ),
    item_repeat_returns AS (
        -- Calculate item repeat return ratio (wardrobing indicator)
        SELECT 
            rt_current.retailer_id,
            rt_current.transaction_id,
            rt_current.item_id,
            rt_current.customer_id,
            COUNT(DISTINCT CASE 
                WHEN rt_past.item_id = rt_current.item_id 
                     AND (pt.purchase_time IS NULL OR 
                          (julianday(rt_past.return_time) - julianday(pt.purchase_time)) * 24 * 60 >= 5)
                THEN rt_past.transaction_id
                ELSE NULL
            END) as repeat_item_returns,
            COUNT(DISTINCT CASE 
                WHEN rt_past.transaction_id IS NOT NULL 
                     AND (pt.purchase_time IS NULL OR 
                          (julianday(rt_past.return_time) - julianday(pt.purchase_time)) * 24 * 60 >= 5)
                THEN rt_past.transaction_id
                ELSE NULL
            END) as total_true_returns
        FROM return_transactions rt_current
        LEFT JOIN return_transactions rt_past
            ON rt_current.retailer_id = rt_past.retailer_id
            AND rt_current.customer_id = rt_past.customer_id
            AND rt_past.return_time < rt_current.return_time
            AND julianday(rt_current.return_time) - julianday(rt_past.return_time) <= 365
        LEFT JOIN purchase_transactions pt
            ON rt_past.retailer_id = pt.retailer_id
            AND rt_past.orig_transaction_id = pt.transaction_id
        GROUP BY 
            rt_current.retailer_id,
            rt_current.transaction_id,
            rt_current.item_id,
            rt_current.customer_id
    ),
    return_acceleration AS (
        -- Calculate return acceleration (fraud escalation indicator)
        -- Compares returns in last 90 days vs 91-365 days
        SELECT 
            rt_current.retailer_id,
            rt_current.transaction_id,
            rt_current.item_id,
            rt_current.customer_id,
            COUNT(DISTINCT CASE 
                WHEN julianday(rt_current.return_time) - julianday(rt_past.return_time) <= 90
                     AND (pt.purchase_time IS NULL OR 
                          (julianday(rt_past.return_time) - julianday(pt.purchase_time)) * 24 * 60 >= 5)
                THEN rt_past.transaction_id
                ELSE NULL
            END) as returns_last_90,
            COUNT(DISTINCT CASE 
                WHEN julianday(rt_current.return_time) - julianday(rt_past.return_time) > 90
                     AND julianday(rt_current.return_time) - julianday(rt_past.return_time) <= 365
                     AND (pt.purchase_time IS NULL OR 
                          (julianday(rt_past.return_time) - julianday(pt.purchase_time)) * 24 * 60 >= 5)
                THEN rt_past.transaction_id
                ELSE NULL
            END) as returns_91_365
        FROM return_transactions rt_current
        LEFT JOIN return_transactions rt_past
            ON rt_current.retailer_id = rt_past.retailer_id
            AND rt_current.customer_id = rt_past.customer_id
            AND rt_past.return_time < rt_current.return_time
            AND julianday(rt_current.return_time) - julianday(rt_past.return_time) <= 365
        LEFT JOIN purchase_transactions pt
            ON rt_past.retailer_id = pt.retailer_id
            AND rt_past.orig_transaction_id = pt.transaction_id
        GROUP BY 
            rt_current.retailer_id,
            rt_current.transaction_id,
            rt_current.item_id,
            rt_current.customer_id
    )
    SELECT 
        md.*,
        COALESCE(tr.num_true_ret, 0) as num_true_ret,
        lrd.days_since_last_return,
        CASE 
            WHEN irr.total_true_returns > 0 
            THEN CAST(irr.repeat_item_returns AS REAL) / irr.total_true_returns
            ELSE NULL
        END as item_repeat_return_ratio,
        CASE
            WHEN ra.returns_91_365 > 0
            THEN (CAST(ra.returns_last_90 AS REAL) / ra.returns_91_365) - 1
            ELSE NULL
        END as return_acceleration
    FROM model_data md
    LEFT JOIN true_returns tr
        ON md.retailer_id = tr.retailer_id
        AND md.transaction_id = tr.transaction_id
        AND md.item_id = tr.item_id
    LEFT JOIN last_return_dates lrd
        ON md.retailer_id = lrd.retailer_id
        AND md.transaction_id = lrd.transaction_id
        AND md.item_id = lrd.item_id
    LEFT JOIN item_repeat_returns irr
        ON md.retailer_id = irr.retailer_id
        AND md.transaction_id = irr.transaction_id
        AND md.item_id = irr.item_id
    LEFT JOIN return_acceleration ra
        ON md.retailer_id = ra.retailer_id
        AND md.transaction_id = ra.transaction_id
        AND md.item_id = ra.item_id
    """
    
    logger.info("Creating model_data_new table with num_true_ret, days_since_last_return, item_repeat_return_ratio, and return_acceleration features...")
    
    # Drop existing table if it exists
    cursor.execute("DROP TABLE IF EXISTS model_data_new")
    
    # Execute the main query
    cursor.execute(sql_num_true_ret)
    conn.commit()
    
    # Validate the results
    cursor.execute("SELECT COUNT(*) FROM model_data_new")
    row_count = cursor.fetchone()[0]
    logger.info(f"Created model_data_new with {row_count} rows")
    
    # Check for nulls in num_true_ret
    cursor.execute("SELECT COUNT(*) FROM model_data_new WHERE num_true_ret IS NULL")
    null_count = cursor.fetchone()[0]
    logger.info(f"Null values in num_true_ret: {null_count}")
    
    # Compare num_ret vs num_true_ret
    cursor.execute("""
        SELECT 
            AVG(num_ret) as avg_num_ret,
            AVG(num_true_ret) as avg_num_true_ret,
            AVG(num_ret - num_true_ret) as avg_diff
        FROM model_data_new
    """)
    stats = cursor.fetchone()
    logger.info(f"Average num_ret: {stats[0]:.2f}")
    logger.info(f"Average num_true_ret: {stats[1]:.2f}")
    logger.info(f"Average difference (quick voids): {stats[2]:.2f}")
    
    # Add additional engineered features
    logger.info("Adding additional engineered features...")
    
    # Calculate high value threshold (90th percentile of return amounts)
    cursor.execute("SELECT amount FROM model_data_new ORDER BY amount DESC LIMIT 1 OFFSET ?", 
                   (int(row_count * 0.1),))
    high_value_threshold = cursor.fetchone()
    if high_value_threshold:
        high_value_threshold = high_value_threshold[0]
    else:
        high_value_threshold = 0
    logger.info(f"High value return threshold (90th percentile): ${high_value_threshold:.2f}")
    
    add_features_sql = f"""
    -- Feature 2: return_to_purchase_ratio (renamed from true_return_rate)
    ALTER TABLE model_data_new ADD COLUMN return_to_purchase_ratio REAL;
    
    UPDATE model_data_new
    SET return_to_purchase_ratio = CASE 
        WHEN num_pur > 0 THEN CAST(num_true_ret AS REAL) / num_pur
        ELSE NULL
    END;
    
    -- Feature 3: avg_return_amount
    ALTER TABLE model_data_new ADD COLUMN avg_return_amount REAL;
    
    UPDATE model_data_new
    SET avg_return_amount = CASE 
        WHEN num_true_ret > 0 THEN CAST(amt_ret AS REAL) / num_true_ret
        ELSE NULL
    END;
    
    -- Feature 4: high_value_return_flag
    ALTER TABLE model_data_new ADD COLUMN high_value_return_flag INTEGER;
    
    UPDATE model_data_new
    SET high_value_return_flag = CASE 
        WHEN amount >= {high_value_threshold} THEN 1
        ELSE 0
    END;
    """
    
    cursor.executescript(add_features_sql)
    conn.commit()
    
    logger.info("Feature engineering complete!")
    
    # Show sample of new features
    cursor.execute("""
        SELECT 
            retailer_id, 
            transaction_id, 
            item_id, 
            num_ret, 
            num_true_ret,
            return_to_purchase_ratio,
            days_since_last_return,
            item_repeat_return_ratio,
            return_acceleration,
            avg_return_amount,
            high_value_return_flag
        FROM model_data_new 
        LIMIT 5
    """)
    
    logger.info("Sample of new features:")
    for row in cursor.fetchall():
        logger.info(f"  {row}")
    
    # Feature summary statistics
    cursor.execute("""
        SELECT 
            COUNT(*) as total_rows,
            AVG(num_true_ret) as avg_num_true_ret,
            AVG(return_to_purchase_ratio) as avg_return_to_purchase_ratio,
            AVG(days_since_last_return) as avg_days_since_last_return,
            AVG(item_repeat_return_ratio) as avg_item_repeat_return_ratio,
            AVG(return_acceleration) as avg_return_acceleration,
            AVG(avg_return_amount) as avg_avg_return_amount,
            SUM(high_value_return_flag) as high_value_returns
        FROM model_data_new
    """)
    
    summary = cursor.fetchone()
    logger.info("\nFeature Summary Statistics:")
    logger.info(f"  Total rows: {summary[0]}")
    logger.info(f"  Avg num_true_ret: {summary[1]:.2f}")
    logger.info(f"  Avg return_to_purchase_ratio: {summary[2]:.4f}" if summary[2] else "  Avg return_to_purchase_ratio: NULL")
    logger.info(f"  Avg days_since_last_return: {summary[3]:.2f}" if summary[3] else "  Avg days_since_last_return: NULL")
    logger.info(f"  Avg item_repeat_return_ratio: {summary[4]:.4f}" if summary[4] else "  Avg item_repeat_return_ratio: NULL")
    logger.info(f"  Avg return_acceleration: {summary[5]:.4f}" if summary[5] else "  Avg return_acceleration: NULL")
    logger.info(f"  Avg avg_return_amount: ${summary[6]:.2f}" if summary[6] else "  Avg avg_return_amount: NULL")
    logger.info(f"  High value returns: {summary[7]}")


if __name__ == "__main__":
    # For standalone execution
    import sys
    
    db_path = sys.argv[1] if len(sys.argv) > 1 else "sample.db"
    
    conn = sqlite3.connect(db_path)
    create_new_features(conn)
    conn.close()