# Return Fraud Detection - Solution Documentation

## Executive Summary

This project improves Appriss's Verify model for retail return fraud detection by engineering new features that filter out false signals and better capture customer behavior patterns. The enhanced Random Forest model achieves a 91% improvement in F1 score (0.39 → 0.74) compared to the baseline, while maintaining high precision (82%) to minimize false positives.

Identified that ~49% of historical returns were "quick voids" (< 5 minutes) - cashier errors rather than customer behavior - creating significant noise in the original model.

---

## Technical Approach

### 1. Feature Engineering

All feature engineering was performed directly in SQLite to ensure portability and reproducibility.

**Primary Innovation: `num_true_ret`**
- **Problem:** Original `num_ret` included immediate cashier voids, artificially inflating return counts
- **Solution:** Excluded returns made within 5 minutes of purchase using SQLite's `julianday()` function
- **Impact:** Average return count dropped from 4.99 to 2.55, providing cleaner fraud signals

**Additional Engineered Features:**
1. **`return_to_purchase_ratio`** - Normalized return rate (num_true_ret / num_pur)
2. **`days_since_last_return`** - Recency indicator for fraud escalation
3. **`item_repeat_return_ratio`** - Wardrobing detection (same item repeatedly returned)
4. **`return_acceleration`** - Fraud velocity (returns last 90d vs 91-365d)
5. **`avg_return_amount`** - Average dollar value per return
6. **`high_value_return_flag`** - Binary flag for returns ≥ 90th percentile
7. **`is_first_return`** - Binary flag for customers with no history (45.6% of data)

**SQL Implementation Highlights:**
- CTEs for modular query construction
- Robust timestamp arithmetic using `julianday()` for minute-level precision
- LEFT JOINs to handle orphan returns (purchases outside data window)

### 2. Data Pipeline Design

**OOP Architecture:**
- Abstract `DataLoader` base class for extensibility
- `SQLiteDataLoader` implementation with validation
- Designed for future PostgreSQL support without code changes

**Preprocessing Strategy:**
- Median imputation for missing values (primarily first-time returners)
- StandardScaler for feature normalization
- Stratified train-test split (80/20) to maintain 1.1% fraud rate

### 3. Class Imbalance Handling

**Challenge:** Severe imbalance (98.9% non-fraud, 1.1% fraud)

**Solutions Implemented:**
- **Logistic Regression:** `class_weight='balanced'` 
- **Random Forest:** `class_weight='balanced'`
- **XGBoost:** `scale_pos_weight=89` (ratio of negative to positive class)

### 4. Model Selection & Training

Baseline probabilities were thresholded at 0.5 to enable fair precision-recall comparison.

**Models Evaluated:**
1. **Baseline** - Existing model predictions (conservative threshold)
2. **Logistic Regression** - Linear baseline with balanced weights
3. **Random Forest** - Ensemble method with 100 trees
4. **XGBoost** - Gradient boosting with class weighting

**Training Details:**
- 5-fold stratified cross-validation for robust evaluation
- Random state = 42 for reproducibility
- Feature scaling applied to Logistic Regression only (tree models scale-invariant)

---

## Results & Model Comparison

Metrics reported on test split (20 % hold-out, stratified).

### Performance Metrics (Test Set)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Baseline** | 99.12% | 82.76% | 25.53% | **39.02%** | 0.8877 |
| Logistic Regression | 88.26% | 6.79% | 75.53% | 12.47% | 0.8709 |
| **Random Forest** ⭐ | 99.48% | 82.05% | 68.09% | **74.42%** | 0.8769 |
| XGBoost | 98.36% | 37.57% | 72.34% | 49.45% | 0.8633 |

### Cross-Validation Results

| Model | CV ROC-AUC (mean ± std) |
|-------|-------------------------|
| Random Forest | 0.8679 ± 0.0165 |
| XGBoost | 0.8615 ± 0.0217 |
| Logistic Regression | 0.8372 ± 0.0188 |

### Key Insights

**1. Random Forest Wins by F1 Score**
- **91% improvement** over baseline (0.39 → 0.74 F1)
- Balances precision (82%) and recall (68%) effectively
- Most stable performance in cross-validation (lowest std dev)

**2. Baseline Model Analysis**
- Extremely conservative: only predicts fraud when very confident
- High precision (83%) but low recall (26%) - misses most fraud
- ROC-AUC of 0.89 shows good ranking ability, but poor decision threshold

**3. Logistic Regression Struggles**
- Low precision (7%) suggests linear boundaries insufficient for this problem
- High recall (76%) but at cost of many false positives
- Convergence warnings indicate optimization difficulties with imbalanced data

**4. XGBoost Middle Ground**
- Decent F1 (49%) but lower than Random Forest
- More false positives than Random Forest (lower precision)
- Slightly lower cross-validation stability

### Feature Importance (Random Forest)

Top predictive features:
1. **amt_ret** - Historical return dollar amount
2. **return_to_purchase_ratio** - Newly engineered normalized rate
3. **num_true_ret** - filtered return count
4. **num_pur** - Purchase volume
5. **return_rate** - Original return rate

**Key Finding:** Our new features (`num_true_ret`, `return_to_purchase_ratio`) rank highly, validating the quick-void filtering approach.

---

## Business Impact

### Fraud Detection Improvement
- **Catch 68% of fraud cases** (up from 26%)
- **82% precision** - minimize legitimate customer friction
- **2.67x more fraud caught** while maintaining similar precision

### Cost-Benefit Analysis
Assuming:
- Average fraudulent return: $150
- Cost of false positive (angry customer): $50
- 10,000 returns/day, 1.1% fraud rate (110 fraud cases)

**Baseline Model:**
- Catches: 28 fraud cases ($4,200 saved)
- False positives: ~6 ($300 cost)
- Net: $3,900/day

**Random Forest:**
- Catches: 75 fraud cases ($11,250 saved)
- False positives: ~16 ($800 cost)
- Net: **$10,450/day** → **168% improvement**

---

## Code Quality & Architecture

### OOP Design Patterns
- **Abstract Base Class:** `DataLoader` enables database-agnostic pipeline
- **Factory Pattern:** `main.py` selects loader based on `--db-type` argument
- **Single Responsibility:** Separate modules for ETL, training, testing

### Testing Strategy
- 17 unit tests covering `ModelPrediction` class
- Edge case handling (division by zero, null values)
- 100% coverage on tested class
- Run with: `pytest test_model_prediction.py -v`

### Logging & Observability
- INFO-level logging throughout pipeline
- Performance metrics logged at each stage
- Visualization outputs saved to `plots/` directory

### Reproducibility
- Fixed random seed (42) across all models
- Stratified splits maintain class distribution
- Requirements.txt pins all dependencies

---

## Repository Structure

```
appriss_takehome/
├── create_dataset.py          # Feature engineering SQL + table creation
├── data_etl.py                # Abstract DataLoader + SQLiteDataLoader
├── main.py                    # CLI entry point with db-type argument
├── train_model.py             # Model training, evaluation, visualization
├── test_model_prediction.py   # Unit tests for ModelPrediction class
├── model_prediction.py        # ModelPrediction dataclass (provided)
├── eda.ipynb                  # Exploratory data analysis notebook
├── solution.md                # This documentation
├── requirements.txt           # Python dependencies
├── README.md                  # Assignment instructions
├── sample.db                  # SQLite database
└── plots/                     # Generated visualizations
    ├── confusion_matrix_*.png
    ├── feature_importance_*.png
    └── roc_curves_comparison.png
```

---

## How to Run

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Feature Engineering
```bash
python create_dataset.py sample.db
```

### Model Training
```bash
# Full pipeline
python main.py --db-type sqlite --db-path sample.db

# Or directly
python train_model.py sample.db
```

### Testing
```bash
# Run unit tests
pytest test_model_prediction.py -v

# With coverage
pytest test_model_prediction.py -v --cov=model_prediction
```

### Exploratory Data Analysis
```bash
jupyter notebook eda.ipynb
```

---