"""
Unit tests for ModelPrediction class.

Run with: pytest test_model_prediction.py -v
Or with coverage: pytest test_model_prediction.py -v --cov=model_prediction
"""

import pytest
from model_prediction import ModelPrediction


class TestModelPredictionIsCorrect:
    """Test cases for the is_correct property."""
    
    def test_is_correct_with_model_pred_correct(self):
        """Test is_correct when model_pred matches return_fraud and new_pred is None."""
        pred = ModelPrediction(
            retailer_id='A',
            transaction_id=1001,
            item_id=2001,
            return_fraud=True,
            model_pred=True,
            new_pred=None
        )
        assert pred.is_correct is True
    
    def test_is_correct_with_model_pred_incorrect(self):
        """Test is_correct when model_pred doesn't match return_fraud and new_pred is None."""
        pred = ModelPrediction(
            retailer_id='A',
            transaction_id=1001,
            item_id=2001,
            return_fraud=True,
            model_pred=False,
            new_pred=None
        )
        assert pred.is_correct is False
    
    def test_is_correct_with_new_pred_correct(self):
        """Test is_correct when new_pred matches return_fraud."""
        pred = ModelPrediction(
            retailer_id='A',
            transaction_id=1001,
            item_id=2001,
            return_fraud=True,
            model_pred=False,  # Incorrect
            new_pred=True      # Correct
        )
        assert pred.is_correct is True
    
    def test_is_correct_with_new_pred_incorrect(self):
        """Test is_correct when new_pred doesn't match return_fraud."""
        pred = ModelPrediction(
            retailer_id='A',
            transaction_id=1001,
            item_id=2001,
            return_fraud=True,
            model_pred=True,   # Would be correct
            new_pred=False     # Incorrect
        )
        assert pred.is_correct is False
    
    def test_is_correct_uses_new_pred_when_available(self):
        """Test that new_pred takes precedence over model_pred when set."""
        pred = ModelPrediction(
            retailer_id='A',
            transaction_id=1001,
            item_id=2001,
            return_fraud=False,
            model_pred=False,  # Would be correct
            new_pred=True      # Incorrect - this should be used
        )
        assert pred.is_correct is False
    
    def test_is_correct_both_false(self):
        """Test is_correct when both return_fraud and prediction are False."""
        pred = ModelPrediction(
            retailer_id='A',
            transaction_id=1001,
            item_id=2001,
            return_fraud=False,
            model_pred=False
        )
        assert pred.is_correct is True


class TestModelPredictionTrueReturnRate:
    """Test cases for the true_return_rate property."""
    
    def test_true_return_rate_normal_calculation(self):
        """Test true_return_rate with normal values."""
        pred = ModelPrediction(
            retailer_id='A',
            transaction_id=1001,
            item_id=2001,
            num_true_ret=5.0,
            num_pur=20.0
        )
        assert pred.true_return_rate == 0.25
    
    def test_true_return_rate_zero_returns(self):
        """Test true_return_rate when num_true_ret is zero."""
        pred = ModelPrediction(
            retailer_id='A',
            transaction_id=1001,
            item_id=2001,
            num_true_ret=0.0,
            num_pur=10.0
        )
        assert pred.true_return_rate == 0.0
    
    def test_true_return_rate_zero_purchases(self):
        """Test true_return_rate when num_pur is zero (division by zero)."""
        pred = ModelPrediction(
            retailer_id='A',
            transaction_id=1001,
            item_id=2001,
            num_true_ret=5.0,
            num_pur=0.0
        )
        assert pred.true_return_rate is None
    
    def test_true_return_rate_both_zero(self):
        """Test true_return_rate when both num_true_ret and num_pur are zero."""
        pred = ModelPrediction(
            retailer_id='A',
            transaction_id=1001,
            item_id=2001,
            num_true_ret=0.0,
            num_pur=0.0
        )
        assert pred.true_return_rate is None
    
    def test_true_return_rate_default_values(self):
        """Test true_return_rate with default values (both 0)."""
        pred = ModelPrediction(
            retailer_id='A',
            transaction_id=1001,
            item_id=2001
        )
        assert pred.true_return_rate is None
    
    def test_true_return_rate_fractional_result(self):
        """Test true_return_rate returns exact fraction."""
        pred = ModelPrediction(
            retailer_id='A',
            transaction_id=1001,
            item_id=2001,
            num_true_ret=1.0,
            num_pur=3.0
        )
        expected = 1.0 / 3.0
        assert abs(pred.true_return_rate - expected) < 1e-10
    
    def test_true_return_rate_greater_than_one(self):
        """Test true_return_rate when returns exceed purchases (edge case)."""
        pred = ModelPrediction(
            retailer_id='A',
            transaction_id=1001,
            item_id=2001,
            num_true_ret=10.0,
            num_pur=5.0
        )
        assert pred.true_return_rate == 2.0


class TestModelPredictionInitialization:
    """Test cases for ModelPrediction initialization."""
    
    def test_minimal_initialization(self):
        """Test creating ModelPrediction with only required fields."""
        pred = ModelPrediction(
            retailer_id='A',
            transaction_id=1001,
            item_id=2001
        )
        assert pred.retailer_id == 'A'
        assert pred.transaction_id == 1001
        assert pred.item_id == 2001
        assert pred.return_fraud is False
        assert pred.model_pred is False
        assert pred.new_pred is None
    
    def test_full_initialization(self):
        """Test creating ModelPrediction with all fields."""
        pred = ModelPrediction(
            retailer_id='B',
            transaction_id=2002,
            item_id=3003,
            return_fraud=True,
            model_pred=True,
            return_rate=0.15,
            num_pur=100.0,
            num_ret=15.0,
            num_true_ret=12.0,
            amt_ret=500.0,
            amount=50.0,
            new_pred=False
        )
        assert pred.retailer_id == 'B'
        assert pred.return_fraud is True
        assert pred.num_true_ret == 12.0
        assert pred.new_pred is False
    
    def test_keyword_initialization(self):
        """Test initialization using keyword arguments."""
        pred = ModelPrediction(
            retailer_id='C',
            transaction_id=3003,
            item_id=4004,
            return_fraud=True
        )
        assert pred.return_fraud is True
        assert pred.model_pred is False  # Default
    
    def test_mixed_positional_keyword(self):
        """Test initialization with mixed positional and keyword arguments."""
        pred = ModelPrediction(
            'D',
            4004,
            item_id=5005,
            return_fraud=True
        )
        assert pred.retailer_id == 'D'
        assert pred.transaction_id == 4004
        assert pred.item_id == 5005
        assert pred.return_fraud is True