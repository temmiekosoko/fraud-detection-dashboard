from typing import Union, Optional
from dataclasses import dataclass


@dataclass
class ModelPrediction:
    """
    Use for step 4 in the test only.
    """

    retailer_id: str
    transaction_id: int
    item_id: int
    return_fraud: bool = False
    model_pred: bool = False
    return_rate: float = 0
    num_pur: float = 0
    num_ret: float = 0
    num_true_ret: float = 0
    amt_ret: float = 0
    amount: float = 0
    new_pred: Optional[bool] = None  # Changed this line for newer python versions

    @property
    def is_correct(self) -> bool:
        if self.new_pred is None:
            return self.model_pred == self.return_fraud
        else:
            return self.new_pred == self.return_fraud

    @property
    def true_return_rate(self) -> Optional[float]:  
        try:
            return self.num_true_ret / self.num_pur
        except ZeroDivisionError:
            return None