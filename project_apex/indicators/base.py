from abc import ABC, abstractmethod
import pandas as pd


class Indicator(ABC):
    """
    Abstract base class for all technical indicators.
    """

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the indicator values based on the provided data.
        
        Args:
            data (pd.DataFrame): DataFrame containing at least 'close' prices.
                                 Might also contain 'open', 'high', 'low', 'volume'.
                                 
        Returns:
            pd.DataFrame: The original DataFrame with the new indicator column(s) added.
        """
        pass
