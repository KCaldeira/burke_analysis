import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from data_processing.generate_bootstrap_data import (
    load_data,
    bootstrap_model,
    create_dummies,
    create_lagged_variables
)

class TestBootstrap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load data once for all tests."""
        cls.data = load_data()
        print("\nAvailable columns in dataset:")
        print(cls.data.columns.tolist())
    
    def test_data_loading(self):
        """Test that data is loaded correctly."""
        self.assertIsInstance(self.data, pd.DataFrame)
        self.assertTrue('growthWDI' in self.data.columns)
        self.assertTrue('UDel_temp_popweight' in self.data.columns)
        self.assertTrue('UDel_precip_popweight' in self.data.columns)
    
    def test_dummy_creation(self):
        """Test creation of dummy variables."""
        df = create_dummies(self.data)
        self.assertTrue(any(col.startswith('year_') for col in df.columns))
        self.assertTrue(any(col.startswith('iso_') for col in df.columns))
    
    def test_lagged_variables(self):
        """Test creation of lagged variables."""
        df = create_lagged_variables(self.data)
        self.assertTrue('L1_UDel_temp_popweight' in df.columns)
        self.assertTrue('L5_UDel_temp_popweight' in df.columns)
    
    def test_bootstrap_no_lag(self):
        """Test no-lag bootstrap model."""
        results = bootstrap_model(self.data, model_type='no_lag', n_bootstrap=10)
        self.assertEqual(len(results), 11)  # 1 baseline + 10 bootstrap
        self.assertTrue(all(col in results.columns for col in ['run', 'temp', 'temp2', 'prec', 'prec2']))
    
    def test_bootstrap_rich_poor(self):
        """Test rich/poor bootstrap model."""
        results = bootstrap_model(self.data, model_type='rich_poor', n_bootstrap=10)
        self.assertEqual(len(results), 11)
        self.assertTrue(all(col in results.columns for col in ['run', 'temp', 'temppoor', 'temp2', 'temp2poor']))
    
    def test_bootstrap_5lag(self):
        """Test 5-lag bootstrap model."""
        results = bootstrap_model(self.data, model_type='5lag', n_bootstrap=10)
        self.assertEqual(len(results), 11)
        self.assertTrue('tlin' in results.columns)
        self.assertTrue('tsq' in results.columns)
    
    def test_bootstrap_rich_poor_5lag(self):
        """Test rich/poor 5-lag bootstrap model."""
        results = bootstrap_model(self.data, model_type='rich_poor_5lag', n_bootstrap=10)
        self.assertEqual(len(results), 11)
        self.assertTrue(all(col in results.columns for col in ['tlin', 'tlinpoor', 'tsq', 'tsqpoor']))

if __name__ == '__main__':
    unittest.main() 