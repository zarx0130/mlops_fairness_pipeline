# test_backend.py

import unittest 
import pandas as pd  
import numpy as np 
from app import app, FairnessConfig, preprocess, train_model, calc_fairness, inject_bias

class TestBackendFunctions(unittest.TestCase):
    """
    Unit tests for backend func & API endpoints
    """

    @classmethod
    def setUpClass(cls):
        """ load datasets """
        # adult census
        cls.adult_data = pd.read_csv('data/adult.data', header=None, nrows=2000)

        # make headers - requirement for pipeline functionality
        columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                   'marital-status', 'occupation', 'relationship', 'race', 'sex',
                   'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
        cls.adult_data.columns = columns

        # employee attrition
        cls.employee_data = pd.read_csv('data/Employee.csv', nrows=2000)

        print("\nDatasets Loaded")
        print(f"Adult: {len(cls.adult_data)} rows")
        print(f"Employee: {len(cls.employee_data)} rows")
        
    # FairnessConfig tests (only Employee dataset):
    def test_fairness_config_employee(self):
        """ test FairnessConfig with employee attrition dataset """
        config = FairnessConfig(self.employee_data, 'LeaveOrNot', 'Gender')
        
        self.assertEqual(config.target, 'LeaveOrNot')
        self.assertEqual(config.protected_attr, 'Gender')
        self.assertGreater(len(config.feature_cols), 0)
        print(f"Employee dataset: {len(config.feature_cols)} features detected")

    # preprocess() function tests (both datasets):
    def test_preprocess_adult(self):
        """ test preprocess func with adult dataset """
        X, y, protected, preprocessor = preprocess(self.adult_data, 'income', 'sex')
        
        self.assertEqual(X.shape[0], len(y))
        self.assertEqual(len(protected), len(y))
        self.assertEqual(y.dtype, int)
        self.assertEqual(protected.dtype, int)
        self.assertGreater(X.shape[1], 0)
        print(f"Preprocessed Adult: {X.shape[0]} samples, {X.shape[1]} features")

    def test_preprocess_employee(self):
        """ test preproess func with employee dataset """
        X, y, protected, preprocessor = preprocess(self.employee_data, 'LeaveOrNot', 'Gender')
        
        self.assertEqual(X.shape[0], len(y))
        self.assertEqual(len(protected), len(y))
        print(f"Preprocessed Employee: {X.shape[0]} samples, {X.shape[1]} features")

    # model training tests (only employee dataset):
    def test_train_model_employee(self):
        """ test model training with employee dataset """
        X, y, protected, _ = preprocess(self.employee_data, 'LeaveOrNot', 'Gender')
        model, X_fair, y_fair, protected_fair = train_model(X, y, protected)
    
        self.assertIsNotNone(model)
        self.assertEqual(X_fair.shape[0], y_fair.shape[0])
        self.assertEqual(len(protected_fair), len(y_fair))
        
        # check balanced fair set
        self.assertEqual(np.sum(protected_fair == 1), np.sum(protected_fair == 0))
        print(f"Model trained on Employee: {X_fair.shape[0]} fairness samples")

    # bias inject func tests w/ intensity checks (adult dataset only):
    def test_inject_bias(self):
        """ test bias injection & if its flips predictions correctly for adult dataset """
        predictions = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        protected = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        biased = inject_bias(predictions, protected, intensity=0.5)
        
        # flip ~50% of unprivileged pos
        flipped = (predictions != biased) & (protected == 0)
        self.assertGreaterEqual(flipped.sum(), 1)
        self.assertLessEqual(flipped.sum(), 3)
        print(f"Bias injection: flipped {flipped.sum()} predictions")

    def test_inject_bias_zero(self):
        """ test bias injection w/ zero intensity, should do nothing """
        predictions = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        protected = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        biased = inject_bias(predictions, protected, intensity=0)
        
        self.assertTrue(np.array_equal(predictions, biased))
        print("Zero intensity: no changes made")

    def test_inject_bias_max(self):
        """ test bias intjection w/ full intensity """
        predictions = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        protected = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        biased = inject_bias(predictions, protected, intensity=1.0)
        
        # flip all unprivileged positives
        flipped = (predictions != biased) & (protected == 0)
        self.assertEqual(flipped.sum(), 4)
        print("Full intensity: all unprivileged positives flipped")


class TestAPIEndpoints(unittest.TestCase):
    """ Test Flask API endpoints """
    
    def setUp(self):
        """ setup client before tests """
        app.config['TESTING'] = True
        self.client = app.test_client()

    # test health & run batch endpoints to ensure proper response:
    def test_health_endpoint(self):
        """ test /health endpoint & that it returns correct status """
        response = self.client.get('/health')
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'running')
        self.assertIn('model', data)
        print("Health endpoint working")

    def test_run_batch_no_model(self):
        """ test /run_batch without uplaoding model, should fail """
        response = self.client.post('/run_batch', 
                                    json={'bias': False, 'intensity': 0})
        
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data['status'], 'error')
        print("Rejects batch run without model")

if __name__ == '__main__':
    unittest.main()
        