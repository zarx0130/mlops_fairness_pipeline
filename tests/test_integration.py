# test_integration.property

import unittest
from app import app

class TestIntegrationWorkflow(unittest.TestCase):
    """ full API integration testing """

    def setUp(self):
        """ setup test client before """
        app.config['TESTING'] = True
        self.client = app.test_client()

    # test dataset uplaods & training (adult & employee)
    def test_upload_adult(self):
        """ test upload and training using Adult Census dataset """
    
        with open('data/adult.data', 'rb') as f:
            response = self.client.post('/upload', data={
                'dataset': (f, 'adult.data'),
                'target': 'income',
                'protected_attr': 'sex'
            })
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        
        self.assertEqual(data['status'], 'success')
        self.assertIn('baseline', data)
        self.assertIn('stat_diff', data['baseline'])
        print(f"Adult model trained - baseline: {data['baseline']}")

    def test_upload_employee(self):
        """ test upload & training using Employee Attrition dataset """
        with open('data/Employee.csv', 'rb') as f:
            response = self.client.post('/upload', data={
                'dataset': (f, 'Employee.csv'),
                'target': 'LeaveOrNot',
                'protected_attr': 'Gender'
            })
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        
        self.assertEqual(data['status'], 'success')
        print("Employee model trained")
    
    # test batch runs without/with bias (using adult dataset only):
    def test_run_batch_zero_bias(self):
        """ test batch run w/ zero intensity bias injection """
        # upload model
        with open('data/adult.data', 'rb') as f:
            self.client.post('/upload', data={
                'dataset': (f, 'adult.data'),
                'target': 'income',
                'protected_attr': 'sex'
            })
        
        # run batch w/ no bias
        response = self.client.post('/run_batch', json={
            'bias': False,
            'intensity': 0
        })
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('stat_diff', data)
        self.assertIn('eq_opp_diff', data)
        print(f"Zero bias batch: stat_diff={data['stat_diff']:.4f}")

    def test_run_batch_bias(self):
        """ test batch run w/ bias injecction """
        # upload model
        with open('data/adult.data', 'rb') as f:
            self.client.post('/upload', data={
                'dataset': (f, 'adult.data'),
                'target': 'income',
                'protected_attr': 'sex'
            })
        
        # run batch with ~70% bias injection (near-maximum)
        response = self.client.post('/run_batch', json={
            'bias': True,
            'intensity': 0.7
        })
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('stat_diff', data)
        print(f"(70%) Bias batch: stat_diff={data['stat_diff']:.4f}")

    # test that inject_bias() degrades metrics
    def test_bias_metric_degradation(self):
        """ Test that bias injection causes metric degradation """
        # upload model
        with open('data/adult.data', 'rb') as f:
            self.client.post('/upload', data={
                'dataset': (f, 'adult.data'),
                'target': 'income',
                'protected_attr': 'sex'
            })
        
        # run w/ zero bias
        response1 = self.client.post('/run_batch', json={
            'bias': False,
            'intensity': 0
        })
        data1 = response1.get_json()
        
        # run with ~70% bias 
        response2 = self.client.post('/run_batch', json={
            'bias': True,
            'intensity': 0.7
        })
        data2 = response2.get_json()
        
        # metrics w/ bias should be worse
        self.assertGreater(abs(data2['stat_diff']), abs(data1['stat_diff']))
        print(f"Bias degraded metrics: {abs(data1['stat_diff']):.4f} → {abs(data2['stat_diff']):.4f}")

    # test varying intensity
    def test_varying_intensity(self):
        """ Test effect of various bias intensities """
        # upload model
        with open('data/adult.data', 'rb') as f:
            self.client.post('/upload', data={
                'dataset': (f, 'adult.data'),
                'target': 'income',
                'protected_attr': 'sex'
            })
        
        intensities = [0, 0.3, 0.5, 0.7, 1.0]
        results = []
        
        for intensity in intensities:
            response = self.client.post('/run_batch', json={
                'bias': True,
                'intensity': intensity
            })
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            results.append((intensity, abs(data['stat_diff'])))
        
        # higher intensities should have worse output
        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i+1][1], results[i][1])
        
        print(f"Degradation increases with bias intensity: {results}")

    # test running multiple consequitive batches
    def test_multiple_batches(self):
        # upload  model
        with open('data/adult.data', 'rb') as f:
            self.client.post('/upload', data={
                'dataset': (f, 'adult.data'),
                'target': 'income',
                'protected_attr': 'sex'
            })

        results = []
        for i in range(5):
            response = self.client.post('/run_batch', json={
                'bias': i % 2 == 0,
                'intensity': 0.3
            })
            self.assertEqual(response.status_code, 200)
            results.append(response.get_json())
        
        self.assertEqual(len(results), 5)
        print("Ran 5 consecutive batches successfully")

if __name__ == '__main__':
    unittest.main()


        