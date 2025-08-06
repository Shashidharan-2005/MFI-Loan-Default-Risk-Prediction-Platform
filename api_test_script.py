# create_api_test.py
"""
API Testing Script for MFI Loan Risk Assessment
Tests all API endpoints and validates responses
"""

import requests
import json
import time
from datetime import datetime

class MFIAPITester:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.test_results = []
    
    def log_test(self, test_name, success, message, response_time=None):
        """Log test results"""
        result = {
            'test_name': test_name,
            'success': success,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'response_time': response_time
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        time_info = f" ({response_time:.3f}s)" if response_time else ""
        print(f"{status}: {test_name}{time_info}")
        if not success:
            print(f"    Error: {message}")
        print()
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}/api/health", timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if 'status' in data and data['status'] == 'healthy':
                    self.log_test("Health Check", True, "API is healthy", response_time)
                    return True
                else:
                    self.log_test("Health Check", False, f"Unexpected response: {data}")
                    return False
            else:
                self.log_test("Health Check", False, f"HTTP {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Health Check", False, f"Connection error: {str(e)}")
            return False
    
    def test_model_info_endpoint(self):
        """Test model info endpoint"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}/api/model-info", timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ['model_type', 'version', 'description']
                
                if all(field in data for field in required_fields):
                    self.log_test("Model Info", True, f"Model: {data.get('model_type')}", response_time)
                    return True
                else:
                    self.log_test("Model Info", False, "Missing required fields in response")
                    return False
            else:
                self.log_test("Model Info", False, f"HTTP {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Model Info", False, f"Connection error: {str(e)}")
            return False
    
    def test_single_prediction(self):
        """Test single prediction endpoint"""
        
        # Test data - low risk profile
        test_data = {
            "age": 32,
            "gender": "Female",
            "income": 4500,
            "loan_amount": 7,
            "education": "Secondary",
            "employment": "Formal",
            "marital_status": "Married",
            "location": "Urban",
            "mobile_usage": 24,
            "transaction_freq": 35,
            "previous_loans": 1,
            "previous_defaults": 0
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/predict", 
                json=test_data, 
                timeout=30,
                headers={'Content-Type': 'application/json'}
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                required_fields = [
                    'default_probability', 'debt_to_income_ratio', 
                    'mobile_score', 'credit_score', 'risk_factors', 'recommendation'
                ]
                
                if all(field in data for field in required_fields):
                    prob = data['default_probability']
                    self.log_test(
                        "Single Prediction", 
                        True, 
                        f"Default probability: {prob:.2%}", 
                        response_time
                    )
                    return True, data
                else:
                    missing = [f for f in required_fields if f not in data]
                    self.log_test("Single Prediction", False, f"Missing fields: {missing}")
                    return False, None
            else:
                error_msg = response.text if response.text else f"HTTP {response.status_code}"
                self.log_test("Single Prediction", False, error_msg)
                return False, None
                
        except requests.exceptions.RequestException as e:
            self.log_test("Single Prediction", False, f"Connection error: {str(e)}")
            return False, None
    
    def test_high_risk_prediction(self):
        """Test prediction with high-risk profile"""
        
        # High risk profile
        high_risk_data = {
            "age": 22,
            "gender": "Male",
            "income": 1500,
            "loan_amount": 10,
            "education": "Primary",
            "employment": "Informal",
            "marital_status": "Single",
            "location": "Rural",
            "mobile_usage": 3,
            "transaction_freq": 5,
            "previous_loans": 3,
            "previous_defaults": 2
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/predict", 
                json=high_risk_data, 
                timeout=30,
                headers={'Content-Type': 'application/json'}
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                prob = data.get('default_probability', 0)
                
                # High risk should have probability > 40%
                if prob > 0.4:
                    self.log_test(
                        "High Risk Prediction", 
                        True, 
                        f"High risk detected: {prob:.2%}", 
                        response_time
                    )
                    return True
                else:
                    self.log_test(
                        "High Risk Prediction", 
                        False, 
                        f"Risk too low for high-risk profile: {prob:.2%}"
                    )
                    return False
            else:
                self.log_test("High Risk Prediction", False, f"HTTP {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("High Risk Prediction", False, f"Connection error: {str(e)}")
            return False
    
    def test_batch_prediction(self):
        """Test batch prediction endpoint"""
        
        batch_data = [
            {
                "age": 28, "gender": "Female", "income": 3000, "loan_amount": 6,
                "education": "Secondary", "employment": "Self-employed", 
                "marital_status": "Single", "location": "Urban",
                "mobile_usage": 12, "transaction_freq": 20, 
                "previous_loans": 0, "previous_defaults": 0
            },
            {
                "age": 45, "gender": "Male", "income": 5500, "loan_amount": 9,
                "education": "Higher", "employment": "Formal", 
                "marital_status": "Married", "location": "Urban",
                "mobile_usage": 36, "transaction_freq": 45, 
                "previous_loans": 2, "previous_defaults": 0
            },
            {
                "age": 19, "gender": "Female", "income": 1200, "loan_amount": 8,
                "education": "Primary", "employment": "Informal", 
                "marital_status": "Single", "location": "Rural",
                "mobile_usage": 2, "transaction_freq": 3, 
                "previous_loans": 1, "previous_defaults": 1
            }
        ]
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/batch-predict", 
                json=batch_data, 
                timeout=60,
                headers={'Content-Type': 'application/json'}
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                if 'predictions' in data and len(data['predictions']) == 3:
                    successful_predictions = sum(1 for p in data['predictions'] if 'default_probability' in p)
                    self.log_test(
                        "Batch Prediction", 
                        True, 
                        f"Processed {successful_predictions}/3 predictions", 
                        response_time
                    )
                    return True
                else:
                    self.log_test("Batch Prediction", False, "Unexpected response format")
                    return False
            else:
                self.log_test("Batch Prediction", False, f"HTTP {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Batch Prediction", False, f"Connection error: {str(e)}")
            return False
    
    def test_validation_errors(self):
        """Test input validation"""
        
        # Test missing fields
        incomplete_data = {
            "age": 30,
            "gender": "Female"
            # Missing required fields
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/predict", 
                json=incomplete_data, 
                timeout=10,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 400:
                self.log_test("Validation - Missing Fields", True, "Correctly rejected incomplete data")
            else:
                self.log_test("Validation - Missing Fields", False, f"Expected 400, got {response.status_code}")
            
            # Test invalid data types
            invalid_data = {
                "age": "thirty",  # Should be int
                "gender": "Female",
                "income": 3000,
                "loan_amount": 7,
                "education": "Secondary",
                "employment": "Formal",
                "marital_status": "Married",
                "location": "Urban",
                "mobile_usage": 12,
                "transaction_freq": 20,
                "previous_loans": 0,
                "previous_defaults": 0
            }
            
            response = requests.post(
                f"{self.base_url}/api/predict", 
                json=invalid_data, 
                timeout=10,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 400:
                self.log_test("Validation - Invalid Types", True, "Correctly rejected invalid data types")
            else:
                self.log_test("Validation - Invalid Types", False, f"Expected 400, got {response.status_code}")
            
            # Test business logic validation
            invalid_business_data = {
                "age": 30,
                "gender": "Female",
                "income": 3000,
                "loan_amount": 7,
                "education": "Secondary",
                "employment": "Formal",
                "marital_status": "Married",
                "location": "Urban",
                "mobile_usage": 12,
                "transaction_freq": 20,
                "previous_loans": 1,
                "previous_defaults": 3  # More defaults than loans
            }
            
            response = requests.post(
                f"{self.base_url}/api/predict", 
                json=invalid_business_data, 
                timeout=10,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 400:
                self.log_test("Validation - Business Logic", True, "Correctly rejected invalid business logic")
                return True
            else:
                self.log_test("Validation - Business Logic", False, f"Expected 400, got {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Validation Tests", False, f"Connection error: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run all API tests"""
        
        print("=" * 80)
        print("MFI API TESTING SUITE")
        print("=" * 80)
        print(f"Testing API at: {self.base_url}")
        print(f"Started at: {datetime.now()}")
        print()
        
        # Run tests
        tests = [
            ("Health Check", self.test_health_endpoint),
            ("Model Info", self.test_model_info_endpoint),
            ("Single Prediction", lambda: self.test_single_prediction()[0]),
            ("High Risk Prediction", self.test_high_risk_prediction),
            ("Batch Prediction", self.test_batch_prediction),
            ("Input Validation", self.test_validation_errors)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed_tests += 1
            except Exception as e:
                self.log_test(test_name, False, f"Test error: {str(e)}")
        
        # Summary
        print("=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        
        success_rate = (passed_tests / total_tests) * 100
        print(f"Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! API is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the details above.")
        
        print(f"\nCompleted at: {datetime.now()}")
        
        # Save test results
        with open('api_test_results.json', 'w') as f:
            json.dump(self.test_results, f, indent=2)
        print("\nTest results saved to 'api_test_results.json'")
        
        return passed_tests == total_tests

def main():
    """Main test execution"""
    
    # Check if server is likely running
    import socket
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('localhost', 5000))
        sock.close()
        
        if result != 0:
            print("⚠️  Warning: Flask server doesn't appear to be running on localhost:5000")
            print("   Please start the server with: python app.py")
            print("   Then run this test script again.")
            return False
    except Exception:
        pass
    
    # Run tests
    tester = MFIAPITester()
    return tester.run_all_tests()

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)