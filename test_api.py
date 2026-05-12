#!/usr/bin/env python3
"""
Quick test suite for Car Price Prediction API
Run: python test_api.py
"""

import requests
import json
import sys
from time import time

BASE_URL = "http://localhost:8000"

def test_endpoint(method, endpoint, data=None, name="Test"):
    """Test a single endpoint"""
    try:
        start = time()
        if method == "GET":
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=30)
        else:
            response = requests.post(f"{BASE_URL}{endpoint}", json=data, timeout=30)
        elapsed = time() - start
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ {name}")
            print(f"   Status: {response.status_code} | Time: {elapsed:.2f}s")
            
            # Print sample of response
            if isinstance(result, dict):
                keys = list(result.keys())[:3]
                print(f"   Response keys: {keys}")
            return True, result
        else:
            print(f"❌ {name} - HTTP {response.status_code}")
            print(f"   {response.text[:200]}")
            return False, None
    except Exception as e:
        print(f"❌ {name} - {str(e)}")
        return False, None

def main():
    print("🧪 Car Price Prediction API - Test Suite\n")
    print(f"Target: {BASE_URL}\n")
    
    # Test data - typical BMW
    car_input = {
        "brand": "BMW",
        "model": "3 Series",
        "year": 2020,
        "mileage_km": 45000,
        "horsepower": 200,
        "doors": 4,
        "condition_score": 8.5,
        "fuel_type": "Gasoline",
        "transmission": "Automatic",
        "country": "USA",
        "city": "New York",
        "color": "Black"
    }
    
    tests = [
        ("GET", "/api/health", None, "1. Health Check"),
        ("GET", "/api/config", None, "2. Configuration"),
        ("POST", "/api/predict", car_input, "3. Prediction"),
        ("POST", "/api/feature-engineering", car_input, "4. Feature Engineering"),
        ("POST", "/api/explain/shap", car_input, "5. SHAP Explanation"),
        ("POST", "/api/explain/lime", car_input, "6. LIME Explanation"),
        ("POST", "/api/explain/price-effects", car_input, "7. Price Effects Analysis"),
        ("GET", "/api/explain/permutation", None, "8. Permutation Importance (Global)"),
        ("GET", "/api/explain/model-importance", None, "9. Model Feature Importance (Global)"),
        ("GET", "/api/explain/partial-dependence", None, "10. Partial Dependence Plots (Global)"),
        ("GET", "/api/explain/global-summary", None, "11. Global XAI Summary"),
        ("GET", "/api/explain/xai-metrics", None, "12. XAI Quality Metrics"),
        ("POST", "/api/counterfactual", {**car_input, "budget": 40000}, "13. Counterfactual (DiCE)"),
    ]
    
    passed = 0
    failed = 0
    
    for method, endpoint, data, name in tests:
        success, result = test_endpoint(method, endpoint, data, name)
        if success:
            passed += 1
        else:
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("✅ All tests passed! Backend is ready for frontend.")
        return 0
    else:
        print(f"❌ {failed} tests failed. Check backend logs.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
