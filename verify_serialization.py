import pandas as pd
import numpy as np
from src.clustering_engine import ClusteringEngine

def test_serialization():
    # Create sample data with potential numpy types in index or values
    data = {
        'EmployeeID': ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10'],
        'Salary': [50000, 60000, 55000, 80000, 90000, 52000, 58000, 62000, 85000, 95000],
        'Tenure': [1, 2, 1.5, 5, 6, 1.2, 2.5, 3, 5.5, 7],
        'LastRating': [3, 4, 3.5, 5, 5, 3.2, 3.8, 4.2, 4.8, 5]
    }
    df = pd.DataFrame(data)
    
    engine = ClusteringEngine(df)
    results = engine.train(n_clusters=2, auto_tune=False)
    
    print("Clustering Results Type Check:")
    def check_types(obj, path="root"):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(k, (np.integer, np.floating)):
                    print(f"❌ ERROR: Key at {path} is numpy type: {type(k)}")
                check_types(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                check_types(item, f"{path}[{i}]")
        elif isinstance(obj, (np.integer, np.floating, np.ndarray)):
            print(f"❌ ERROR: Value at {path} is numpy type: {type(obj)}")
        else:
            # print(f"✅ {path}: {type(obj)}")
            pass

    check_types(results)
    print("Verification complete.")

if __name__ == "__main__":
    test_serialization()
