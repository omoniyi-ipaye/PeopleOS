"""
Clustering Engine module for PeopleOS.

Provides unsupervised learning capabilities to segment employees into groups
based on their attributes (Salary, Tenure, Performance, etc.).
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from src.logger import get_logger

logger = get_logger('clustering_engine')

class ClusteringEngine:
    """
    Engine for grouping employees into clusters using Unsupervised Learning.
    
    Uses K-Means clustering to find natural groupings in the data.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the Clustering Engine.
        
        Args:
            df: DataFrame containing employee data.
        """
        self.df = df
        self.model: Optional[KMeans] = None
        self.scaler = StandardScaler()
        self.feature_cols: List[str] = []
        self.cluster_labels: Optional[np.ndarray] = None
        self.results: Dict[str, Any] = {}
    def _prepare_data(self) -> pd.DataFrame:
        """
        Select and preprocess features for clustering.
        Pivoting to Active Employees Only: Personas should represent current staff.
        
        Returns:
            DataFrame with scaled features.
        """
        if 'Attrition' in self.df.columns:
            active_df = self.df[self.df['Attrition'] == 0]
        else:
            active_df = self.df
            
        # Potential features for clustering
        candidate_cols = [
            'Salary', 'Tenure', 'LastRating', 'Age', 
            'YearsInCurrentRole', 'YearsSinceLastPromotion', 
            'CompaRatio', 'EngagementScore', 'Pulse_Score',
            'InterviewScore'
        ]
        
        # Select available numeric columns
        self.feature_cols = [
            col for col in candidate_cols 
            if col in active_df.columns and pd.api.types.is_numeric_dtype(active_df[col])
        ]
        
        if not self.feature_cols:
            raise ValueError("No suitable numeric columns found for clustering.")
            
        # Drop rows with NaNs in feature columns for training
        X = active_df[self.feature_cols].dropna()
        
        if X.empty:
            raise ValueError("No data remaining after dropping NaNs.")
            
        return X

    def train(self, n_clusters: int = 3, auto_tune: bool = True) -> Dict[str, Any]:
        """
        Train the clustering model.
        
        Args:
            n_clusters: Number of clusters to create (if auto_tune is False).
            auto_tune: If True, automatically finds optimal n_clusters using Silhouette Score.
            
        Returns:
            Dictionary with clustering results.
        """
        try:
            X = self._prepare_data()
            if len(X) < 10:
                logger.warning("Insufficient data for clustering (need <10 rows).")
                return {'success': False, 'reason': 'Insufficient data'}
                
            X_scaled = self.scaler.fit_transform(X)
            
            best_n = n_clusters
            best_score = -1.0
            best_model = None
            
            if auto_tune:
                # Try 2 to 6 clusters
                max_k = min(6, len(X))
                for k in range(2, max_k):
                    model = KMeans(n_clusters=k, random_state=42, n_init='auto')
                    labels = model.fit_predict(X_scaled)
                    score = silhouette_score(X_scaled, labels)
                    
                    if score > best_score:
                        best_score = score
                        best_n = k
                        best_model = model
            else:
                best_model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                best_model.fit(X_scaled)
                
            self.model = best_model
            
            # Predict labels for the full dataset (handling NaNs by skipping or filling?)
            # For simplicity, we only label rows we trained on (dropped NaNs)
            # A more robust approach would impute NaNs.
            self.cluster_labels = self.model.predict(X_scaled)
            
            # Identify valid employee IDs for these clusters
            # We need to map back to original DataFrame index
            valid_indices = X.index

            # Add cluster labels to a copy of the original dataframe to get departments
            df_labeled = self.df.loc[valid_indices].copy()
            df_labeled['Cluster'] = self.cluster_labels

            # Cluster counts
            cluster_counts = df_labeled['Cluster'].value_counts().to_dict()

            # Generate cluster summaries
            feature_summary = df_labeled.groupby('Cluster')[self.feature_cols].mean()

            # Top departments per cluster
            top_depts = {}
            if 'Dept' in df_labeled.columns:
                for cluster_id in range(best_n):
                    depts = df_labeled[df_labeled['Cluster'] == cluster_id]['Dept'].value_counts().head(3).to_dict()
                    top_depts[cluster_id] = depts

            # Generate persona names/descriptions
            cluster_descriptions = self._generate_cluster_descriptions(feature_summary)

            def _to_python_types(obj):
                """Recursively convert numpy types to native Python types."""
                if isinstance(obj, dict):
                    return {str(k) if isinstance(k, (np.integer, np.floating)) else k: _to_python_types(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple, np.ndarray)):
                    return [_to_python_types(item) for item in obj]
                elif isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                elif pd.isna(obj):
                    return None
                return obj

            self.results = _to_python_types({
                'success': True,
                'n_clusters': best_n,
                'silhouette_score': best_score if auto_tune else None,
                'feature_summary': feature_summary.to_dict(),
                'cluster_counts': cluster_counts,
                'top_departments': top_depts,
                'cluster_descriptions': cluster_descriptions,
                'labels': {str(self.df.loc[idx]['EmployeeID']): int(label) for idx, label in zip(valid_indices, self.cluster_labels)},
                'excluded_count': len(self.df) - len(valid_indices)
            })
            
            return self.results
            
        except Exception as e:
            logger.error(f"Clustering training failed: {e}")
            return {'success': False, 'reason': str(e)}

    def _generate_cluster_descriptions(self, summary: pd.DataFrame) -> Dict[int, str]:
        """
        Generate human-readable names for clusters based on their centroids.
        
        Args:
            summary: DataFrame of cluster means.
            
        Returns:
            Dict mapping cluster ID -> Description string.
        """
        descriptions = {}
        global_means = self.df[self.feature_cols].mean()
        
        for cluster_id, row in summary.iterrows():
            traits = []
            
            # Identify defining traits - calculate z-score for each column correctly
            sorted_traits = []
            for col in self.feature_cols:
                col_mean = global_means[col]
                col_std = self.df[col].std()
                z_score = (row[col] - col_mean) / col_std if col_std > 0 else 0
                if abs(z_score) > 0.4:
                    sorted_traits.append((col, z_score))
            
            # Sort by absolute impact
            sorted_traits.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Map technical names to friendly names
            COLUMN_NAME_MAPPING = {
                'YearsSinceLastPromotion': 'Stagnation',
                'Pulse_Score': 'Engagement',
                'LastRating': 'Performance',
                'CompaRatio': 'Compensation Level',
                'JobLevel': 'Seniority',
                'PriorExperienceYears': 'Experience',
                'ManagerChangeCount': 'Stability',
                'InterviewScore': 'Interview Quality',
                'Training_Hours': 'Upskilling',
                'Overtime_Hours': 'Workload',
                'Sick_Leaf_Days': 'Health Checks',
                'Remote_Days_Ratio': 'Remote Work',
                'Team_Size': 'Team Scale',
                'Salary': 'Compensation'
            }

            traits = []
            for col, z in sorted_traits[:3]:
                prefix = "High" if z > 0 else "Low"
                friendly_name = COLUMN_NAME_MAPPING.get(col, col)
                traits.append(f"{prefix} {friendly_name}")
            
            # Construct description
            if not traits:
                desc = "Average Profile"
            else:
                desc = ", ".join(traits)
                
            descriptions[cluster_id] = desc
            
        return descriptions

    def get_employee_clusters(self) -> pd.DataFrame:
        """
        Get DataFrame with EmployeeID and their assigned Cluster.
        
        Returns:
            DataFrame with 'EmployeeID', 'Cluster', 'Cluster_Name'
        """
        if not self.results or not self.results.get('success'):
            return pd.DataFrame()
            
        labels_map = self.results['labels']
        descriptions = self.results['cluster_descriptions']
        
        # Create result DF
        result_df = self.df.loc[labels_map.keys(), ['EmployeeID']].copy()
        result_df['Cluster'] = [labels_map[i] for i in result_df.index]
        result_df['Cluster_Name'] = result_df['Cluster'].map(descriptions)
        
        return result_df
