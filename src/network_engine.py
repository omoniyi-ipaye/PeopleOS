"""
Organizational Network Analysis (ONA) Engine for PeopleOS.

Uses NetworkX to analyze organizational influence patterns and identify
key connectors, isolated employees, and collaboration networks.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from src.logger import get_logger

logger = get_logger('network_engine')

# Try to import NetworkX
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("NetworkX not available. Network analysis will use basic fallback.")


class NetworkEngine:
    """
    Engine for Organizational Network Analysis.
    
    Builds employee collaboration graphs and identifies key influencers,
    connectors, and isolated employees.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the Network Engine.
        
        Args:
            df: DataFrame with employee data including ManagerID.
        """
        self.df = df.copy()
        self.graph: Optional[Any] = None  # nx.Graph when available
        self._build_network()
    
    def _build_network(self) -> None:
        """Build the organizational network graph."""
        if not NETWORKX_AVAILABLE:
            return
        
        self.graph = nx.DiGraph()  # Directed graph for reporting relationships
        
        # Add all employees as nodes
        for _, row in self.df.iterrows():
            emp_id = str(row.get('EmployeeID', ''))
            if not emp_id:
                continue
                
            self.graph.add_node(
                emp_id,
                name=row.get('Name', f'Employee {emp_id}'),
                dept=row.get('Dept', 'Unknown'),
                job_level=row.get('JobLevel', 0),
                tenure=row.get('Tenure', 0),
                risk_score=row.get('AttritionRisk', 0)
            )
        
        # Add edges based on reporting relationships
        if 'ManagerID' in self.df.columns:
            for _, row in self.df.iterrows():
                emp_id = str(row.get('EmployeeID', ''))
                manager_id = str(row.get('ManagerID', ''))
                
                if emp_id and manager_id and manager_id != emp_id:
                    if manager_id in self.graph.nodes:
                        self.graph.add_edge(manager_id, emp_id, relationship='reports_to')
        
        # Add edges based on department (proxy for collaboration)
        if 'Dept' in self.df.columns:
            dept_groups = self.df.groupby('Dept')['EmployeeID'].apply(list).to_dict()
            
            for dept, employees in dept_groups.items():
                employees = [str(e) for e in employees]
                # Connect all employees in same dept (undirected proxy)
                for i, emp1 in enumerate(employees):
                    for emp2 in employees[i+1:]:
                        if emp1 in self.graph.nodes and emp2 in self.graph.nodes:
                            # Add weak collaboration edge
                            if not self.graph.has_edge(emp1, emp2):
                                self.graph.add_edge(emp1, emp2, relationship='collaborates', weight=0.5)
        
        logger.info(f"Built network with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def get_key_influencers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Identify key influencers based on network centrality.
        
        Uses PageRank and Betweenness Centrality to find employees
        whose departure would most impact the organization.
        
        Args:
            limit: Maximum number of influencers to return.
            
        Returns:
            List of influencer dicts with scores and HR interpretation.
        """
        if not NETWORKX_AVAILABLE or self.graph is None or self.graph.number_of_nodes() == 0:
            return self._fallback_influencers(limit)
        
        try:
            # Calculate centrality metrics
            undirected = self.graph.to_undirected()
            
            pagerank = nx.pagerank(undirected, max_iter=100)
            betweenness = nx.betweenness_centrality(undirected)
            degree = dict(undirected.degree())
            
            # Combine into influence score
            influencers = []
            for node in self.graph.nodes:
                pr = pagerank.get(node, 0)
                bt = betweenness.get(node, 0)
                dg = degree.get(node, 0)
                
                # Weighted influence score
                influence = (pr * 0.4) + (bt * 0.4) + (dg / max(degree.values()) * 0.2)
                
                node_data = self.graph.nodes[node]
                
                influencers.append({
                    'employee_id': node,
                    'name': node_data.get('name', f'Employee {node}'),
                    'dept': node_data.get('dept', 'Unknown'),
                    'job_level': node_data.get('job_level', 0),
                    'influence_score': round(influence, 4),
                    'pagerank': round(pr, 4),
                    'betweenness': round(bt, 4),
                    'connections': dg,
                    'risk_score': node_data.get('risk_score', 0)
                })
            
            # Sort by influence score
            influencers.sort(key=lambda x: x['influence_score'], reverse=True)
            
            # Add HR interpretations
            for i, inf in enumerate(influencers[:limit]):
                if i < 3:
                    inf['hr_insight'] = "🔴 Critical Connector: Departure would significantly impact operations."
                    inf['action'] = "Prioritize retention; conduct stay interview."
                elif i < 6:
                    inf['hr_insight'] = "🟡 Key Influencer: Important to team dynamics and information flow."
                    inf['action'] = "Monitor engagement; ensure career growth."
                else:
                    inf['hr_insight'] = "🟢 Solid Contributor: Well-connected within their network."
                    inf['action'] = "Consider for cross-functional projects."
            
            return influencers[:limit]
            
        except Exception as e:
            logger.error(f"Influencer analysis failed: {e}")
            return self._fallback_influencers(limit)
    
    def _fallback_influencers(self, limit: int) -> List[Dict[str, Any]]:
        """Fallback when NetworkX is unavailable."""
        # Use tenure and job level as proxy for influence
        if 'Tenure' not in self.df.columns or 'JobLevel' not in self.df.columns:
            return []
        
        df = self.df.copy()
        df['influence_proxy'] = df['Tenure'] * 0.4 + df['JobLevel'] * 0.6
        df = df.nlargest(limit, 'influence_proxy')
        
        return [
            {
                'employee_id': str(row['EmployeeID']),
                'name': row.get('Name', f"Employee {row['EmployeeID']}"),
                'dept': row.get('Dept', 'Unknown'),
                'influence_score': round(row['influence_proxy'] / df['influence_proxy'].max(), 4),
                'hr_insight': '🟡 Experienced Employee (NetworkX unavailable for full analysis)',
                'action': 'Consider for mentorship programs.'
            }
            for _, row in df.iterrows()
        ]
    
    def get_isolated_employees(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Identify isolated employees with few connections.
        
        These employees may be at risk of disengagement.
        
        Returns:
            List of isolated employee dicts.
        """
        if not NETWORKX_AVAILABLE or self.graph is None:
            return []
        
        try:
            undirected = self.graph.to_undirected()
            degree = dict(undirected.degree())
            
            # Find employees with low connectivity
            isolated = []
            for node, deg in degree.items():
                if deg <= 2:  # Few connections
                    node_data = self.graph.nodes[node]
                    isolated.append({
                        'employee_id': node,
                        'name': node_data.get('name', f'Employee {node}'),
                        'dept': node_data.get('dept', 'Unknown'),
                        'connections': deg,
                        'tenure': node_data.get('tenure', 0),
                        'hr_insight': '⚠️ Low Network Integration: May feel disconnected.',
                        'action': 'Consider mentorship pairing or cross-functional assignment.'
                    })
            
            # Sort by lowest connections first
            isolated.sort(key=lambda x: x['connections'])
            
            return isolated[:limit]
            
        except Exception as e:
            logger.error(f"Isolation analysis failed: {e}")
            return []
    
    def get_network_summary(self) -> Dict[str, Any]:
        """Get summary statistics about the organizational network."""
        if not NETWORKX_AVAILABLE or self.graph is None:
            return {
                'success': False,
                'reason': 'NetworkX not available'
            }
        
        try:
            undirected = self.graph.to_undirected()
            
            return {
                'success': True,
                'total_nodes': self.graph.number_of_nodes(),
                'total_edges': self.graph.number_of_edges(),
                'avg_connections': round(sum(dict(undirected.degree()).values()) / max(1, self.graph.number_of_nodes()), 2),
                'network_density': round(nx.density(undirected), 4),
                'connected_components': nx.number_connected_components(undirected),
                'interpretation': self._interpret_network_health(undirected)
            }
            
        except Exception as e:
            return {'success': False, 'reason': str(e)}
    
    def _interpret_network_health(self, graph) -> str:
        """Generate HR interpretation of network health."""
        density = nx.density(graph)
        components = nx.number_connected_components(graph)
        
        if density > 0.3:
            health = "Strong collaboration culture detected."
        elif density > 0.1:
            health = "Moderate collaboration; opportunities for cross-team initiatives."
        else:
            health = "Siloed organization; consider breaking down department barriers."
        
        if components > 1:
            health += f" Warning: {components} disconnected groups identified."
        
        return health
