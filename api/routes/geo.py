"""
Geographic Distribution API Routes

Provides endpoints for employee geographic distribution data.
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from src.database import get_database

router = APIRouter(prefix="/api/geo", tags=["Geographic"])


@router.get("/distribution")
async def get_geo_distribution() -> List[Dict[str, Any]]:
    """
    Get employee count by country.
    
    Returns:
        List of countries with employee counts.
    """
    try:
        db = get_database()
        distribution = db.get_employee_distribution_by_country()
        
        # Calculate total for percentage
        total = sum(item['count'] for item in distribution)
        
        # Add percentage and filter out Remote/Unknown
        result = []
        for item in distribution:
            country = item['country']
            count = item['count']
            
            # Skip Remote and Unknown for map display
            if country in ['Remote', 'Unknown', None, '']:
                continue
                
            result.append({
                'country': country,
                'count': count,
                'percentage': round((count / total) * 100, 1) if total > 0 else 0
            })
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_geo_summary() -> Dict[str, Any]:
    """
    Get geographic distribution summary including Remote workers.
    
    Returns:
        Summary with total, countries, and remote count.
    """
    try:
        db = get_database()
        distribution = db.get_employee_distribution_by_country()
        
        total = sum(item['count'] for item in distribution)
        remote_count = next((item['count'] for item in distribution if item['country'] == 'Remote'), 0)
        countries_count = len([item for item in distribution if item['country'] not in ['Remote', 'Unknown', None, '']])
        
        return {
            'total_employees': total,
            'countries_represented': countries_count,
            'remote_workers': remote_count,
            'remote_percentage': round((remote_count / total) * 100, 1) if total > 0 else 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
