import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Configuration
NUM_EMPLOYEES = 800  # Increased size
np.random.seed(42)  # For reproducibility

# Locations and their metadata
LOCATIONS = {
    'United States': {'cities': ['San Francisco', 'New York', 'Chicago', 'Austin'], 'currency': 'USD', 'base_salary': 100000},
    'United Kingdom': {'cities': ['London', 'Manchester'], 'currency': 'GBP', 'base_salary': 75000}, # approx USD equiv
    'Singapore': {'cities': ['Singapore'], 'currency': 'SGD', 'base_salary': 90000},
    'India': {'cities': ['Mumbai', 'Bangalore'], 'currency': 'INR', 'base_salary': 40000},
    'France': {'cities': ['Paris'], 'currency': 'EUR', 'base_salary': 70000},
    'Germany': {'cities': ['Berlin', 'Munich'], 'currency': 'EUR', 'base_salary': 75000},
    'Australia': {'cities': ['Sydney', 'Melbourne'], 'currency': 'AUD', 'base_salary': 85000},
    'Japan': {'cities': ['Tokyo'], 'currency': 'JPY', 'base_salary': 80000},
    'Brazil': {'cities': ['São Paulo'], 'currency': 'BRL', 'base_salary': 45000},
    'Canada': {'cities': ['Toronto', 'Vancouver'], 'currency': 'CAD', 'base_salary': 90000}
}

DEPARTMENTS = ['Engineering', 'Sales', 'Customer Success', 'Product', 'Marketing', 'Finance', 'HR', 'Operations']
JOB_LEVELS = [1, 2, 3, 4, 5, 6] # 1=Junior, ..., 6=Director/VP
HIRE_SOURCES = ['LinkedIn', 'Referral', 'Agency', 'Website', 'Campus', 'Internal', 'JobBoard']
GENDERS = ['Male', 'Female', 'Non-Binary']

def generate_performance_text(rating, dept):
    # Simple templates based on rating
    if rating >= 4.5:
        templates = [
            "Exceptional performer who consistently exceeds expectations. Demonstrates strong leadership and technical skills. Highly collaborative team player.",
            "Star performer with outstanding technical and interpersonal skills. Proactive in identifying improvements. Key contributor to team success.",
            "Top talent. Delivers high-quality work ahead of schedule. Inspires others with their dedication and expertise.",
            "Outstanding contributor with excellent problem-solving abilities. Shows initiative and drives innovation. Strong communication skills and mentorship qualities."
        ]
    elif rating >= 3.5:
        templates = [
            "Solid performer meeting expectations. Good technical foundation with room for growth. Reliable and consistent in delivery.",
            "Consistent contributor who meets job requirements. Technical skills are adequate. Shows potential for development.",
            "Reliable employee with satisfactory performance. Good attention to detail. Positive attitude toward learning.",
            "Meets expectations with occasional exceeding. Shows good collaboration with team. Technical skills are developing well."
        ]
    else:
        templates = [
            "Performance needs improvement in several areas. Technical skills require development. Recommended for coaching and training.",
            "Inconsistent performance requiring attention. Quality of work needs enhancement. Would benefit from clearer goal setting.",
            "Below expectations in key areas. Communication could be more effective. Close monitoring advised.",
            "Performance gap identified in multiple areas. Needs to improve engagement and initiative. Technical training recommended."
        ]
    
    base_text = np.random.choice(templates)
    
    # Add dimension-specific text
    dimensions = {
        'Engineering': ['Strong in Python.', 'Strong in system design.', 'Strong in cloud architecture.', 'Strong in DevOps.', 'Strong in JavaScript.', 'Strong in API development.', 'Strong in code review.'],
        'Sales': ['Strong in closing deals.', 'Strong in pipeline management.', 'Strong in negotiation.', 'Strong in CRM expertise.', 'Strong in client relations.'],
        'Customer Success': ['Strong in account management.', 'Strong in customer retention.', 'Strong in renewal negotiation.', 'Strong in onboarding.', 'Strong in relationship building.'],
        'Product': ['Strong in roadmap planning.', 'Strong in user research.', 'Strong in stakeholder management.', 'Strong in feature prioritization.', 'Strong in agile methodology.'],
        'Marketing': ['Strong in content strategy.', 'Strong in digital marketing.', 'Strong in analytics.', 'Strong in brand development.', 'Strong in campaign management.'],
        'Finance': ['Strong in financial modeling.', 'Strong in forecasting.', 'Strong in budgeting.', 'Strong in audit preparation.', 'Strong in compliance.'],
        'HR': ['Strong in talent acquisition.', 'Strong in employee relations.', 'Strong in performance management.', 'Strong in training.', 'Strong in policy development.'],
        'Operations': ['Strong in process optimization.', 'Strong in logistics.', 'Strong in vendor management.', 'Strong in supply chain.', 'Strong in quality assurance.']
    }
    
    dim_text = np.random.choice(dimensions.get(dept, ['Strong in collaboration.']))
    
    # Add tenure-based nuance
    extras = []
    if np.random.random() > 0.8:
        extras.append("Adapting well to new role.")
    if np.random.random() > 0.9:
        extras.append("Valuable institutional knowledge.")
        
    return f"{base_text} {dim_text} {' '.join(extras)}".strip()

data = []

for i in range(1, NUM_EMPLOYEES + 1):
    emp_id = f"EMP{i:04d}"
    dept = np.random.choice(DEPARTMENTS, p=[0.35, 0.20, 0.15, 0.08, 0.07, 0.05, 0.05, 0.05]) # Weighted
    
    # Location
    country = np.random.choice(list(LOCATIONS.keys()))
    city = np.random.choice(LOCATIONS[country]['cities'])
    
    # Demographics
    age = int(np.random.normal(35, 8))
    age = max(22, min(65, age))
    gender = np.random.choice(GENDERS, p=[0.48, 0.48, 0.04])
    
    # Role & Tenure
    level = np.random.choice(JOB_LEVELS, p=[0.2, 0.3, 0.25, 0.15, 0.08, 0.02])
    
    # Calculating Salary based on Level and Location
    base = LOCATIONS[country]['base_salary']
    salary = int(base * (1 + (level - 1) * 0.25) * np.random.uniform(0.9, 1.1))
    
    # Calculate Tenure
    # Higher levels usually more tenure, but not always
    tenure_mean = 2 + (level - 1) * 1.5
    tenure = round(max(0.1, np.random.normal(tenure_mean, 2)), 2)
    
    # Hire Date from Tenure
    hire_date = datetime.now() - timedelta(days=int(tenure * 365))
    
    # Performance
    last_rating = round(min(5.0, max(1.0, np.random.normal(3.5, 0.8))), 1)
    
    # Attrition Risk Factors
    # Lower rating, lower pay ratio, longer hours -> higher attrition chance
    # This is "ground truth" logic we are baking in
    attrition_prob = 0.1 # Base
    if last_rating < 3.0: attrition_prob += 0.2
    if tenure < 1.0: attrition_prob += 0.1 # Early churn
    if tenure > 4.0 and level < 3: attrition_prob += 0.15 # Stagnation
    
    attrition = 1 if np.random.random() < attrition_prob else 0
    
    # Manager ID logic (simplified: Manager is someone with level > current)
    # For simplicity, assign random manager ID from MGR001 to MGR050
    manager_id = f"MGR{np.random.randint(1, 51):03d}"
    
    # Job Title construction
    titles = {
        'Engineering': ['Junior Engineer', 'Engineer', 'Senior Engineer', 'Staff Engineer', 'Principal Engineer', 'VP Engineering'],
        'Sales': ['Sales Rep', 'Account Executive', 'Senior AE', 'Sales Manager', 'Regional Director', 'VP Sales'],
        'Customer Success': ['CS Representative', 'CS Specialist', 'CS Manager', 'Senior CS Manager', 'Director of CS', 'VP CS'],
        'Product': ['Associate PM', 'Product Manager', 'Senior PM', 'Group PM', 'Director of Product', 'VP Product'],
        'Marketing': ['Marketing Coordinator', 'Marketing Specialist', 'Marketing Manager', 'Senior Marketing Manager', 'Director of Marketing', 'CMO'],
        'Finance': ['Financial Analyst', 'Senior Analyst', 'Finance Manager', 'Senior Finance Manager', 'Director of Finance', 'CFO'],
        'HR': ['HR Coordinator', 'HR Specialist', 'HR Business Partner', 'HR Manager', 'Director of HR', 'CHRO'],
        'Operations': ['Operations Associate', 'Operations Specialist', 'Operations Manager', 'Senior Ops Manager', 'Director of Operations', 'COO']
    }
    
    job_title = titles[dept][level-1]
    
    # Other Optional Fields
    compa_ratio = round(salary / (base * (1 + (level - 1) * 0.25)), 2)
    work_life_balance = round(np.random.uniform(1, 5), 1)
    
    # Interview Scores (Correlated with Performance generally)
    interview_score = round(min(5.0, max(1.0, last_rating + np.random.normal(0, 0.5))), 1)
    
    # Age Group
    if age < 30: age_group = '20-29'
    elif age < 40: age_group = '30-39'
    elif age < 50: age_group = '40-49'
    elif age < 60: age_group = '50-59'
    else: age_group = '60+'

    # Calculate Promotion metrics
    years_in_role = round(min(tenure, np.random.exponential(1.5)), 2)
    promotions = max(0, level - 1 - np.random.randint(0, 2)) # Rough estimate
    years_since_promo = round(min(tenure, np.random.exponential(2.0)), 1) if promotions > 0 else 0.0

    row = {
        'EmployeeID': emp_id,
        'Gender': gender,
        'Age': age,
        'Dept': dept,
        'JobTitle': job_title,
        'Tenure': tenure,
        'YearsInCurrentRole': years_in_role,
        'YearsSinceLastPromotion': years_since_promo,
        'Education': np.random.choice(['High School', 'Associate', 'Bachelor', 'Master', 'PhD'], p=[0.05, 0.1, 0.45, 0.3, 0.1]),
        'Location': city,
        'Country': country,
        'Salary': salary,
        'LastRating': last_rating,
        'ManagerID': manager_id,
        'Attrition': attrition, # 0 or 1
        'PerformanceText': generate_performance_text(last_rating, dept),
        'CompaRatio': compa_ratio,
        'HireSource': np.random.choice(HIRE_SOURCES),
        'HireDate': hire_date.strftime('%Y-%m-%d'),
        
        # Interview & Assessment Data
        'InterviewScore': interview_score,
        'InterviewScore_Technical': round(np.random.uniform(2.0, 5.0), 1),
        'InterviewScore_Cultural': round(np.random.uniform(2.0, 5.0), 1),
        'InterviewScore_Curiosity': round(np.random.uniform(2.0, 5.0), 1),
        'AssessmentScore': round(np.random.uniform(40, 100), 1),
        
        'PriorExperienceYears': round(max(0, age - 22 - tenure), 1),
        'JobLevel': level,
        'PromotionCount': promotions,
        'LastPromotionDate': (datetime.now() - timedelta(days=int(years_since_promo*365))).strftime('%Y-%m-%d') if promotions > 0 else '',
        'ManagerChangeCount': np.random.randint(0, 5),
        
        'Age_Group': age_group,
        
        # More dimensions for completeness
        'InterviewScore_Communication': round(np.random.uniform(2.0, 5.0), 1),
        'InterviewScore_Leadership': round(np.random.uniform(2.0, 5.0), 1),
        
        # Survey Data placeholders
        'eNPS_Score': np.random.randint(0, 11),
        'Pulse_Score': round(np.random.uniform(1, 5), 1),
        'ManagerSatisfaction': round(np.random.uniform(1, 5), 1),
        'WorkLifeBalance': work_life_balance,
        'CareerGrowthSatisfaction': round(np.random.uniform(1, 5), 1),
        
        # Onboarding (only for recent hires < 6 months)
        'Onboarding_30d': round(np.random.uniform(3, 5), 1) if tenure < 0.5 else '',
        'Onboarding_60d': round(np.random.uniform(3, 5), 1) if tenure > 0.1 and tenure < 0.5 else '',
        'Onboarding_90d': round(np.random.uniform(3, 5), 1) if tenure > 0.25 and tenure < 0.5 else ''
    }
    data.append(row)

df = pd.DataFrame(data)
df.to_csv('/Users/omoniyi/Downloads/PeopleOS/sample_hr_data.csv', index=False)
print(f"Generated {NUM_EMPLOYEES} employee records in sample_hr_data.csv")
