import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load original data
df = pd.read_csv('/Users/omoniyi/Downloads/PeopleOS/sample_hr_data.csv')

# Add Age_Group
def get_age_group(age):
    if age < 30: return '20-29'
    if age < 40: return '30-39'
    if age < 50: return '40-49'
    if age < 60: return '50-59'
    return '60+'

df['Age_Group'] = df['Age'].apply(get_age_group)

# Add HireSource and InterviewScore if missing or populate them
if 'HireSource' not in df.columns or df['HireSource'].isna().all():
    sources = ['LinkedIn', 'Referral', 'JobBoard', 'Agency', 'Internal', 'Campus', 'Website']
    df['HireSource'] = np.random.choice(sources, size=len(df))

if 'InterviewScore' not in df.columns or df['InterviewScore'].isna().all():
    df['InterviewScore'] = np.round(np.random.uniform(2.5, 4.5, size=len(df)), 1)

# Ensure all dimensions are present
np.random.seed(42)
dimensions = ['Technical', 'Cultural', 'Curiosity', 'Communication', 'Leadership']
for dim in dimensions:
    col = f'InterviewScore_{dim}'
    df[col] = np.round(np.random.uniform(2.0, 5.0, size=len(df)), 1)

# Add recent hires to populate "New Hire Watch" (hires within last 6 months)
# Current mock date is Jan 2026 based on previous conversation
today = datetime(2026, 1, 10)
recent_hire_indices = np.random.choice(df.index, size=25, replace=False)
for idx in recent_hire_indices:
    # Random date in last 5 months
    days_ago = np.random.randint(5, 150)
    hire_date = today - timedelta(days=days_ago)
    df.loc[idx, 'HireDate'] = hire_date.strftime('%Y-%m-%d')
    # New hires shouldn't have promotions yet
    df.loc[idx, 'PromotionCount'] = 0
    df.loc[idx, 'YearsInCurrentRole'] = round(days_ago / 365, 2)
    df.loc[idx, 'Tenure'] = round(days_ago / 365, 2)

# Save updated data
df.to_csv('/Users/omoniyi/Downloads/PeopleOS/sample_hr_data.csv', index=False)
print("Updated sample_hr_data.csv with Quality of Hire data and 25 Recent Hires for Risk Watch")

# Create eNPS Sample Data (100 rows)
enps_data = []
depts = df['Dept'].unique()
for i in range(100):
    score = np.random.randint(0, 11)
    dept = np.random.choice(depts)
    tenure = np.random.uniform(0.1, 10.0)
    enps_data.append({
        'EmployeeID': f'EMP{1000+i}',
        'Score': score,
        'Department': dept,
        'Tenure': round(tenure, 1),
        'SurveyDate': '2024-01-15'
    })
pd.DataFrame(enps_data).to_csv('/Users/omoniyi/Downloads/PeopleOS/templates/sample_enps_data.csv', index=False)
print("Created templates/sample_enps_data.csv")

# Create Onboarding Sample Data (150 rows)
onboarding_data = []
for i in range(150):
    prep = np.random.uniform(3.0, 5.0)
    first_week = np.random.uniform(3.0, 5.0)
    manager = np.random.uniform(3.0, 5.0)
    culture = np.random.uniform(3.0, 5.0)
    overall = (prep + first_week + manager + culture) / 4
    onboarding_data.append({
        'EmployeeID': f'EMP{2000+i}',
        'Prep_Score': round(prep, 1),
        'FirstWeek_Score': round(first_week, 1),
        'Manager_Support': round(manager, 1),
        'Culture_Fit': round(culture, 1),
        'Overall_Satisfaction': round(overall, 1),
        'Comments': 'Smooth process' if overall > 4 else 'Needs improvement',
        'HireDate': '2023-11-01'
    })
pd.DataFrame(onboarding_data).to_csv('/Users/omoniyi/Downloads/PeopleOS/templates/sample_onboarding_data.csv', index=False)
print("Created templates/sample_onboarding_data.csv")
