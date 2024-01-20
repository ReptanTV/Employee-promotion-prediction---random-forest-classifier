import numpy as np
from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from scipy import stats
#from scipy.stats import boxcox
from scipy.stats import yeojohnson

df = pd.read_csv('C:\\Users\\sasta\\OneDrive\\Desktop\\datasets\\employee_promotion.csv', sep=';')

app = Flask(__name__, static_url_path='/static', static_folder='static')

# Load the trained model
model = joblib.load('EmployeePromo.joblib')

# Preprocessing functions (you can also move these to a separate module)
def preprocess_data(df):
    prv_yr_r = 'previous_year_rating'
    imputer = SimpleImputer(strategy='median')
    df[prv_yr_r] = imputer.fit_transform(df[[prv_yr_r]])

    avg_tr_s = 'avg_training_score'
    imputer = SimpleImputer(strategy='median')
    df[avg_tr_s] = imputer.fit_transform(df[[avg_tr_s]])

    age = 'age'
    imputer = SimpleImputer(strategy='median')
    df[age] = imputer.fit_transform(df[[age]])

    trainings = 'no_of_trainings'
    imputer = SimpleImputer(strategy='median')
    df[trainings] = imputer.fit_transform(df[[trainings]])
    transformed_data, lmbda = yeojohnson(df['previous_year_rating'])
    
    df['previous_year_rating'] = transformed_data

    z_scores = np.abs(stats.zscore(df['previous_year_rating']))
    thr = 3
    out_indices = np.where(z_scores > thr)

    transformed_data, lmbda = yeojohnson(df['avg_training_score'])
    df['avg_training_score'] = transformed_data

    z_scores = np.abs(stats.zscore(df['avg_training_score']))
    thr = 3
    out_indices = np.where(z_scores > thr)

    transformed_data, lmbda = yeojohnson(df['no_of_trainings'])
    df['no_of_trainings'] = transformed_data

    z_scores = np.abs(stats.zscore(df['no_of_trainings']))
    thr = 3
    out_indices = np.where(z_scores > thr)

    transformed_data, lmbda = yeojohnson(df['age'])
    df['age'] = transformed_data

    z_scores = np.abs(stats.zscore(df['age']))
    thr = 3
    out_indices = np.where(z_scores > thr)


x = df.drop(columns=['is_promoted', 'region', 'gender', 'recruitment_channel', 'department', 'education'])
preprocess_data(x)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    not_found= None
    emp_info = None
    if request.method == 'POST':
        employee_id = int(request.form.get('employee_id'))

        # Fetch features for the provided employee ID from the loaded data
        employee_row = x[x['employee_id'] == employee_id]

        print("Employee ID:", employee_id)
        print("Employee Row:", employee_row)

        if not employee_row.empty:
            # Make predictions using the loaded model
            prediction = model.predict(employee_row)[0]
            
            emp_info = df[df['employee_id']== employee_id].iloc[0]
        else:
            not_found = f"employee with ID {employee_id} not found!"
            
        print("Prediction:", prediction)

    return render_template('home.html', prediction=prediction, not_found=not_found,emp_info=emp_info)
    
if __name__ == '__main__':
    app.run(debug=True)
