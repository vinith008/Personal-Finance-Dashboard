from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import json, os, hashlib, warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'fintrack_secret_2024'

# ─── In-memory user store ───────────────────────────────────────────────────
USERS = {}  # {username: {password_hash, name, email, transactions}}

def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

# ─── Load & train ML models ─────────────────────────────────────────────────
df = pd.read_csv('data/dataset.csv')
df.dropna(inplace=True)

le_scenario  = LabelEncoder()
le_income    = LabelEncoder()
le_stress    = LabelEncoder()
le_flow      = LabelEncoder()
le_category  = LabelEncoder()

df['financial_scenario_enc'] = le_scenario.fit_transform(df['financial_scenario'])
df['income_type_enc']        = le_income.fit_transform(df['income_type'])
df['financial_stress_enc']   = le_stress.fit_transform(df['financial_stress_level'])
df['cash_flow_enc']          = le_flow.fit_transform(df['cash_flow_status'])
df['category_enc']           = le_category.fit_transform(df['category'])

FEATURES = ['monthly_income','monthly_expense_total','savings_rate',
            'credit_score','debt_to_income_ratio','loan_payment',
            'investment_amount','subscription_services','emergency_fund',
            'transaction_count','discretionary_spending','essential_spending',
            'rent_or_mortgage','income_type_enc']

# Model 1: Savings goal predictor (classifier)
X_cls = df[FEATURES]
y_cls = df['savings_goal_met']
X_tr, X_te, y_tr, y_te = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_tr, y_tr)

# Model 2: Financial advice score regressor
y_reg = df['financial_advice_score']
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_cls, y_reg, test_size=0.2, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_tr2, y_tr2)

# Model 3: Stress predictor
y_stress = df['financial_stress_enc']
X_tr3, X_te3, y_tr3, y_te3 = train_test_split(X_cls, y_stress, test_size=0.2, random_state=42)
stress_model = RandomForestClassifier(n_estimators=80, random_state=42)
stress_model.fit(X_tr3, y_tr3)

# ─── Dataset stats for demo ──────────────────────────────────────────────────
def get_dataset_stats():
    return {
        'avg_income':     round(df['monthly_income'].mean(), 2),
        'avg_expense':    round(df['monthly_expense_total'].mean(), 2),
        'avg_savings':    round(df['actual_savings'].mean(), 2),
        'avg_credit':     round(df['credit_score'].mean(), 2),
        'positive_flow':  round((df['cash_flow_status']=='Positive').mean()*100, 1),
        'goal_met_pct':   round(df['savings_goal_met'].mean()*100, 1),
        'top_categories': df['category'].value_counts().head(5).to_dict(),
        'scenarios':      df['financial_scenario'].value_counts().to_dict(),
        'income_types':   df['income_type'].value_counts().to_dict(),
        'stress_dist':    df['financial_stress_level'].value_counts().to_dict(),
    }

# ─── Routes ──────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username','').strip()
    if username in USERS:
        return jsonify({'ok': False, 'msg': 'Username already exists'})
    USERS[username] = {
        'password': hash_pw(data['password']),
        'name': data.get('name', username),
        'email': data.get('email', ''),
        'monthly_income': float(data.get('monthly_income', 4000)),
        'monthly_expense': float(data.get('monthly_expense', 2500)),
        'savings_goal': float(data.get('savings_goal', 500)),
        'credit_score': int(data.get('credit_score', 700)),
        'transactions': []
    }
    session['user'] = username
    return jsonify({'ok': True})

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    user = USERS.get(data['username'])
    if user and user['password'] == hash_pw(data['password']):
        session['user'] = data['username']
        return jsonify({'ok': True})
    return jsonify({'ok': False, 'msg': 'Invalid credentials'})

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('index'))
    return render_template('dashboard.html', username=session['user'])

# ─── API ─────────────────────────────────────────────────────────────────────
@app.route('/api/profile')
def api_profile():
    if 'user' not in session: return jsonify({'error': 'unauth'}), 401
    u = USERS[session['user']]
    return jsonify({k: v for k, v in u.items() if k != 'password'})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'user' not in session: return jsonify({'error': 'unauth'}), 401
    d = request.json
    income_enc = le_income.transform([d.get('income_type','Salary')])[0] if d.get('income_type','Salary') in le_income.classes_ else 0
    feat = [[
        float(d.get('monthly_income', 4000)),
        float(d.get('monthly_expense', 2500)),
        float(d.get('savings_rate', 0.2)),
        float(d.get('credit_score', 700)),
        float(d.get('debt_to_income', 0.3)),
        float(d.get('loan_payment', 300)),
        float(d.get('investment_amount', 200)),
        int(d.get('subscription_services', 4)),
        float(d.get('emergency_fund', 1000)),
        int(d.get('transaction_count', 50)),
        float(d.get('discretionary_spending', 400)),
        float(d.get('essential_spending', 1500)),
        float(d.get('rent_or_mortgage', 1000)),
        income_enc
    ]]
    goal_prob   = rf_model.predict_proba(feat)[0][1] * 100
    advice_score = gb_model.predict(feat)[0]
    stress_enc  = stress_model.predict(feat)[0]
    stress_label = le_stress.inverse_transform([stress_enc])[0]

    # Generate advice text
    income = float(d.get('monthly_income', 4000))
    expense = float(d.get('monthly_expense', 2500))
    tips = []
    if expense / income > 0.7: tips.append("🔴 Your expenses exceed 70% of income — review discretionary spending.")
    if float(d.get('emergency_fund', 0)) < income * 3: tips.append("🟡 Emergency fund is low — aim for 3–6 months of expenses.")
    if float(d.get('investment_amount', 0)) < income * 0.1: tips.append("💡 Consider investing at least 10% of monthly income.")
    if float(d.get('debt_to_income', 0)) > 0.4: tips.append("⚠️ High debt-to-income ratio — focus on debt reduction.")
    if not tips: tips.append("✅ Your finances look healthy! Keep maintaining your budget.")

    return jsonify({
        'goal_probability': round(goal_prob, 1),
        'advice_score': round(advice_score, 1),
        'stress_level': stress_label,
        'tips': tips
    })

@app.route('/api/add_transaction', methods=['POST'])
def api_add_transaction():
    if 'user' not in session: return jsonify({'error': 'unauth'}), 401
    d = request.json
    USERS[session['user']]['transactions'].append(d)
    return jsonify({'ok': True})

@app.route('/api/transactions')
def api_transactions():
    if 'user' not in session: return jsonify({'error': 'unauth'}), 401
    return jsonify(USERS[session['user']]['transactions'])

@app.route('/api/stats')
def api_stats():
    return jsonify(get_dataset_stats())

@app.route('/api/chart/spending_by_category')
def chart_category():
    data = df.groupby('category')['monthly_expense_total'].mean().round(2).to_dict()
    return jsonify(data)

@app.route('/api/chart/income_vs_expense')
def chart_income_vs_expense():
    monthly = df.groupby(df['date'].str[:7])[['monthly_income','monthly_expense_total']].mean().tail(12)
    return jsonify({'labels': list(monthly.index),
                    'income': list(monthly['monthly_income'].round(2)),
                    'expense': list(monthly['monthly_expense_total'].round(2))})

@app.route('/api/chart/savings_trend')
def chart_savings():
    data = df.groupby(df['date'].str[:7])['actual_savings'].mean().tail(12).round(2)
    return jsonify({'labels': list(data.index), 'values': list(data)})

@app.route('/api/chart/stress_dist')
def chart_stress():
    data = df['financial_stress_level'].value_counts().to_dict()
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, port=5050)
