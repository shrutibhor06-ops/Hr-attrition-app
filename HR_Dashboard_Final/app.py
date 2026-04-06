from flask import Flask, render_template, request, jsonify, send_file, session, redirect
import sqlite3
import pandas as pd
import joblib
import json
import os
import io
import numpy as np

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = 'super_secret_hr_key'

DB_PATH = "employees.db"
MODEL_PATH = "model.joblib"
FEATURES_PATH = "features.json"
METRICS_PATH = "model_metrics.json"

# Load the model and features configuration globally
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    
features_config = {}
if os.path.exists(FEATURES_PATH):
    with open(FEATURES_PATH, 'r') as f:
        features_config = json.load(f)
        
metrics_config = {}
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, 'r') as f:
        metrics_config = json.load(f)

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.before_request
def require_login():
    public_endpoints = ['login', 'dashboard', 'static']
    if request.endpoint not in public_endpoints and 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401

@app.route('/')
def dashboard():
    """Renders the main HR dashboard HTML."""
    return render_template('index.html')

@app.route('/api/me', methods=['GET'])
def me():
    if 'user' in session:
        return jsonify({"logged_in": True, "username": session['user']})
    return jsonify({"logged_in": False})

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username', '').lower()
    password = data.get('password', '')
    
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password)).fetchone()
    if user:
        session['user'] = username
        conn.execute("UPDATE users SET is_logged_in = 1 WHERE username = ?", (username,))
        conn.commit()
        conn.close()
        return jsonify({"success": True})
    conn.close()
    return jsonify({"error": "Invalid credentials"}), 401

@app.route('/api/logout', methods=['POST'])
def logout():
    if 'user' in session:
        conn = get_db_connection()
        conn.execute("UPDATE users SET is_logged_in = 0 WHERE username = ?", (session['user'],))
        conn.commit()
        conn.close()
        session.pop('user', None)
    return jsonify({"success": True})

@app.route('/api/users', methods=['GET'])
def get_users():
    """Returns admins and their login status for the Admin Panel."""
    conn = get_db_connection()
    users = conn.execute("SELECT username, is_logged_in FROM users").fetchall()
    conn.close()
    return jsonify([dict(u) for u in users])

@app.route('/api/models', methods=['GET'])
def get_models():
    """Returns the JSON metrics for model comparison and feature importances."""
    return jsonify(metrics_config)

@app.route('/api/history', methods=['GET'])
def get_history():
    """Returns the prediction history for the app."""
    conn = get_db_connection()
    history = conn.execute("SELECT * FROM prediction_history ORDER BY id DESC LIMIT 50").fetchall()
    conn.close()
    return jsonify([dict(h) for h in history])

@app.route('/api/features', methods=['GET'])
def get_features():
    """Returns the parsed feature schema for the frontend to dynamically build forms."""
    return jsonify(features_config)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Returns aggregated statistical data for dashboard charts."""
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM employees", conn)
    conn.close()

    total_employees = len(df)
    
    # Assume target column Attrition exists and is either 'Yes'/'No' or 1/0
    attrition_count = 0
    if 'Attrition' in df.columns:
        if pd.api.types.is_numeric_dtype(df['Attrition']):
            attrition_count = int(df['Attrition'].sum())
        else:
            attrition_count = len(df[df['Attrition'].astype(str).str.lower() == 'yes'])
            
    attrition_rate = (attrition_count / total_employees * 100) if total_employees > 0 else 0
    
    # Optional: aggregate by a useful categorical column like Department if it exists
    dept_stats = {}
    if 'Department' in df.columns:
        dept_counts = df['Department'].value_counts().to_dict()
        dept_stats = dept_counts
        
    stats = {
        "totalEmployees": int(total_employees),
        "attritionCount": int(attrition_count),
        "attritionRate": float(f"{attrition_rate:.2f}"),
        "departmentDistribution": dept_stats
    }
    return jsonify(stats)

@app.route('/api/employees', methods=['GET'])
def get_employees():
    """Returns employees list, with optional simple search filter."""
    search_query = request.args.get('search', '').lower()
    
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM employees", conn)
    conn.close()
    
    if search_query:
        # Filter across all string columns
        mask = df.astype(str).apply(lambda x: x.str.lower().str.contains(search_query)).any(axis=1)
        df = df[mask]
        
    return jsonify(df.to_dict(orient='records'))

@app.route('/api/export', methods=['GET'])
def export_employees():
    """Exports the currently filtered employee set to CSV."""
    search_query = request.args.get('search', '').lower()
    
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM employees", conn)
    conn.close()
    
    if search_query:
        mask = df.astype(str).apply(lambda x: x.str.lower().str.contains(search_query)).any(axis=1)
        df = df[mask]

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    mem = io.BytesIO()
    mem.write(csv_buffer.getvalue().encode('utf-8'))
    mem.seek(0)
    
    return send_file(
        mem,
        mimetype='text/csv',
        download_name='employees_export.csv',
        as_attachment=True
    )

@app.route('/api/predict', methods=['POST'])
def predict():
    """Receives dynamic employee data, transforms it, and returns the model prediction."""
    if not model:
        return jsonify({"error": "Model not loaded"}), 500
        
    data = request.json
    try:
        # Convert incoming JSON dict to a single-row DataFrame
        df_input = pd.DataFrame([data])
        
        # Predict
        prediction = model.predict(df_input)
        
        pred_val = prediction[0]
        if isinstance(pred_val, (int, float, np.integer, np.floating)):
            pred_val = 'High Risk' if pred_val == 1 else 'Low Risk'
        elif isinstance(pred_val, str):
            pred_val = 'High Risk' if pred_val.lower() == 'yes' else 'Low Risk'
            
        prob = []
        conf = 0.0
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df_input)[0]
            prob = [float(p) for p in probs]
            conf = max(prob) * 100
            
        # Log to history
        conn = get_db_connection()
        conn.execute('''
            INSERT INTO prediction_history (username, inputs, prediction, confidence)
            VALUES (?, ?, ?, ?)
        ''', (session.get('user', 'unknown'), json.dumps(data), pred_val, conf))
        conn.commit()
        conn.close()
            
        return jsonify({
            "prediction": pred_val,
            "probabilities": prob
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=8080)
