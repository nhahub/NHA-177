from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pickle
import math
# Ensure flask_cors is installed: pip install flask-cors

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'diet_model.pkl')
diet_model = None

def load_diet_model():
    global diet_model
    try:
        with open(MODEL_PATH, 'rb') as f:
            diet_model = pickle.load(f)
        print(f"[INFO] Loaded diet model: {MODEL_PATH}")
    except Exception as e:
        diet_model = None
        print(f"[WARN] Failed to load diet model: {e}")

load_diet_model()

def _safe_float(val, default=0.0):
    try:
        x = float(val)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default

def _encode_gender(g):
    return 1.0 if str(g).strip().lower() == 'male' else 0.0

def _encode_goal(goal):
    g = str(goal).strip().lower()
    return (
        1.0 if g == 'maintain' else 0.0,
        1.0 if g == 'muscle gain' else 0.0,
        1.0 if g == 'weight loss' else 0.0,
    )

def build_features(data: dict):
    age = _safe_float(data.get('age', 0))
    gender = _encode_gender(data.get('gender', 'Female'))
    weight = _safe_float(data.get('weight', 0))
    height = _safe_float(data.get('height', 0))
    bmi = _safe_float(data.get('bmi', 0))
    fat = _safe_float(data.get('fatPercent', 0))
    freq = _safe_float(data.get('workoutFrequency', 0))
    maintain, muscle, loss = _encode_goal(data.get('goalLabel', 'Maintain'))
    return [age, gender, weight, height, bmi, fat, freq, maintain, muscle, loss]

def normalize_diet_type(pred):
    try:
        v = pred
        if isinstance(v, (list, tuple)) and len(v) > 0:
            v = v[0]
        s = str(v).strip()
        m = {
            'balanced': 'Balanced',
            'high-protein': 'High-Protein',
            'high protein': 'High-Protein',
            'low-carb': 'Low-Carb',
            'low carb': 'Low-Carb'
        }
        if s.replace('_', '-').lower() in m:
            return m[s.replace('_', '-').lower()]
        try:
            n = int(float(s))
            return ['Balanced', 'High-Protein', 'Low-Carb'][n] if n in (0,1,2) else 'Balanced'
        except Exception:
            return 'Balanced'
    except Exception:
        return 'Balanced'

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "API is running"}), 200

@app.route('/api/calculate-nutrients', methods=['POST'])
def calculate_nutrients():
    """
    Calculate daily nutrition requirements based on user input
    Expected JSON payload:
    {
        "age": int,
        "gender": "Male" | "Female",
        "weight": float (kg),
        "height": float (m),
        "bmi": float,
        "fatPercent": float,
        "workoutFrequency": int (0-7),
        "goalLabel": "Maintain" | "Muscle Gain" | "Weight Loss"
    }
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['age', 'gender', 'weight', 'height', 'bmi', 'fatPercent', 'workoutFrequency', 'goalLabel']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Extract and validate data
        age = int(data['age'])
        gender = data['gender']
        weight = float(data['weight'])
        height = float(data['height'])
        bmi = float(data['bmi'])
        fat_percent = float(data['fatPercent'])
        workout_frequency = int(data['workoutFrequency'])
        goal_label = data['goalLabel']
        
        # Validate gender
        if gender not in ['Male', 'Female']:
            return jsonify({"error": "Gender must be 'Male' or 'Female'"}), 400
        
        # Validate goal
        if goal_label not in ['Maintain', 'Muscle Gain', 'Weight Loss']:
            return jsonify({"error": "Goal must be 'Maintain', 'Muscle Gain', or 'Weight Loss'"}), 400
        
        # Validate workout frequency
        if workout_frequency < 0 or workout_frequency > 7:
            return jsonify({"error": "Workout frequency must be between 0 and 7"}), 400
        
        # Calculate BMR using Mifflin-St Jeor Equation
        bmr = calculate_bmr(age, gender, weight, height)
        
        # Calculate TDEE based on activity level
        tdee = calculate_tdee(bmr, workout_frequency)
        
        # Adjust TDEE based on goal
        adjusted_tdee = adjust_tdee_for_goal(tdee, goal_label)
        
        # Calculate macronutrients
        macros = calculate_macronutrients(weight, adjusted_tdee, goal_label)
        
        # Prepare response
        response = {
            "success": True,
            "data": {
                "bmr": round(bmr, 2),
                "tdee": round(tdee, 2),
                "adjusted_tdee": round(adjusted_tdee, 2),
                "daily_nutrients": {
                    "calories": round(adjusted_tdee),
                    "protein": round(macros['protein']),
                    "carbs": round(macros['carbs']),
                    "fats": round(macros['fats'])
                },
                "macronutrient_breakdown": {
                    "protein_percentage": round((macros['protein'] * 4 / adjusted_tdee) * 100, 1),
                    "carbs_percentage": round((macros['carbs'] * 4 / adjusted_tdee) * 100, 1),
                    "fats_percentage": round((macros['fats'] * 9 / adjusted_tdee) * 100, 1)
                }
            }
        }
        
        return jsonify(response), 200
        
    except ValueError as e:
        return jsonify({"error": f"Invalid input value: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/predict-diet-type', methods=['POST'])
def predict_diet_type():
    """
    Predict diet type based on user input
    Expected JSON payload:
    {
        "age": int,
        "gender": "Male" | "Female",
        "weight": float,
        "height": float,
        "bmi": float,
        "fatPercent": float,
        "workoutFrequency": int,
        "goalLabel": "Maintain" | "Muscle Gain" | "Weight Loss"
    }
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        if 'goalLabel' not in data or 'workoutFrequency' not in data or 'bmi' not in data:
            return jsonify({"error": "Missing required fields"}), 400
        
        goal = data['goalLabel']
        workout_freq = int(data['workoutFrequency'])
        bmi = float(data['bmi'])
        
        # Simple rule-based prediction (replace with ML model later)
        diet_type = predict_diet_type_logic(goal, workout_freq, bmi)
        
        response = {
            "success": True,
            "dietType": diet_type
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/predict-diet-ml', methods=['POST'])
def predict_diet_ml():
    """Predict diet type using the ML model; accepts the same payload as /api/predict-diet-type"""
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid or missing JSON payload"}), 400
        features = build_features(data)
        goal = data.get('goalLabel', 'Maintain')
        workout_freq = int(_safe_float(data.get('workoutFrequency', 0)))
        bmi = _safe_float(data.get('bmi', 0))
        if diet_model is None:
            diet_type = predict_diet_type_logic(goal, workout_freq, bmi)
            return jsonify({"success": True, "dietType": diet_type}), 200
        try:
            if hasattr(diet_model, 'predict'):
                raw = diet_model.predict([features])
            else:
                raise RuntimeError('Model does not implement predict()')
            diet_type = normalize_diet_type(raw)
            return jsonify({"success": True, "dietType": diet_type}), 200
        except Exception:
            diet_type = predict_diet_type_logic(goal, workout_freq, bmi)
            return jsonify({"success": True, "dietType": diet_type}), 200
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

def calculate_bmr(age, gender, weight, height):
    """
    Calculate Basal Metabolic Rate using Mifflin-St Jeor Equation
    BMR = 10 * weight(kg) + 6.25 * height(cm) - 5 * age + s
    where s = +5 for males, -161 for females
    """
    height_cm = height * 100
    if gender == 'Male':
        bmr = 10 * weight + 6.25 * height_cm - 5 * age + 5
    else:  # Female
        bmr = 10 * weight + 6.25 * height_cm - 5 * age - 161
    
    return bmr

def calculate_tdee(bmr, workout_frequency):
    """
    Calculate Total Daily Energy Expenditure based on activity level
    Updated activity multipliers by workout frequency:
    - 0-1 days: 1.2 (Sedentary)
    - 2-3 days: 1.375 (Light exercise)
    - 4-5 days: 1.55 (Moderate exercise)
    - 6-7 days: 1.65 (Heavy exercise)
    """
    # Determine activity level based on workout frequency ranges
    if workout_frequency <= 1:
        multiplier = 1.2      # 0-1 days: Sedentary
    elif workout_frequency <= 3:
        multiplier = 1.375    # 2-3 days: Light exercise
    elif workout_frequency <= 5:
        multiplier = 1.55     # 4-5 days: Moderate exercise
    else:  # 6-7 days
        multiplier = 1.65     # 6-7 days: Heavy exercise
    
    return bmr * multiplier

def adjust_tdee_for_goal(tdee, goal_label):
    """
    Adjust TDEE based on fitness goal
    - Weight Loss: 15% deficit
    - Muscle Gain: 15% surplus
    - Maintain: No adjustment
    """
    if goal_label == 'Weight Loss':
        return tdee * 0.85  # 15% calorie deficit
    elif goal_label == 'Muscle Gain':
        return tdee * 1.15  # 15% calorie surplus
    else:  # Maintain
        return tdee

def calculate_macronutrients(weight, calories, goal_label):
    """
    Calculate daily macronutrient requirements based on goal
    Updated protein factors:
    - Weight Loss: 1.8 g/kg
    - Muscle Gain: 2.0 g/kg
    - Maintain: 1.6 g/kg
    
    Fat percentage:
    - Weight Loss: 25% of total calories
    - Muscle Gain: 25% of total calories
    - Maintain: 30% of total calories
    
    Carbs: Remaining calories after Protein and Fat, divided by 4 to get grams
    
    Returns: dict with protein, carbs, and fats in grams
    """
    if goal_label == 'Muscle Gain':
        # Updated: 2.0g per kg body weight (changed from 2.2g)
        protein_grams = weight * 2.0
        # 25% of calories from fats
        fat_calories = calories * 0.25
        fat_grams = fat_calories / 9
        # Calculate remaining calories for carbs
        protein_calories = protein_grams * 4
        remaining_calories = calories - protein_calories - fat_calories
        # Carbs: remaining calories divided by 4
        carb_grams = remaining_calories / 4
        
    elif goal_label == 'Weight Loss':
        # Updated: 1.8g per kg body weight (changed from 2.0g)
        protein_grams = weight * 1.8
        # Updated: 25% of calories from fats (changed from 30%)
        fat_calories = calories * 0.25
        fat_grams = fat_calories / 9
        # Calculate remaining calories for carbs
        protein_calories = protein_grams * 4
        remaining_calories = calories - protein_calories - fat_calories
        # Carbs: remaining calories divided by 4
        carb_grams = remaining_calories / 4
        
    else:  # Maintain / Balanced
        # 1.6g per kg body weight (unchanged)
        protein_grams = weight * 1.6
        # 30% of calories from fats (unchanged)
        fat_calories = calories * 0.30
        fat_grams = fat_calories / 9
        # Calculate remaining calories for carbs
        protein_calories = protein_grams * 4
        remaining_calories = calories - protein_calories - fat_calories
        # Carbs: remaining calories divided by 4
        carb_grams = remaining_calories / 4
    
    return {
        'protein': max(protein_grams, 0),
        'carbs': max(carb_grams, 0),
        'fats': max(fat_grams, 0)
    }

def predict_diet_type_logic(goal, workout_frequency, bmi):
    """
    Predict diet type based on simple rules
    (Replace with ML model prediction later)
    """
    if goal == 'Muscle Gain' or workout_frequency >= 4:
        return 'High-Protein'
    elif goal == 'Weight Loss' or (bmi > 25 and bmi < 30):
        return 'Low-Carb'
    else:
        return 'Balanced'

if __name__ == '__main__':
    print("Starting Fitness API server...")
    print("API endpoints:")
    print("  - GET  /api/health")
    print("  - POST /api/calculate-nutrients")
    print("  - POST /api/predict-diet-type")
    print("  - POST /api/predict-diet-ml")
    app.run(debug=False, host='0.0.0.0', port=5000)

