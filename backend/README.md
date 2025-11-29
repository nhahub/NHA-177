# Fitness API Backend

Python Flask backend for calculating nutrition requirements and diet predictions.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python app.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### 1. Health Check
- **GET** `/api/health`
- Returns API status

### 2. Calculate Nutrients
- **POST** `/api/calculate-nutrients`
- Calculates daily nutrition requirements based on user input
- **Request Body:**
```json
{
    "age": 25,
    "gender": "Male",
    "weight": 75.5,
    "height": 1.75,
    "bmi": 24.7,
    "fatPercent": 15.0,
    "workoutFrequency": 4,
    "goalLabel": "Muscle Gain"
}
```
- **Response:**
```json
{
    "success": true,
    "data": {
        "bmr": 1750.5,
        "tdee": 2713.3,
        "adjusted_tdee": 3120.3,
        "daily_nutrients": {
            "calories": 3120,
            "protein": 166,
            "carbs": 390,
            "fats": 87
        },
        "macronutrient_breakdown": {
            "protein_percentage": 21.3,
            "carbs_percentage": 50.0,
            "fats_percentage": 25.1
        }
    }
}
```

### 3. Predict Diet Type
- **POST** `/api/predict-diet-type`
- Predicts diet type (Balanced, High-Protein, Low-Carb)
- **Request Body:** Same as calculate-nutrients
- **Response:**
```json
{
    "success": true,
    "dietType": "High-Protein"
}
```

## Calculation Methods

- **BMR**: Mifflin-St Jeor Equation
- **TDEE**: BMR × Activity Multiplier
- **Macronutrients**: Goal-based distribution
  - Muscle Gain: High protein (2.2g/kg)
  - Weight Loss: Moderate protein (2.0g/kg), lower carbs
  - Maintain: Balanced (1.6g/kg)

