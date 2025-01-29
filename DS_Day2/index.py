from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

app = Flask(__name__)

data = [
    ["Red", "Light-Bodied", None, "Pinot Noir"],
    ["Red", "Full-Bodied", None, "Shiraz or Zinfandel"],
    ["White", None, "Dry", "Sauvignon Blanc"],
    ["White", None, "Sweet", "Gewurztraminer"],
    ["Red-Fruity", None, None, "Pinot Noir"],
    ["Red-Earthy", None, None, "Chianti"],
    ["White-Crisp", None, None, "Sauvignon Blanc"],
    ["White-Creamy", None, None, "Chardonnay"],
    ["Red-Spicy", None, None, "Shiraz or Zinfandel"],
    ["Red-Rich", None, None, "Cabernet Sauvignon"],
    ["White-Floral", None, None, "Gewurztraminer"],
    ["White-Citrus", None, None, "Riesling"],
    ["Red", None, None, "Pinot Noir"],
    ["Red", None, None, "Chianti"],
    ["White", None, None, "Sauvignon Blanc"],
    ["White", None, None, "Chardonnay"],
    ["Red", None, None, "Shiraz or Zinfandel"],
    ["Red", None, None, "Cabernet Sauvignon"],
    ["White", None, None, "Gewurztraminer"],
    ["White", None, None, "Riesling"],
    ["Red-Fruity", "Light-Bodied", None, "Pinot Noir"],
    ["Red-Fruity", "Full-Bodied", None, "Shiraz or Zinfandel"],
    ["Red-Earthy", "Light-Bodied", None, "Pinot Noir"],
    ["Red-Earthy", "Full-Bodied", None, "Cabernet Sauvignon"],
    ["White-Crisp", None, "Dry", "Sauvignon Blanc"],
    ["White-Crisp", None, "Sweet", "Pinot Noir"],
    ["White-Creamy", None, "Dry", "Pinot Noir"],
    ["White-Creamy", None, "Sweet", "Chardonnay"],
    ["Red-Spicy", "Light-Bodied", None, "Pinot Noir"],
    ["Red-Spicy", "Full-Bodied", None, "Shiraz or Zinfandel"],
    ["Red-Rich", "Light-Bodied", None, "Pinot Noir"],
    ["Red-Rich", "Full-Bodied", None, "Cabernet Sauvignon"],
    ["White-Floral", None, "Dry", "Pinot Noir"],
    ["White-Floral", None, "Sweet", "Gewurztraminer"],
    ["White-Citrus", None, "Dry", "Sauvignon Blanc"],
    ["White-Citrus", None, "Sweet", "Riesling"],
    ["Red-Fruity", None, "Sweet", "Pinot Noir"],
    ["Red-Fruity", None, "Dry", "Pinot Noir"],
    ["Red-Earthy", None, "Sweet", "Chianti"],
    ["Red-Earthy", None, "Dry", "Pinot Noir"]
]

# Encode categorical variables
le_preference = LabelEncoder()
le_body = LabelEncoder()
le_sweetness = LabelEncoder()
le_recommendation = LabelEncoder()

df = pd.DataFrame(data, columns=["Preference", "Body", "Sweetness", "Recommendation"])


df['Preference'] = le_preference.fit_transform(df['Preference'])
df['Body'] = le_body.fit_transform(df['Body'])
df['Sweetness'] = le_sweetness.fit_transform(df['Sweetness'])
df['Recommendation'] = le_recommendation.fit_transform(df['Recommendation'])

# Split the data into features and target
X = df[['Preference', 'Body', 'Sweetness']]
y = df['Recommendation']


# Train the decision tree model
dt_model = DecisionTreeClassifier()
dt_model.fit(X, y)

# Train the random forest model
rf_model = RandomForestClassifier()
rf_model.fit(X, y)

# Train the AdaBoost model
ada_model = AdaBoostClassifier()
ada_model.fit(X, y)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    preference = data.get('preference')
    body = data.get('body')
    sweetness = data.get('sweetness')

    # Encode the input
    preference_encoded = le_preference.transform([preference])[0]
    body_encoded = le_body.transform([body])[0]
    sweetness_encoded = le_sweetness.transform([sweetness])[0]

    # Make a prediction
    dt_prediction = dt_model.predict([[preference_encoded, body_encoded, sweetness_encoded]])
    dt_recommendation = le_recommendation.inverse_transform(dt_prediction)[0]

    rf_prediction = rf_model.predict([[preference_encoded, body_encoded, sweetness_encoded]])
    rf_recommendation = le_recommendation.inverse_transform(rf_prediction)[0]

    ada_prediction = ada_model.predict([[preference_encoded, body_encoded, sweetness_encoded]])
    ada_recommendation = le_recommendation.inverse_transform(ada_prediction)[0]

    recommendations = {
        'Decision Tree': dt_recommendation,
        'Random Forest': rf_recommendation,
        'AdaBoost': ada_recommendation
    }
    
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True)