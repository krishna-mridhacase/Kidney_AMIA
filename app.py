import os
from flask import Flask, render_template, request, send_file
import pickle
import numpy as np
import pandas as pd
import shap
import lime
from lime import lime_tabular
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-GUI environments
import matplotlib.pyplot as plt



app = Flask(__name__)

# Load the model
model = pickle.load(open('models/kidney.pkl', 'rb'))

# Load the training data for SHAP KernelExplainer (example: replace with actual training data)
X_train = pd.read_csv('notebooks/X_train.csv')  # Replace with your actual training data path
X_train = X_train.drop(columns=['Unnamed: 0'])


X_test = pd.read_csv('notebooks/X_test.csv')  # Replace with your actual training data path
X_test = X_test.drop(columns=['Unnamed: 0'])

# Initialize SHAP explainers
tree_explainer = shap.TreeExplainer(model)
kernel_explainer = shap.KernelExplainer(model.predict_proba, X_train)

# Feature names (in the same order as the DataFrame columns used for training)
feature_names = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells',
                 'pus_cell', 'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea',
                 'serum_creatinine', 'sodium', 'potassium', 'haemoglobin', 'packed_cell_volume',
                 'white_blood_cell_count', 'red_blood_cell_count', 'hypertension', 'diabetes_mellitus',
                 'coronary_artery_disease', 'appetite', 'peda_edema', 'aanemia']

# Ensure static directory exists
if not os.path.exists('static'):
    os.makedirs('static')


class_names=['Healthy', 'CKD']


def predict(values):
    try:
        if len(values) == 24:
            # Convert the input values to a DataFrame with the appropriate feature names
            input_df = pd.DataFrame([values], columns=feature_names)

            pred = model.predict(input_df)[0]
            # print("The output is",pred)
           
            shap_values = tree_explainer.shap_values(input_df)
            # print(f"SHAP values: {shap_values}")  # Log the SHAP values for debugging
            return pred, shap_values, input_df
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None, None

def plot_shap_values(shap_impact):
    # Create a bar plot of the top 5 SHAP values
    features = list(shap_impact.keys())
    values = list(shap_impact.values())
    
    plt.figure(figsize=(8, 4))  # Adjust the figure size to fit the page better
    plt.barh(features, values, color='skyblue')
    plt.xlabel('SHAP Value')
    plt.title('Top 5 Feature SHAP Values')
    plt.gca().invert_yaxis()  # To display the highest value on top
    plt.tight_layout()  # Adjust layout to fit everything nicely
    plt.savefig('static/shap_plot.png')
    plt.close()

def plot_force_plot(user_input_df):
    try:
        # Initialize the KernelExplainer
        explainer = shap.KernelExplainer(model.predict_proba, X_train)
        shap_values = explainer.shap_values(user_input_df.iloc[0, :])
        
        # Ensure shap_values is properly indexed
        if len(shap_values) == 2:  # Binary classification case
            shap_values = shap_values[1]  # Focus on the positive class

        # Generate and display the force plot
        shap.initjs()
        plt.figure(figsize=(10, 5))
        
        # Ensure feature names are consistent
        if 'feature_names' not in user_input_df.columns:
            user_input_df.columns = X_train.columns
        
        shap.force_plot(explainer.expected_value[1], shap_values, user_input_df.iloc[0, :],  matplotlib=True)

        force_plot_path = os.path.join('static', 'force_plot.png')
        plt.savefig(force_plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Force plot saved at {force_plot_path}")
    except Exception as e:
        print(f"Error displaying force plot: {e}")


def plot_lime_explanation(input_df, class_names, file_path='static/lime_explanation.html'):
    # Initialize LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_train.columns,
        class_names=class_names,
        mode='classification'
    )
    
    # Explain a single instance
    exp = explainer.explain_instance(
        data_row=input_df.iloc[0],
        predict_fn=model.predict_proba
    )
    
    # Save the explanation to an HTML file
    with open(file_path, 'w') as f:
        f.write(exp.as_html())
    print(f"Explanation saved as {file_path}")

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/predict", methods=['POST', 'GET'])
def predictPage():
    high_values = {}
    shap_impact = {}
    pred = None
    user_input_dict = {}
    try:
        if request.method == 'POST':
            # Capture user input as a dictionary
            user_input_dict = request.form.to_dict()

            # Convert the input to the appropriate types (e.g., int, float)
            for key, value in user_input_dict.items():
                try:
                    user_input_dict[key] = int(value)
                except ValueError:
                    user_input_dict[key] = float(value)

            # Convert the dictionary to a list in the correct order of features
            to_predict_list = [user_input_dict[feature] for feature in feature_names]

            # Convert to a DataFrame for prediction
            input_df = pd.DataFrame([to_predict_list], columns=feature_names)

            # Make a prediction using the model
            pred, shap_values, input_df = predict(to_predict_list)

            # Process SHAP values
            if shap_values is not None and len(shap_values) > 0:
                if pred == 1:
                    # Analyze SHAP values for Class 1
                    shap_impact = {feature_names[i]: shap_values[1][0][i] for i in range(len(feature_names))}
                else:
                    # Analyze SHAP values for Class 0
                    shap_impact = {feature_names[i]: shap_values[0][0][i] for i in range(len(feature_names))}

                top_features = sorted(shap_impact.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                shap_impact = dict(top_features)
                print("shap impact:", shap_impact)

                # Plot the top 5 SHAP values
                plot_shap_values(shap_impact)

                # Plot the force plot
                plot_force_plot(input_df)

                if pred is not None:
                    plot_lime_explanation(input_df, class_names)

    except Exception as e:
        print(f"Error during prediction: {e}")
        message = "Please enter valid data"
        return render_template("home.html", message=message)

    # Pass the user_input_dict to the template along with other data
    return render_template('predict.html', pred=pred, high_values=high_values, shap_impact=shap_impact, user_input=user_input_dict)



if __name__ == '__main__':
    app.run(debug=True)
