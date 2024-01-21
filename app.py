from flask import Flask, render_template
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from io import BytesIO
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)

# Mock data generation
def generate_mock_data():
    buildings = ['Building 1', 'Building 2', 'Building 3', 'Building 4', 'Building 5']
    df = pd.DataFrame()

    for b in buildings:
        days = np.arange('2022-01-01', '2022-12-31', dtype='datetime64[D]')
        usage = np.random.randint(50, 100, len(days))
        noise = np.random.normal(0, 5, len(days))
        usage += noise
        bldg_df = pd.DataFrame({'Date': days, 'Usage': usage}) 
        bldg_df['Building'] = b
        df = df.append(bldg_df)

    return df

# Train a simple linear regression model for each building
def train_models(df):
    models = {}
    for b in df['Building'].unique():
        X = df[df['Building'] == b]['Date'].values.reshape(-1, 1)
        y = df[df['Building'] == b]['Usage'].values
        models[b] = LinearRegression().fit(X, y)
    return models

# Generate a plot for a building
def create_plot(building, models):
    plt.figure()
    X_test = np.array([(pd.Timestamp('2022-12-31') + pd.DateOffset(days=i)).strftime('%Y-%m-%d') for i in range(14)])
    X_test = X_test.reshape(-1, 1)
    model = models[building]
    predictions = model.predict(X_test)

    plt.plot(X_test, predictions, label='Predicted Usage')
    plt.title(f'Building {building} Energy Usage Prediction')
    plt.xlabel('Date')
    plt.ylabel('Energy Usage')
    plt.legend()
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    encoded_plot = base64.b64encode(image_stream.getvalue()).decode('utf-8')
    plt.close()
    return encoded_plot

@app.route('/')
def index():
    df = generate_mock_data()
    models = train_models(df)
    building_plots = {i: create_plot(i, models) for i in range(1, 6)}
    return render_template('index.html', building_plots=building_plots)

if __name__ == '__main__':
    app.run(debug=True)
