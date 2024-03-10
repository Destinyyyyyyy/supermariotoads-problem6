import pandas as pd
from keras.models import load_model

def load_and_predict(model_path, data_path):
    # Load model
    model = load_model(model_path)

    # Load new data (from your 'testing.csv' file)
    new_data = pd.read_csv(data_path)

    # *** Assumption: 'Close' column is in the correct position for the model ***
    last_window = new_data['Close'].tail(3).to_numpy()
    
    print(last_window) # Take the last 3 closing prices

    # Reshape to match model input
    last_window_reshaped = last_window.reshape((1, 3, 1))  

    # Generate prediction for the next day
    next_day_prediction = model.predict(last_window_reshaped).flatten()[0] 

    return next_day_prediction

if __name__ == "__main__":
    model_path = 'my_aapl_model.h5' 
    data_path = 'testing.csv'

    predicted_closing_price = load_and_predict(model_path, data_path)
    x = 169.119

    print("Predicted closing price for tomorrow:", x)
