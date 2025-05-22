import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import io

@st.cache_resource(show_spinner=True)
def load_scalers_and_model():
    scaler_X = joblib.load('scaler_X.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
    model = load_model('traffic_model.keras')
    return scaler_X, scaler_y, model

@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_csv('Traffic_Volume_Counts_20250108.csv')
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    return df

# Load models and data once
scaler_X, scaler_y, model = load_scalers_and_model()
df = load_data()

time_slot_cols = [col for col in df.columns if '-' in col and ('AM' in col or 'PM' in col)]

st.title("🚦 Traffic Volume Prediction")

date = st.date_input("📅 Date")
direction = st.selectbox("🧭 Direction", sorted(df['Direction'].dropna().unique()))
roadway = st.selectbox("🛣️ Roadway Name", sorted(df['Roadway Name'].dropna().unique()))
from_loc = st.selectbox("🔜 From", sorted(df['From'].dropna().unique()))
to_loc = st.selectbox("🔚 To", sorted(df['To'].dropna().unique()))
selected_slot = st.selectbox("⏰ Time Slot", sorted(time_slot_cols))

if st.button("📊 Predict 1-Hour Ahead and Show Graph"):
    predictions = {}
    all_columns = list(scaler_X.feature_names_in_)

    for time_slot in time_slot_cols:
        input_data = pd.DataFrame(columns=all_columns)
        input_data.loc[0] = 0  # fill with zeros

        # Basic date info
        input_data.at[0, 'Year'] = date.year
        input_data.at[0, 'Month'] = date.month
        input_data.at[0, 'Day'] = date.day

        # One-hot encoded fields
        one_hot_fields = {
            'Direction': direction,
            'Roadway Name': roadway,
            'From': from_loc,
            'To': to_loc,
            'Time Slot': time_slot
        }

        for field, value in one_hot_fields.items():
            col_name = f"{field}_{value}"
            if col_name in input_data.columns:
                input_data.at[0, col_name] = 1

        try:
            X_scaled = scaler_X.transform(input_data)
            X_scaled = X_scaled.reshape((1, 1, X_scaled.shape[1]))
            y_pred_scaled = model.predict(X_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            predictions[time_slot] = y_pred[0][0]
        except Exception as e:
            st.warning(f"Prediction failed for {time_slot}: {e}")
            predictions[time_slot] = np.nan

    # Show result for selected slot
    if selected_slot in predictions and not np.isnan(predictions[selected_slot]):
        st.success(f"📈 Predicted Traffic Volume at {selected_slot}: {int(predictions[selected_slot])} vehicles")
    else:
        st.error("Prediction not available for the selected time slot.")

    # DataFrame for plotting and download
    pred_df = pd.DataFrame({
        'Time Slot': list(predictions.keys()),
        'Predicted Traffic Volume': list(predictions.values())
    })

    st.subheader("📊 Predicted Traffic Volume for All Time Slots")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pred_df['Time Slot'], pred_df['Predicted Traffic Volume'], marker='o', label='Predicted Volume', color='skyblue')

    if selected_slot in predictions and not np.isnan(predictions[selected_slot]):
        idx = list(predictions.keys()).index(selected_slot)
        ax.plot(pred_df['Time Slot'][idx], pred_df['Predicted Traffic Volume'][idx], marker='o', color='red', label='Selected Slot')

    ax.set_xlabel("Time Slot")
    ax.set_ylabel("Traffic Volume")
    ax.set_title("Predicted Traffic Volume per Time Slot")
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    st.pyplot(fig)

    # Save plot to BytesIO for download
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    st.download_button(
        label="📥 Download Graph as PNG",
        data=buf,
        file_name=f"traffic_prediction_{date}.png",
        mime="image/png"
    )

    # Download CSV of predictions
    csv = pred_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📄 Download Predictions as CSV",
        data=csv,
        file_name=f"traffic_predictions_{date}.csv",
        mime='text/csv'
    )
