import os
import pandas as pd
from nixtla import NixtlaClient
import streamlit as st

csv_path = "csv files/"
filename = "all_campaign_daily_data_2023-2024.csv"

st.title("TimeGPT App")
st.write("This is a simple app to generate time series forecasts using TimeGPT model.")


nixtla_api_key = st.text_input("Enter your Nixtla API key", type="password")
#os.getenv('NIXTLA_API_KEY')

nixtla_client = NixtlaClient(
    api_key = nixtla_api_key,
)
nixtla_client.validate_api_key()

if nixtla_api_key:
    # Upload csv file
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        # print(uploaded_file)
        # Read csv file
        df = pd.read_csv(csv_path + uploaded_file.name) # skiprows=6 (skip first 6 rows)
        # df = df[:-1] # remove last row
        # Convert to datetime format
        print(df.columns)
        df["Date"] = pd.to_datetime(df["Date"])
        # Preview data
        st.write("Data preview:")
        st.write(df.head(10))

        d = st.date_input(
            "Select your date range as historical data: ",
            (pd.to_datetime(df["Date"].min()), pd.to_datetime(df["Date"].max())),
            format="MM.DD.YYYY",
        )
        print("Min date:", d[0])
        print("Max date:", d[1])
        min_date = str(d[0])
        max_date = str(d[1])
        df_filter = df.loc[(df["Date"] >= min_date) & (df["Date"] <= max_date)]
        
        column_list = df_filter.drop(columns=["Date"]).columns
        
        if d:
            TARGET = st.selectbox("Select Target variable", column_list)
            forecast_horizon = st.number_input("Forecast horizon", value=None, min_value=1, max_value=365)

        if TARGET and forecast_horizon is not None:
            df_processed = df_filter.copy()
            df_processed["unique_id"] = 1
            df_processed = df_processed[["unique_id", "Date", TARGET]]
            df_processed = df_processed.rename(columns={"Date":"ds",
                                    TARGET:"y"})
            timegpt_fcst_df = nixtla_client.forecast(df=df_processed, h=forecast_horizon, freq='D', time_col='ds', target_col='y')
            if df_processed.y.dtypes == "int64":
                timegpt_fcst_df["TimeGPT"] = round(timegpt_fcst_df["TimeGPT"])
            
            st.write("TimeGPT forecast result:")
            st.write(timegpt_fcst_df)
            st.write("Plot:")
            st.write(nixtla_client.plot(df_processed, timegpt_fcst_df, time_col='ds', target_col='y'))
    