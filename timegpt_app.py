import os
import pandas as pd
import numpy as np
from io import StringIO
from nixtla import NixtlaClient
import streamlit as st

st.title("TimeGPT App")
st.write("This is a simple app to generate time series forecasts using TimeGPT model.")

nixtla_api_key = st.text_input("Enter your Nixtla API key", type="password")
#os.getenv('NIXTLA_API_KEY')

if nixtla_api_key:
    nixtla_client = NixtlaClient(
        api_key = nixtla_api_key,
    )
    nixtla_client.validate_api_key()
    # Upload csv file
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding='unicode_escape')
        if "Date" or "date" in df.columns.tolist():
            # Convert to datetime format
            print(df.columns)
            df["Date"] = pd.to_datetime(df["Date"])

            d = st.date_input(
                "Select your date range as historical data: ",
                (pd.to_datetime(df["Date"].min()), pd.to_datetime(df["Date"].max())),
                format="MM.DD.YYYY",
            )
            if d:
                min_date = str(d[0])
                max_date = str(d[1])
                df_filter = df.loc[(df["Date"] >= min_date) & (df["Date"] <= max_date)]
                date_range = pd.date_range(start=min_date, end=max_date, freq='D')
                date_df = pd.DataFrame({'Date': date_range})

                # Merge with Original Data
                df_merged = pd.merge(date_df, df_filter, on='Date', how='left')
                # Select numeric columns
                column_list = df_merged.select_dtypes(include=np.number).columns.tolist()
                target = st.selectbox("Select Target variable", column_list)
                if target:
                    df_merged = df_merged[["Date", target]]
                    df_merged[target] = df_merged[target].fillna(0)
                    # Preview selected data
                    st.write("Selected Data preview:")
                    st.write(df_merged.head(10))
                    group_method = st.selectbox("Select groupby aggregate method: ", ["Sum", "Mean"])
                    if group_method == "Sum":
                        df_merged = df_merged.groupby('Date').sum().reset_index()
                    else:
                        df_merged = df_merged.groupby('Date').mean().reset_index()

                    df_merged = df_merged.drop_duplicates(subset=["Date"])
                    forecast_horizon = st.number_input("Forecast horizon", value=None, min_value=1, max_value=365)

            if target and forecast_horizon is not None:
                df_processed = df_merged.copy()
                df_processed["unique_id"] = 1
                df_processed = df_processed[["unique_id", "Date", target]]
                df_processed = df_processed.rename(columns={"Date":"ds",
                                        target:"y"})
                try:
                    timegpt_fcst_df = nixtla_client.forecast(df=df_processed, h=forecast_horizon, freq='D', time_col='ds', target_col='y')
                    if df_processed.y.dtypes == "int64":
                        timegpt_fcst_df["TimeGPT"] = round(timegpt_fcst_df["TimeGPT"])
                    
                    st.write("TimeGPT forecast result:")
                    st.write(timegpt_fcst_df)
                    st.write("Plot:")
                    st.write(nixtla_client.plot(df_processed, timegpt_fcst_df, time_col='ds', target_col='y'))
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.info("Please check your request limit at Nixtla Dashboard: https://dashboard.nixtla.io/")
        else:
            st.info("No date columns in the csv file")