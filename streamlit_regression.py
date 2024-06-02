import streamlit as st
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

#Load from Script
from  regression import download_data, MLapply

#Function to call the function from Regression Script
def calculate_result(input_value,ML_algos,start_date,end_date):
    df=download_data(input_value,start_date,end_date).tail(504)
    arima,randomforest,lstm=MLapply(df,ML_algos)
    #breakpoint()
    return df,arima,randomforest,lstm

#Streamlit App Design

def main():
    
    input_value = st.sidebar.text_input("Enter the list of stocks for forecasting:", "AAPL,NVDA,MSFT")
    
    # Date range input
    start_date = st.sidebar.date_input("Start date", pd.to_datetime('2023-01-01'))
    end_date = st.sidebar.date_input("End date", pd.to_datetime('2023-12-31'))
    
    # Set up the sidebar with checkboxes
    st.sidebar.title("Select Options")
    #option1 = st.sidebar.checkbox("TimeSeries_ARIMA")
    #option2 = st.sidebar.checkbox("Regression")
    #option3 = st.sidebar.checkbox("RNN_LSTM")
    
    ML_options = st.sidebar.multiselect(
    "What are the algorithm you considering",
    ["TimeSeries_ARIMA", "RandomForest", "RNN_LSTM"],["TimeSeries_ARIMA"])
    
    tab1, tab2 = st.tabs(["Graph", "Data"])

    if st.sidebar.button("Forecast"):
        # Validate input
        try:
            tickers = [ticker.strip() for ticker in input_value.split(',')]
        except ValueError:
            st.error("Please enter a valid info.")
            return 
        
        dataset,arima_df,randomforest_df,lstm_df=calculate_result(tickers,ML_options,start_date,end_date)
  
        with tab1:
            st.header("Projection in Graph")
            dataset['LEGEND']='HISTORICAL'
            arima_df['LEGEND']='ARIMA'
            randomforest_df['LEGEND']='RANDOMFOREST'
            lstm_df['LEGEND']='LSTM_RNN'
            #breakpoint()
            master_df=pd.concat([dataset, arima_df, randomforest_df, lstm_df], ignore_index=True)
            caster_df=master_df.drop('LEGEND', axis=1)
            st.line_chart(caster_df)

        with tab2:
            st.header("Historical Data")
            st.dataframe(dataset)

            st.header("ARIMA FORECAST")
            st.dataframe(arima_df)

            st.header("RANDOM FOREST FORECAST")
            st.dataframe(randomforest_df)

            st.header("LSTM RNN FORECAST")
            st.dataframe(lstm_df)
        
if __name__ == "__main__":
#python -m streamlit run  streamlit_regression.py
    main()