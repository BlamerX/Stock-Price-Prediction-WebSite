import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
from stocknews import StockNews
from sklearn.linear_model import LinearRegression

st.title('Stock Price Prediction')
st.sidebar.header('BlamerX')

tickers=['ETH-USD', 'TSLA', 'AAPL', 'GOOGL', 'MSFT', 'RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS']

ticker_symbol = st.sidebar.selectbox('Enter a Ticker Symbol', tickers)

start = '2010-01-01'
end = date.today().strftime("%Y-%m-%d")

start_date = st.sidebar.date_input("Start Date", pd.to_datetime(start)).strftime("%Y-%m-%d")
end_date = st.sidebar.date_input("End Date", pd.to_datetime(end)).strftime("%Y-%m-%d")
no_of_news = st.sidebar.slider("Number of News You Want", 1, 15, 3)

st.sidebar.warning("Disclaimer: This site is for educational purposes only. Do not use the information provided for real trading or investment decisions.")

st.sidebar.markdown('[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/adarsh-kumar-374150171/)')
st.sidebar.markdown('[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/BlamerX)')
st.sidebar.markdown('[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/blamerx)')

# Retrieve the full name for the ticker symbol
ticker = yf.Ticker(ticker_symbol)
ticker_info = ticker.info
full_name = ticker_info.get('longName', '')
description = ticker_info.get('description','')
longbusinesssummary=ticker_info.get('longBusinessSummary','')
st.subheader(f'{full_name} - {ticker_symbol}')
st.write(description)
st.write(longbusinesssummary)


# Download stock price data
data = yf.download(ticker_symbol, start=start_date, end=end_date)

fig = px.line(data, x=data.index, y=data['Adj Close'])
fig.update_layout(title={
    'text': f'{ticker_symbol} - {full_name}',
    'x': 0.2 })
st.plotly_chart(fig)

# Get the most recent stock price
latest_price = data['Close'].iloc[-1]

# Get the previous day's stock price
previous_price = data['Close'].iloc[-2]

# Calculate the price difference and percentage change
price_difference = latest_price - previous_price
percentage_change = (price_difference / previous_price) * 100

# Display the stock price and the increase as a fancy number
st.metric("Current Price", f"${latest_price:.2f}", f"+{price_difference:.2f} (+{percentage_change:.2f}%)")

pricing_data,news,predictions=st.tabs(["Pricing Data",f"Top {no_of_news} News", "Predictions"])

with pricing_data:
    st.header("Price Movements")
    data2 = data
    data2['% Change'] = data["Adj Close"] / data['Adj Close'].shift(1) - 1
    data2.dropna(inplace=True)
    st.table(data2)
    annual_return = data2["% Change"].mean() * 252 * 100
    if annual_return > 0:
        st.write("Annual Return is ", ":arrow_up:", annual_return, '%')
    else:
        st.write("Annual Return is ", ":arrow_down:", annual_return, '%')
    stdev = np.std(data2['% Change']) * np.sqrt(252)
    st.write('Standard Deviation is ', stdev * 100, '%')
    st.write('Risk Adj. Return is ', annual_return / (stdev * 100))

with news:
    st.header(f'News of {ticker_symbol} - {full_name}')
    sn = StockNews(ticker_symbol, save_news=False)
    df_news = sn.read_rss()
    for i in range(no_of_news):
        st.subheader(f'News {i+1}')
        st.write(df_news['published'][i])
        st.write(df_news['title'][i])
        st.write(df_news['summary'][i])
        title_sentiment = df_news['sentiment_title'][i]
        st.write(f'Title Sentiment {title_sentiment}')
        news_sentiment = df_news['sentiment_summary'][i]
        st.write(f'News Sentiment {news_sentiment}')

with predictions:
    st.header("Predictions")

    no_of_weeks = st.slider("No of Weeks you want to use for prediction", 1, 50, 12)
    prediction_data = data[['Open', 'High', 'Low', 'Close']].tail(no_of_weeks * 7)  # Last n weeks of data (assuming 5 trading days in a week)
    prediction_dates = pd.date_range(prediction_data.index[-1], periods=200, freq='B')  # Business days for prediction
    prediction_dates_ord = prediction_dates.to_series().apply(lambda x: x.toordinal())
    X = prediction_data.index.to_series().apply(lambda x: x.toordinal()).values.reshape(-1, 1)
    y = prediction_data['Close'].values

    # Perform linear regression
    reg = LinearRegression()
    reg.fit(X, y)

    # Predict for the next n days
    no_days_prediction = st.slider("No of Future Predictions", 1, 25, 5)
    future_dates = pd.date_range(end_date, periods=no_days_prediction)
    future_dates_ord = future_dates.to_series().apply(lambda x: x.toordinal())
    future_prices = reg.predict(future_dates_ord.values.reshape(-1, 1))

    # Create a dataframe for the predictions
    prediction_data = pd.DataFrame({'Date': future_dates, 'Close': future_prices})
    prediction_data.set_index('Date', inplace=True)

    # Combine original data and predictions
    combined_data = pd.concat([data[['Open', 'High', 'Low', 'Close']], prediction_data[['Close']]], axis=0)

    # Slider for selecting the start year for plotting
    start_year = st.selectbox("Select the start year for plotting", options=list(range(data.index.year.min(), date.today().year + 1)))

    # Filter data based on selected start year
    selected_data = combined_data.loc[pd.to_datetime(f"{start_year}-01-01"):]

    # Create candlestick chart with selected data
    fig = go.Figure(data=[go.Candlestick(
        x=selected_data.index,
        open=selected_data['Open'],
        high=selected_data['High'],
        low=selected_data['Low'],
        close=selected_data['Close'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )])
    fig.update_layout(
        title={'text':f'{ticker_symbol} - {full_name} Candlestick Chart with Predictions',
        'x': 0.2},
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig)

    # Display the predictions table
    st.subheader(f'Predicted Prices for the next {no_days_prediction} days')
    st.dataframe(prediction_data)
    st.warning("Disclaimer: This information is for educational purposes only. Do not use the information provided for real trading or investment decisions.")