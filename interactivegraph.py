import yfinance as yf
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay



# Define the ticker symbol for the S&P 500
tickerSymbol = '^GSPC'

# Retrieve the data for the S&P 500
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('2022-01-01'))
tickerData = yf.Ticker(tickerSymbol)

# Get the historical prices for the S&P 500
tickerDf = tickerData.history(period='1d', start=start_date, end=end_date)

# Get the current S&P 500 price
current_price = tickerDf.iloc[-1]['Close']

# Add sliders for the vertical date and target value
vertical_date = (datetime.today() + timedelta(days=15)).date()
vertical_date = st.sidebar.date_input('Expiration Date', value=vertical_date)

target = st.sidebar.slider('Target', min_value=0, max_value=5000, value=4000)

# Create the plot
fig, ax = plt.subplots()
ax.plot(tickerDf.index, tickerDf['Close'])

# Add the vertical line
ax.axvline(vertical_date, color='r', linestyle='--')
ax.axvline(pd.to_datetime(end_date), color='b', linestyle='--')

# Add the horizontal line
ax.axhline(target, color='r', linestyle='--')
ax.axhline(current_price, color='b', linestyle='--')

# Format the x-axis date labels
fig.autofmt_xdate()

# Display the graph using streamlit
st.pyplot(fig)

# Get the latest S&P 500 price
latest_price = tickerDf.iloc[-1]['Close']

# Calculate the difference between the latest price and the target
difference = -((1-(target / latest_price))*100)

st.write(f"Change in %: {round(difference, 2)}")



# Define the US Federal Holiday Calendar
cal = USFederalHolidayCalendar()
# Define the Custom Business Day offset with the US Federal Holidays
us_bd = CustomBusinessDay(calendar=cal)
# Get today's date
today = pd.Timestamp.today().normalize()

# Calculate the difference between the deadline and today in business days
diff_in_days = pd.date_range(start=today, end=vertical_date, freq=us_bd).size

st.write(f"Days to go: {diff_in_days} days.")

# Get the historical data for the S&P 500
sp500_data = yf.download("^GSPC", start="1950-01-01")

days = diff_in_days
percent_fall = -difference


# Loop through the data and count the instances where the price fell by y percent or more within x days
count = 0
for i in range(len(sp500_data) - days):
    start_price = sp500_data.iloc[i]["Close"]
    end_price = sp500_data.iloc[i + days]["Close"]
    percent_change = (end_price - start_price) / start_price * 100
    if percent_change <= -percent_fall:
        count += 1

# Calculate the percentage of times in history
total_instances = len(sp500_data) - days
if total_instances > 0:
    percentage = count / total_instances * 100
    st.write((f"The S&P 500 fell by {round(percent_fall, 2)}% or more within {days} days {count} times in history, which represents {percentage:.2f}% of all instances."))
else:
    st.write(("Insufficient data."))




prices = sp500_data["Close"]
max_drop = 0
max_drop_index = 0
total_move = 0
for i in range(len(prices) - days):
    start_price = prices[i]
    end_price = prices[i + days]
    percent_change = (end_price - start_price) / start_price * 100
    total_move += percent_change
    if percent_change < max_drop:
        max_drop = percent_change
        max_drop_index = i

# Calculate the average move in the S&P 500 over X days
average_move = total_move / (len(prices) - days)

st.write(f"The average move in the S&P 500 over {days} days is: {average_move:.2f}%")


# Get the date when the maximum drop occurred
max_drop_date = sp500_data.index[max_drop_index + days].date()

# Create a new dataframe with the prices for the period when the maximum drop occurred
start_date = sp500_data.index[max_drop_index].date()
end_date = sp500_data.index[max_drop_index + days].date()
max_drop_df = sp500_data.loc[start_date:end_date]

# Create the plot of the prices for the period when the maximum drop occurred
fig, ax = plt.subplots()
ax.plot(max_drop_df.index, max_drop_df['Close'])
ax.set_title(f"S&P 500 Prices from {start_date} to {end_date}")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
fig.autofmt_xdate()

maxdroppct = (1-(-(max_drop/100)))
woulfallto = round(latest_price*maxdroppct,2)

# Display the maximum drop and the graph in the Streamlit app
st.write(f"The maximum drop in the S&P 500 that happened in {days} days is: {max_drop:.2f}% on {max_drop_date}. If that happened today, the SP500 would fall to {woulfallto} ")

# Find the last time a drop with the same value (or more) and period happened
last_date = ""
last_drop = 0
for i in range(len(prices) - days):
    start_price = prices[i]
    end_price = prices[i + days]
    percent_change = (end_price - start_price) / start_price * 100
    if percent_change <= max_drop and i + days < max_drop_index:
        last_date = sp500_data.index[i + days].date()
        last_drop = percent_change

st.pyplot(fig)
