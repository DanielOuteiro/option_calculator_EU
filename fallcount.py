import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import datetime
import matplotlib.dates as mdates

st.set_page_config(layout="wide")


@st.cache_data
def load_data():
    sp500 = yf.Ticker("^GSPC")
    sp500_history = sp500.history(start="1995-01-01")
     
    ##st.write(sp500_history.head())
    ##st.write(sp500_history.tail())

    return sp500_history


def resample_data(data, period):
    data = data.resample(period).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    })
    return data.dropna()


def create_plot(data, y_range=None, x_range=None, line_value=None):
 
    fig = go.Figure()

    # Add candlestick chart
    fig.add_trace(go.Candlestick(x=data.index, 
                                 open=data['Open'], 
                                 high=data['High'], 
                                 low=data['Low'], 
                                 close=data['Close'], 
                                 name="S&P 500"))

    # Add horizontal line at current S&P 500 value
    current_value = data.iloc[-1]['Close']
    fig.add_trace(go.Scatter(x=[data.index[0], data.index[-1]], 
                             y=[current_value, current_value], 
                             mode='lines', 
                             name='Current Value'))

    # Add horizontal line at specified value
    if line_value is not None:
        fig.add_trace(go.Scatter(x=[x_range[0], x_range[-1]], 
                                 y=[line_value, line_value], 
                                 mode='lines', 
                                 name='Target'))
        fig.add_shape(
            type="line",
            x0=x_range[0],
            y0=line_value,
            x1=x_range[-1],
            y1=line_value,
            line=dict(color="red", width=1, dash="dash"),
            name='Target'
        )



    if y_range:
        fig.update_yaxes(range=y_range)

    return fig





# Add a date picker for the deadline
deadline = st.sidebar.date_input('Deadline', value=pd.Timestamp('2023-02-23'))

# Convert deadline date to same format as date column in sp500_history
deadline_str = deadline.strftime('%Y-%m-%d')
deadline_date = pd.to_datetime(deadline_str)



# Load data
sp500_history = load_data()

# Resample data to reduce the amount of data
sp500_history_resampled = resample_data(sp500_history, "1D")

# Show data table
##st.write(sp500_history_resampled)

line_value = st.sidebar.slider("Target", value=max(sp500_history_resampled["Close"]), 
                       format="%.2f", step=0.01, 
                       min_value=0.0, 
                       max_value=max(sp500_history_resampled["Close"]))

# Store initial x-axis range
x_range = [sp500_history_resampled.index[0], sp500_history_resampled.index[-1]]

# Show interactive chart with initial y-axis range and horizontal line
fig = create_plot(sp500_history_resampled, x_range=x_range, line_value=line_value)

# Add a vertical line at the deadline date
fig.add_vline(x=deadline_date, line_dash="dash", line_color="red", annotation_text="Deadline")

# Store initial x-axis range
initial_x_range = x_range

# Add horizontal line shape
fig.add_shape(
    type="line",
    x0=sp500_history_resampled.index[0],
    y0=line_value,
    x1=sp500_history_resampled.index[-1],
    y1=line_value,
    line=dict(color="red", width=1, dash="dash"),
)

# Add line trace for horizontal line value
fig.add_trace(go.Scatter(x=[sp500_history_resampled.index[0], sp500_history_resampled.index[-1]], 
                         y=[line_value, line_value], 
                         mode='lines', 
                         name='Horizontal Line Value'))

st.plotly_chart(fig, use_container_width=True)


# Define the US Federal Holiday Calendar
cal = USFederalHolidayCalendar()

# Define the Custom Business Day offset with the US Federal Holidays
us_bd = CustomBusinessDay(calendar=cal)

# Get today's date
today = pd.Timestamp.today().normalize()


# Calculate the difference between the deadline and today in business days
diff_in_days = pd.date_range(start=today, end=deadline, freq=us_bd).size




# Add an input field for the change value
change = line_value


# Get the S&P 500 index ticker symbol
sp500_ticker = "^GSPC"

# Retrieve the current data for the S&P 500 index
sp500_data = yf.download(sp500_ticker, period="1d")

# Extract the current value of the S&P 500 index
sp500_value = sp500_data["Close"].iloc[-1]

# Print the current value of the S&P 500 index
##st.write(f"The current value of the S&P 500 index is: {sp500_value:.2f}")

pctdif = (-(1-(change / sp500_value))*100) 

st.write(f"The change would be of: {pctdif:.2f}%")


# Display the difference in business days
st.write(f"The difference between today and the deadline is {diff_in_days} days.")

fallday = pctdif / diff_in_days
fallday_formatted = round(fallday, 2)

st.write(f"That would be a fall of {fallday_formatted}% per day.")

# Define the parameters
days = diff_in_days
percent_fall = -pctdif

# Get the historical data for the S&P 500
sp500_data = yf.download("^GSPC", start="1900-01-01")

# Define the US Federal Holiday Calendar
cal = USFederalHolidayCalendar()

# Define the Custom Business Day offset with the US Federal Holidays
us_bd = CustomBusinessDay(calendar=cal)

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