import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objs as go

@st.cache
def load_data():
    sp500 = yf.Ticker("^GSPC")
    sp500_history = sp500.history(period="max")
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
                                 name='Horizontal Line Value'))
        fig.add_shape(
            type="line",
            x0=x_range[0],
            y0=line_value,
            x1=x_range[-1],
            y1=line_value,
            line=dict(color="red", width=1, dash="dash"),
            name='Horizontal Line Value'
        )

    # Customize layout
    fig.update_layout(
        title="S&P 500 Historical Data",
        xaxis_title="Date",
        yaxis_title="Price",
        height=500,
    )

    if y_range:
        fig.update_yaxes(range=y_range)

    return fig


# Load data
sp500_history = load_data()

# Resample data to reduce the amount of data
sp500_history_resampled = resample_data(sp500_history, "1W")

# Show data table
st.write(sp500_history_resampled)

# Add slider widgets to set y-axis range and horizontal line value
y_min, y_max = st.slider("Y-axis range", value=(min(sp500_history_resampled["Low"]), max(sp500_history_resampled["High"])), 
                         format="%.2f", step=0.01, 
                         min_value=min(sp500_history_resampled["Low"]), 
                         max_value=max(sp500_history_resampled["High"]))

line_value = st.slider("Horizontal Line Value", value=max(sp500_history_resampled["Close"]), 
                       format="%.2f", step=0.01, 
                       min_value=0.0, 
                       max_value=max(sp500_history_resampled["Close"]))

# Store initial x-axis range
x_range = [sp500_history_resampled.index[0], sp500_history_resampled.index[-1]]

# Show interactive chart with initial y-axis range and horizontal line
y_range = [y_min, y_max]
fig = create_plot(sp500_history_resampled, y_range=y_range, x_range=x_range, line_value=line_value)

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

# Show interactive chart
st.plotly_chart(fig)

# Update y-axis range and horizontal line when sliders are moved
if st.sidebar.button("Update"):
    # Get current x-axis range
    current_x_range = fig['layout']['xaxis']['range']

    # Update y-axis range and horizontal line value
    y_min, y_max = st.sidebar.slider("Y-axis range", value=(y_min, y_max), 
                                     format="%.2f", step=0.01, 
                                     min_value=min(sp500_history_resampled["Low"]), 
                                     max_value=max(sp500_history_resampled["High"]))
    
    line_value = st.sidebar.slider("Horizontal Line Value", value=line_value, 
                                   format="%.2f", step=0.01, 
                                   min_value=0.0, 
                                   max_value=max(sp500_history_resampled["Close"]))

    # Redraw chart with updated y-axis range and horizontal line value
    y_range = [y_min, y_max]
    fig = create_plot(sp500_history_resampled, y_range=y_range, x_range=initial_x_range, line_value=line_value)

    # Show updated chart
    st.plotly_chart(fig)


