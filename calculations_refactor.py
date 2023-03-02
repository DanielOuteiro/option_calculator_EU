import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.colors as colors
import re
import yfinance as yf
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import time


# Define a list of ticker symbols to choose from
symbols = ['^GSPC', '^GDAXI']

# Define the ticker symbol for the S&P 500
tickerSymbol = st.sidebar.selectbox('Ticker Symbol', symbols)

# Retrieve the data for the S&P 500
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = st.sidebar.date_input('Load Historical data', value=pd.to_datetime('2022-01-01'))

# Download the data for the S&P 500
tickerData = yf.Ticker(tickerSymbol)
all_tickerDf = tickerData.history(period='1d', start=start_date, end=end_date)

# Filter the data based on the start and end dates selected by the user
tickerDf = all_tickerDf.loc[start_date:end_date]

# Get the current S&P 500 price
current_price = tickerDf.iloc[-1]['Close']

# Set the target range based on the selected ticker symbol
if tickerSymbol == '^GSPC':
    target_range = (0, 5000, 4000)
    ticker_name = 'S&P 500'
    url = 'https://www.onvista.de/derivate/Optionsscheine/Optionsscheine-auf-S-P-500'
else:
    target_range = (0, 16000, 14000)
    ticker_name = 'DAX'
    url = 'https://www.onvista.de/derivate/Optionsscheine/Optionsscheine-auf-DAX'


# Get the target value from the user using a slider
target = st.slider('Target', min_value=target_range[0], max_value=target_range[1], value=target_range[2])

# Calculate the difference between the latest price and the target
difference = -((1-(target / current_price))*100)
st.write(f"Change: **{round(difference, 2)}%**")


# Add sliders for the vertical date and target value
vertical_date = (datetime.today() + timedelta(days=15)).date()
vertical_date = st.date_input('Expiration Date', value=vertical_date)

# Define the US Federal Holiday Calendar
cal = USFederalHolidayCalendar()
# Define the Custom Business Day offset with the US Federal Holidays
us_bd = CustomBusinessDay(calendar=cal)
# Get today's date
today = pd.Timestamp.today().normalize()

# Calculate the difference between the deadline and today in business days
diff_in_days = pd.date_range(start=today, end=vertical_date, freq=us_bd).size

st.write(f"Days to go: **{diff_in_days} days**")

#st.write(current_price)

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

# Filter the historical data for the S&P 500 based on the start date
sp500_data = all_tickerDf



days = diff_in_days
percent_fall = -difference

# Loop through the data and count the instances where the price fell by y percent or more within x or less days
count = 0
total_instances = 0
for days in range(1, diff_in_days + 1):
    for i in range(len(sp500_data) - days):
        start_price = sp500_data.iloc[i]["Close"]
        end_price = sp500_data.iloc[i + days]["Close"]
        percent_change = (end_price - start_price) / start_price * 100
        if percent_change <= -percent_fall:
            count += 1
        total_instances += 1

# Calculate the probability of the event happening
if total_instances > 0:
    probability = count / total_instances
    st.write(f"The {ticker_name} moved by {-(round(percent_fall, 2))}% or more within {days} or less days {count} times in history, which represents a {probability:.2%} probability")
else:
    st.write("Insufficient data.")




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
#st.write(f"The average move in the S&P 500 over {days} days is: {average_move:.2f}%")


# Get the date when the maximum drop occurred
max_drop_date = sp500_data.index[max_drop_index + days].date()

# Create a new dataframe with the prices for the period when the maximum drop occurred
start_date = sp500_data.index[max_drop_index].date()
end_date = sp500_data.index[max_drop_index + days].date()
max_drop_df = sp500_data.loc[start_date:end_date]

# Create the plot of the prices for the period when the maximum drop occurred
fig, ax = plt.subplots()
ax.plot(max_drop_df.index, max_drop_df['Close'])
ax.set_title(f"{ticker_name} Prices from {start_date} to {end_date}")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
fig.autofmt_xdate()

maxdroppct = (1-(-(max_drop/100)))
woulfallto = round(current_price*maxdroppct,2)

# Display the maximum drop and the graph in the Streamlit app
st.write(f"The maximum drop in the {ticker_name} that happened in {days} days is: {max_drop:.2f}% on {max_drop_date}. If that happened today, the {ticker_name} would fall to {woulfallto} ")

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






# Set the base URL and default parameters
#url = 'https://www.onvista.de/derivate/Optionsscheine/Optionsscheine-auf-S-P-500'
params = {
    'sort': 'dateMaturity',
    'order': 'ASC',
    'page': 0,
    'cols': 'instrument,strikeAbs,dateMaturity,quote.bid,quote.ask,leverage,impliedVolatilityAsk,spreadAskPct,derivativesSubCategory.id,premiumAsk,nameExerciseStyle,issuer.name,urlProspectus'
}

# Define a function to format the date range parameter
def format_date_range(start_date, end_date):
    return f"{start_date.strftime('%Y-%m-%d')};{end_date.strftime('%Y-%m-%d')}"

# Set the page title and sidebar
st.sidebar.title('Options Data Parameters')

# Add the exercise right dropdown
exercise_right = st.sidebar.selectbox('Exercise Right', ['1', '2'])

# Get the next Friday after the vertical date
days_to_friday = (4 - vertical_date.weekday()) % 7
if days_to_friday == 0:
    days_to_friday = 7
next_friday = vertical_date + pd.Timedelta(days_to_friday, unit='D')

# Add the date input widgets
start_date = st.sidebar.date_input('Start Date', value=pd.Timestamp(vertical_date))
end_date = st.sidebar.date_input('End Date', value=pd.Timestamp(next_friday))

date_range = format_date_range(start_date, end_date)

# Add the strike range pickers
strike_abs_start = st.sidebar.text_input('Strike Abs Start', value='')
strike_abs_end = st.sidebar.text_input('Strike Abs End', value='')
if strike_abs_start and strike_abs_end:
    params['strikeAbsRange'] = f"{strike_abs_start};{strike_abs_end}"


# Define the US Federal Holiday Calendar
cal = USFederalHolidayCalendar()

# Define the Custom Business Day offset with the US Federal Holidays
us_bd = CustomBusinessDay(calendar=cal)

# Get today's date
today = pd.Timestamp.today().normalize()

# Add a date picker for the deadline
deadline = vertical_date

# Calculate the difference between the deadline and today in business days
diff_in_days = pd.date_range(start=today, end=deadline, freq=us_bd).size




# Add an input field for the change value
change = target

# Add an input field for the change value
investment = st.sidebar.number_input('Investment', value=100)



# Get the S&P 500 index ticker symbol
sp500_ticker = tickerSymbol

# Retrieve the current data for the S&P 500 index
sp500_data = yf.download(sp500_ticker, period="1d")

# Extract the current value of the S&P 500 index
sp500_value = sp500_data["Close"].iloc[-1]

# Print the current value of the S&P 500 index
##st.write(f"The current value of the S&P 500 index is: {sp500_value:.2f}")

pctdif = (-(1-(change / sp500_value))*100) 

# Add a button to start scraping the datas

if st.sidebar.button('Scrape Data'):
    # Update the parameters with the user inputs
    total_start_time =  time.time()
    start_time = time.time()

    
    params['idExerciseRight'] = exercise_right
    params['dateMaturityRange'] = date_range

    # Initialize an empty list to store the data
    data = []

    # Set the flag to True to start the loop
    has_data = True

    # Loop through all the pages
    while has_data:
        # Make a GET request to the page
        response = requests.get(url, params=params)

        # Parse the HTML content of the page with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the table element with the data we need
        table = soup.find('table', {'class': 'table'})

        # If there is a table, extract the data and add it to the list
        if table:
            # Get the column headers from the table
            headers = [th.text for th in table.select('tr th')]

            # Get the data rows from the table
            rows = []
            for tr in table.select('tr')[1:]:
                rows.append([td.text for td in tr.select('td')])

            # Add the data to the list
            data.extend(rows)

            # Increment the page parameter for the next iteration
            params['page'] += 1

        # If there is no table, set the flag to False to break the loop
        else:
            has_data = False

    # Convert the data to a Pandas DataFrame
    df = pd.DataFrame(data, columns=headers)
    df = df.apply(pd.to_numeric, errors='ignore')

    # Display the DataFrame in the app
    #st.dataframe(df)

end_time = time.time()
st.write((f"1/7: {end_time - start_time:.2f} seconds")) 

start_time = time.time()
# Set the API endpoint URL
api_url = 'https://api.onvista.de/api/v1/instruments/query'

# Initialize an empty list to store the API response data
api_data = []

# Loop through the data rows and make an API request for each value in the first column
for row in data:
    search_value = row[0]  # Get the search value from the first column of the row
    params = {'searchValue': search_value}  # Set the searchValue parameter for the API request
    response = requests.get(api_url, params=params)  # Make the API request
    response_data = response.json()
    if response_data and 'list' in response_data and response_data['list']:
        api_data.append(response_data['list'][0])  # Append the first list item to the list

# Convert the API data to a Pandas DataFrame
api_df = pd.json_normalize(api_data)

# Display the DataFrame in the app
#st.write('API Results')
#st.dataframe(api_df)

end_time = time.time()
st.write((f"2/7: {end_time - start_time:.2f} seconds")) 

start_time = time.time()
# Initialize an empty list to store the API response data
calculator_results = []

# Loop through the rows of the "api_df" dataframe
for index, row in api_df.iterrows():
    # Get the "entityValue" value for the row
    entity_value = row['entityValue']
    # Make the API request using the "entityValue" value
    url = f'https://api.onvista.de/api/v1/derivatives/{entity_value}/calculatorResult'
    response = requests.get(url)
    response_data = response.json()
    # Append the response data to the "calculator_results" list
    calculator_results.append(response_data)

# Convert the "calculator_results" list to a Pandas DataFrame
calculator_results_df = pd.json_normalize(calculator_results)



# Display the DataFrame in the app
#st.write('Calculator Results')
#st.dataframe(calculator_results_df)

end_time = time.time()
st.write((f"3/7: {end_time - start_time:.2f} seconds")) 

start_time = time.time()

# Merge the two dataframes on their index
merged_df = api_df.join(calculator_results_df, on=api_df.index)


# Display the merged DataFrame in the app
#st.write('Merged DataFrame')
#st.dataframe(merged_df)

end_time = time.time()
st.write((f"4/7: {end_time - start_time:.2f} seconds")) 

start_time = time.time()

# Initialize an empty list to store the API response data
result_data = []

# Loop through the rows of the "merged_df" dataframe
for index, row in merged_df.iterrows():
    # Get the values for the URL parameters
    entity_value = row['entityValue']
    exchange_rate = row['exchangeRate']
    interest_instrument = row['interestInstrument']
    volatility_instrument = row['volatilityInstrument']
    url = f'https://api.onvista.de/api/v1/derivatives/{entity_value}/calculatorResult'
    params = {
        'dateCalculation': deadline.strftime('%Y-%m-%d'),
        'exchangeRate': exchange_rate,
        'interestInstrument': interest_instrument,
        'priceUnderlying': change,
        'volatility': volatility_instrument
    }
    response = requests.get(url, params=params)
    response_data = response.json()
    # Append the response data to the "result_data" list
    result_data.append(response_data)

# Convert the "result_data" list to a Pandas DataFrame
result_df = pd.json_normalize(result_data)

# Display the DataFrame in the app
#st.write('Result Data')
#st.dataframe(result_df)

end_time = time.time()
st.write((f"5/7: {end_time - start_time:.2f} seconds")) 


start_time = time.time()

# Select the desired columns from the merged_df DataFrame
merged_cols = merged_df.loc[:, ['entityValue', 'name', 'wkn', 'urls.WEBSITE']]

# Select the desired columns from the result_df DataFrame
result_cols = result_df.loc[:, ['calculatedDiffInstrumentPct', 'calculatedPriceInstrument', 'askPriceInstrument', 'bidPriceInstrument' ]]

# Merge the DataFrames by their index
merged_results_df = pd.merge(merged_cols, result_cols, left_index=True, right_index=True)

merged_results_df['calculated_ask_diff'] = merged_results_df['calculatedPriceInstrument'] - merged_results_df['askPriceInstrument']
merged_results_df['Investment Value'] = (investment / merged_results_df['askPriceInstrument']) * merged_results_df['calculatedPriceInstrument']




# Display the merged DataFrame in the app
#st.write('Merged Results')
#st.dataframe(merged_results_df)

end_time = time.time()
st.write((f"6/7: {end_time - start_time:.2f} seconds")) 


start_time = time.time()
merged_cols_2 = df.loc[:, ['Aufgeld', 'Basispreis', 'WKN', 'Bewertungstag']]

result_cols_2 = merged_results_df.loc[:, ['entityValue', 'name', 'wkn', 'urls.WEBSITE', 'calculatedDiffInstrumentPct', 'calculatedPriceInstrument', 'askPriceInstrument', 'bidPriceInstrument', 'calculated_ask_diff', 'Investment Value']]

merged_results_df_final = pd.merge(merged_cols_2, result_cols_2, left_index=True, right_index=True)

# Convert the "Basispreis" column to string and replace the comma and the "Pkt." strings
basispreis_str = merged_results_df_final['Basispreis'].str.replace(',', '').str.replace(' Pkt.', '')

# Convert non-finite values to NaN
basispreis_str = pd.to_numeric(basispreis_str, errors='coerce').multiply(1000)

# Create a numerical mapping of the unique values in the 'Bewertungstag' column
date_mapping = pd.factorize(merged_results_df_final['Bewertungstag'])[0]

# Create a color scale with the same number of colors as the number of unique dates
num_dates = len(merged_results_df_final['Bewertungstag'].unique())
color_scale = colors.qualitative.Light24[:num_dates]

# Create a list of colors indexed by the numerical mapping
color_mapper = [color_scale[i] for i in date_mapping]

# Create the scatter plot with the colors assigned using the color mapper
fig = go.Figure(data=go.Scatter(
    x=merged_results_df_final['Investment Value'],
    y=basispreis_str,
    mode='markers',
    marker=dict(
        color=color_mapper,
        colorscale=color_scale
    )
))

# Display the merged DataFrame in the app
#st.write('Final Results')
#st.dataframe(merged_results_df_final)
end_time = time.time()
st.write((f"7/7: {end_time - start_time:.2f} seconds")) 


# Set the title and axis labels
fig.update_layout(title='Scatter Plot', xaxis_title='Investment Value', yaxis_title='Strike')






# Add a vertical line at the value of the S&P 500 index
fig.add_hline(y=sp500_value, line_dash="dash", line_color="green", annotation_text=f"S&P 500 Value: {sp500_value:.2f}")

# Add a vertical line at the value of the S&P 500 index
fig.add_hline(y=change, line_dash="dash", line_color="blue", annotation_text=f"Change: {change}")

fig.add_vline(x=100, line_dash="dash", line_color="blue")



# Display the plot in the app
st.plotly_chart(fig, use_container_width=False)

total_end_time = time.time()
st.write((f"Total time: {total_end_time - total_start_time:.2f} seconds")) 



















