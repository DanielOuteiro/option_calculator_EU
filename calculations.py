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
import seaborn as sns
from scipy import stats
import webbrowser







tab1, tab2, tab3, tab4 = st.tabs(["Scenario probability", "Option Finder", "Current Options [WIP]", "Known Errors and Todo"])

with tab1:
    # Set the page title and sidebar
    #st.sidebar.title('Options Data Parameters')
    st.sidebar.subheader('Scenario probability Parameters')
    
    # Define a dictionary of ticker symbols and their labels
    symbols = {'SP500': '^GSPC', 'DAX': '^GDAXI'}

    # Get the selected ticker symbol and its corresponding label
    tickerLabel = st.sidebar.selectbox('Ticker Symbol', list(symbols.keys()))

    # Get the actual ticker symbol from the dictionary
    tickerSymbol = symbols[tickerLabel]
    
    # Define a function to download the data for a given ticker symbol and date range
    @st.cache_data(ttl=3600)
    def load_data(symbol, start_date, end_date):
        tickerData = yf.Ticker(symbol)
        all_tickerDf = tickerData.history(period='1d', start=start_date, end=end_date)
        filtered_tickerDf = all_tickerDf
        return all_tickerDf, filtered_tickerDf


    def load_data_button():
        global all_tickerDf, tickerDf
        all_tickerDf, tickerDf = load_data(tickerSymbol, start_date, end_date)



    # Retrieve the data for the selected ticker symbol
    end_date = pd.Timestamp.now().date()
    start_date = st.sidebar.date_input('Load Historical data since:', value=pd.Timestamp('2022-01-01').date())
    all_tickerDf, tickerDf = load_data(tickerSymbol, start_date, end_date)



    # Get the current price of the selected symbol
    current_price = tickerDf.iloc[-1]['Close']

    # Set the target range based on the selected ticker symbol
    if tickerSymbol == '^GSPC':
        target_range = (0, 5000, 3800)
        ticker_name = 'S&P 500'
        url = 'https://www.onvista.de/derivate/Optionsscheine/Optionsscheine-auf-S-P-500'

    else:
        target_range = (0, 16000, 14000)
        ticker_name = 'DAX'
        url = 'https://www.onvista.de/derivate/Optionsscheine/Optionsscheine-auf-DAX'


    col1, col2 = st.columns(2)
    with col1:
        # Get the target value from the user using a slider
        target = st.slider('Target', min_value=target_range[0], max_value=target_range[1], value=target_range[2])
        # Calculate the difference between the latest price and the target
        difference = -((1-(target / current_price))*100)
        st.write(f"Change: **{round(difference, 2)}%**")
    with col2:
        # Add sliders for the vertical date and target value
        vertical_date = (datetime.now() + timedelta(days=15)).date()
        vertical_date = st.date_input('Expiration Date', value=vertical_date)
        # Define the US Federal Holiday Calendar
        cal = USFederalHolidayCalendar()
        # Define the Custom Business Day offset with the US Federal Holidays
        us_bd = CustomBusinessDay(calendar=cal)
        # Get today's date
        today = pd.Timestamp.today().normalize()
        # Calculate the difference between the deadline and today in business days
        diff_in_days = pd.date_range(start=today, end=vertical_date, freq=us_bd).size
        st.write(f"Working days to go: **{diff_in_days} days**")






    exercise_right = 1 if target <= current_price else 2







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

    # Calculate the percentage change in price for each trading day within the time period
    price_changes = []
    for i in range(days, len(sp500_data)):
        start_price = sp500_data.iloc[i-days]['Close']
        end_price = sp500_data.iloc[i]['Close']
        percent_change = (end_price - start_price) / start_price * 100
        price_changes.append(percent_change)

    # Define the bins and count the occurrences of percentage changes that fall within each bin
    bins = np.linspace(-50, 50, 101)
    hist = np.zeros(100)
    for change in price_changes:
        bin_index = int(np.floor((change + 50) / 1))
        hist[bin_index] += 1

    # Calculate summary statistics
    mean_change = np.mean(price_changes)
    median_change = np.median(price_changes)
    mode_change = float(stats.mode(price_changes)[0])
    std_dev = np.std(price_changes)
    max_change = np.max(price_changes)
    min_change = np.min(price_changes)


    num_days = st.selectbox('Chart Zoom (days):', options=[30, 60, 90, 120, 240, 360, 720, 2400, 4800, 9600], index=3)


    # Calculate the probability of the event happening
    if total_instances > 0:
        probability = count / total_instances
        st.markdown(f"<div style='text-align: center;'><h2><b>Probability: {probability:.2%} </b></h2></div>", unsafe_allow_html=True)
        #st.write(f"The {ticker_name} moved by {-(round(percent_fall, 2))}% or more within {days} or less days {count} times, which represents a {probability:.2%} probability")
    else:
        st.write("Insufficient data.")

    # Add the exercise right
    if exercise_right == 1:
        st.markdown(f"<div style='text-align: center; font-size: x-small;'><b>All probabilities and values based on the analyzed historical date (from {start_date} to now)</b></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align: center; font-size: x-small;'><b>All probabilities and values based on the analyzed historical date (from {start_date} to now)</b></div>", unsafe_allow_html=True)



    # Filter the data based on the number of days selected by the user
    tickerDf = all_tickerDf.loc[pd.to_datetime(end_date) - pd.Timedelta(days=num_days):end_date]

    # Create the plot
    fig, ax = plt.subplots()
    ax.plot(tickerDf.index, tickerDf['Close'])

    # Add the vertical line
    ax.axvline(vertical_date, color='black')
    #ax.axvline(pd.to_datetime(end_date), color='b', linewidth=0.5)

    max_asvalue = current_price + ((max_change/100) * current_price)
    min_asvalue = current_price + ((min_change/100) * current_price)

    # Add the horizontal line
    ax.axhline(target, color='black', label=f"Your target: {target}")
    #ax.axhline(current_price, color='b', linewidth=0.5)
    ax.axhline(max_asvalue, color='g', linestyle='--', label=f"Max climb in {days} days: {max_change:.2f}%")
    ax.axhline(min_asvalue, color='r', linestyle='--', label=f"Max drop in {days} days: {min_change:.2f}%")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=1)

    # Check if the target value is within the range
    if target < (min_asvalue - 200) or target > (max_asvalue + 200):
        ax.set_ylim(target*0.95)
    else:
        ax.set_ylim([(min_asvalue - 200), (max_asvalue + 200)])

    # Format the x-axis date labels
    fig.autofmt_xdate()

    # Display the graph using streamlit
    st.pyplot(fig)



    with st.expander(f"Histogram of Price Changes over {days} days"):


        # Display the histogram and summary statistics
        st.write()
        fig, ax = plt.subplots()
        ax.bar(bins[:-1], hist, width=1)
        ax.set_xlabel("Percentage Change")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Histogram of Price Changes over {days} days")

        # Add summary statistics to the plot
        #ax.axvline(mean_change, color='blue', linestyle='--', label=f"Mean: {mean_change:.2f}%", alpha=0.1)
        #ax.axvline(median_change, color='blue', linestyle='--', label=f"Median: {median_change:.2f}%", alpha=0.1)
        ax.axvline(max_change, color='green', linestyle='--', label=f"Max: {max_change:.2f}%")
        ax.axvline(min_change, color='red', linestyle='--', label=f"Min: {min_change:.2f}%")
        ax.axvline(difference, color='black', linestyle='--', label=f"Your Target: {difference:.2f}%")


        # Calculate the minimum value of the price changes variable and add a padding of 5%
        xmin = np.min(price_changes) * 1.25

        # Calculate the minimum value of the price changes variable and add a padding of 5%
        xmax = np.max(price_changes) * 1.25

        # Adjust the x-axis range to span from xmin to the maximum value of the price changes variable
        ax.set_xlim([xmin, xmax])


        ax.legend()
        st.pyplot(fig)



        # Create a new figure and axis for the KDE plot
        fig_kde, ax_kde = plt.subplots()

        # Set the style and create the KDE plot
        sns.set_style('darkgrid')
        sns.kdeplot(price_changes, shade=True, color='blue', ax=ax_kde)

        # Add vertical lines for the mean and median
        #ax_kde.axvline(mean_change, color='blue', linestyle='--', alpha=0.5, label=f"Mean: {mean_change:.2f}%")
        #ax_kde.axvline(median_change, color='blue', linestyle='--', alpha=0.5, label=f"Median: {median_change:.2f}%")
        ax_kde.axvline(max_change, color='green', linestyle='--', label=f"Max: {max_change:.2f}%")
        ax_kde.axvline(min_change, color='red', linestyle='--', label=f"Min: {min_change:.2f}%")
        ax_kde.axvline(difference, color='black', linestyle='--', label=f"Your Target: {difference:.2f}%")


        # Set the x-axis label and title
        ax_kde.set_xlabel("Percentage Change")
        ax_kde.set_title("Price Changes Distribution")

        # Adjust the x-axis range to span from xmin to the maximum value of the price changes variable
        ax_kde.set_xlim([xmin, xmax])

        # Show the legend
        ax_kde.legend()

        # Display the figure
        st.pyplot(fig_kde)

    with st.expander(f"Details of the max move over {days} days"):

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
        st.write(f"The maximum drop in the {ticker_name} that happened in {days} days was {max_drop:.2f}% from {start_date} to {end_date}. If that happened today, the {ticker_name} would fall to {woulfallto} ")
        #col10, col11, col12 = st.columns(3)

        #col10.write("")
        #col11.metric(label=f"Maximum drop in the {ticker_name} in ", value=f"{days} days", delta=f"{max_drop:.2f}%")
        #col12.write("")

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





with tab2:

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
    #st.sidebar.title('Options Data Parameters')
    st.sidebar.subheader('Options Finder Parameters')


    # Get the next Friday after the vertical date
    days_to_friday = (4 - vertical_date.weekday()) % 7
    if days_to_friday == 0:
        days_to_friday = 7
    next_friday = vertical_date + pd.Timedelta(days_to_friday, unit='D')

    # Add the date input widgets
    start_date = st.sidebar.date_input('Options expiration from:', value=pd.Timestamp(vertical_date))
    end_date = st.sidebar.date_input('Options expiration to:', value=pd.Timestamp(next_friday))

    date_range = format_date_range(start_date, end_date)

    # Add the strike range pickers
    strike_abs_start = st.sidebar.text_input('Only strikes above:', value='')
    strike_abs_end = st.sidebar.text_input('Only strikes under:', value='')
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




    if st.button('Find best option for this scenario'):
        with st.spinner('Finding best options. Please wait.'):
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
            #st.write((f"1/7: {end_time - start_time:.2f} seconds")) 
            #st.progress(10, text="Looking for all available options")

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
            #st.write((f"2/7: {end_time - start_time:.2f} seconds")) 

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
            #st.write((f"3/7: {end_time - start_time:.2f} seconds")) 

            start_time = time.time()

            # Merge the two dataframes on their index
            merged_df = api_df.join(calculator_results_df, on=api_df.index)


            # Display the merged DataFrame in the app
            #st.write('Merged DataFrame')
            #st.dataframe(merged_df)

            end_time = time.time()
            #st.write((f"4/7: {end_time - start_time:.2f} seconds")) 

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
            #st.write((f"5/7: {end_time - start_time:.2f} seconds")) 


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
            #st.write((f"6/7: {end_time - start_time:.2f} seconds")) 


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
                ),
                hovertemplate='<b>WKN:</b> %{text}<br>' +
                            '<b>Investment Value:</b> %{x}<br>' +
                            '<b>Basispreis:</b> %{y}<br>',
                text=merged_results_df_final['wkn'],
                name=''
            ))


            # Display the merged DataFrame in the app
            #st.write('Final Results')
            #st.dataframe(merged_results_df_final)
            end_time = time.time()
            #st.write((f"7/7: {end_time - start_time:.2f} seconds")) 


            # Set the title and axis labels
            fig.update_layout(title='Scatter Plot', xaxis_title='Return', yaxis_title='Strike')



            # Add a vertical line at the value of the S&P 500 index
            fig.add_hline(y=sp500_value, line_dash="dash", line_color="green", annotation_text=f"S&P 500 Value: {sp500_value:.2f}")

            # Add a vertical line at the value of the S&P 500 index
            fig.add_hline(y=change, line_dash="dash", line_color="blue", annotation_text=f"Change: {change}")

            fig.add_vline(x=100, line_dash="dash", line_color="blue")



            # Find the row with the maximum Investment Value
            max_row = merged_results_df_final.loc[merged_results_df_final['Investment Value'].idxmax()]


            #st.write(max_row)

            # Get the entityValue for the row with the maximum Investment Value
            max_entity_value = max_row['entityValue']

            # Get a list of unique entityValues
            entity_values = merged_results_df_final['entityValue'].unique()
            
            percent_gain = ((max_row['Investment Value'] / investment)*100)
            
            
            col4, col5 = st.columns(2)
            col4.success(f"For a move of {difference:.2f}% on the {ticker_name} (from {current_price:.0f} to {change:.0f}) in the space of {days} days, the most efficient option is:")
            col5.metric(label=f"Strike: {max_row['Basispreis']}", value=f"Return: {max_row['Investment Value']:.0f}€", delta=f"{percent_gain:.0f}%")
                
            with st.expander(f"Details for {max_row['wkn']}"):
                
                st.write(max_row['urls.WEBSITE'])
                st.write(f"Ask: {max_row['askPriceInstrument']:.2f}€")
                st.write(f"Bid: {max_row['bidPriceInstrument']:.2f}€")
                spread = -(1-(((max_row['bidPriceInstrument']/max_row['askPriceInstrument'])))*100)
                st.write(f"Spread: {spread:.2f}%")
                st.write(f"Expiration: {max_row['Bewertungstag']}")
            
            
            # Display the plot in the app
            st.plotly_chart(fig, use_container_width=False)

            # Create a dictionary that maps each entityValue to its corresponding "Basispreis", "Investment Value", and "Aufgeld"
            entity_info = {}
            for entity_value in entity_values:
                row = merged_results_df_final.loc[merged_results_df_final['entityValue'] == entity_value]
                basispreis = row['Basispreis'].iloc[0]
                investment_value = row['Investment Value'].iloc[0]
                aufgeld = row['Aufgeld'].iloc[0]
                entity_info[entity_value] = (basispreis, investment_value, aufgeld)

            # Set the selected entityValue to the entityValue with the highest Investment Value
            selected_entity_value = max_entity_value

            # Create a dropdown widget to select the entityValue
            #selected_entity_value = st.selectbox('Select Entity Value', entity_values, index=int(np.where(entity_values==max_entity_value)[0][0]))

            # Get the corresponding "Basispreis", "Investment Value", and "Aufgeld" for the selected entityValue
            #basispreis, investment_value, aufgeld = entity_info[selected_entity_value]

            # Display the "Basispreis", "Investment Value", and "Aufgeld"
            #st.write(f'Basispreis: {basispreis}, Investment Value: {investment_value}, Aufgeld: {aufgeld}')



            start_time = time.time()





            # Define the range of dates to calculate for
            start_date_matrix = datetime.today()
            #end_date_matrix = start_date_matrix + timedelta(days=10)
            date_range_matrix = pd.date_range(start_date_matrix, end_date)



            # Define the range of prices to calculate for
            price_range_matrix = range(target, int(current_price), 50)

            # Get the entityValue for the row with the maximum Investment Value
            #max_entity_value = max_row['entityValue']

            # Initialize an empty DataFrame to store the results
            results_df = pd.DataFrame(columns=['dateCalculation', 'priceUnderlying', 'calculatedPriceInstrument'])

            for date in date_range_matrix:
                for price in price_range_matrix:
                    url = f'https://api.onvista.de/api/v1/derivatives/{selected_entity_value}/calculatorResult'
                    params = {
                        'dateCalculation': date.strftime('%Y-%m-%d'),
                        'exchangeRate': exchange_rate,
                        'interestInstrument': interest_instrument,
                        'priceUnderlying': price,
                        'volatility': volatility_instrument
                    }
                    response = requests.get(url, params=params)
                    if response.status_code == 200:
                        data = response.json()
                        try:
                            calculated_price = data['calculatedPriceInstrument']
                        except KeyError:
                            st.warning(f"Missing 'calculatedPriceInstrument' key in API response")
                            st.write(data)
                            continue
                        results_df = results_df.append({
                            'dateCalculation': date,
                            'priceUnderlying': price,
                            'calculatedPriceInstrument': calculated_price
                        }, ignore_index=True)
                    else:
                        st.warning(f"Failed to retrieve data for {date.strftime('%Y-%m-%d')} and price {price}")
                        st.write(response.text)

            end_time = time.time()
            #st.write((f"7/8: {end_time - start_time:.2f} seconds"))

            # Get the row for the selected entityValue
            selected_entity_row = merged_results_df_final.loc[merged_results_df_final['entityValue'] == selected_entity_value]

            # Get the corresponding "Basispreis", "Investment Value", and "Aufgeld" for the selected entityValue
            basispreis = selected_entity_row['Basispreis'].iloc[0]
            investment_value = selected_entity_row['Investment Value'].iloc[0]
            aufgeld = selected_entity_row['Aufgeld'].iloc[0]


            # Get the ask price for the selected entityValue
            price_paid = selected_entity_row['askPriceInstrument'].iloc[0]
            amount_stock = (investment / price_paid)
            
            #st.write(investment_value)
            #st.write(price_paid)
            #st.write(amount_stock)


            # Pivot the results to create the matrix table
            matrix_table = results_df.pivot(index='priceUnderlying', columns='dateCalculation', values='calculatedPriceInstrument')

            # Multiply the values in the matrix table by the amount_stock
            matrix_table = matrix_table.applymap(lambda x: ((x*amount_stock)-100))

            # Sort the rows in descending order based on the sum of each row
            matrix_table = matrix_table.loc[matrix_table.sum(axis=1).sort_values(ascending=False).index]

            # Format the X-axis labels to show only YYYY-MM-DD
            matrix_table.columns = pd.to_datetime(matrix_table.columns).strftime('%Y-%m-%d')


            fig, ax = plt.subplots(figsize=(16, 9))
            sns.heatmap(matrix_table, cmap='Greens', annot=True, fmt='.0f', annot_kws={"size": 11})
            plt.title('Matrix Table')
            plt.xlabel('Date Calculation')
            plt.ylabel('Price Underlying')
            ax.invert_yaxis()

            # Save the plot to a file
            filename = 'heatmap.png'
            fig.savefig(filename)

            
            # Display the file in Streamlit
            st.image(filename)





            total_end_time = time.time()
            #st.write((f"Total time: {total_end_time - total_start_time:.2f} seconds")) 



with tab3:
    st.write("WIP")
    
    
with tab4:
    st.write("**Known errors**")
    st.caption("- Probabilities for Calls are wrong")
    st.caption("- When loading a big history range the app slows down")
    st.caption("- DAX options fail more often because the ammount of options available is huge compared to the SP500, and should be filtered by using the filter parameters")    
    st.caption("- The spread calculation is wrong on the option details")    
    st.caption("- EU options expire on Fridays. The calculator only works until -1 of Fridays, so it is advised to choose deadlines as Thursdays ")
    st.caption("- Caching is not implemented, some UI selections will restart the search, such as Investment, which makes no sense")
    st.caption("- Different Expiration dates represented with color on the scatter plot is glitchy")         
    
    st.write("**Todo**")
    st.caption("- Merge the columns of the probabilities with the main graph, so the user can see directly on the graph, the areas of move with more probability")
    st.caption("- Improve chart zoom")
    st.caption("- Allow user to define target+deadline directly on the graph itself")    
    st.caption("- Allow user to see calculation matrix directly by selecting a point on the scatter plot")
    st.caption("- Compare two options calculation matrix side by side")    
    
