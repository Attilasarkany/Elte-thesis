import pandas as pd
from datetime import timedelta
import numpy as np
import scipy.stats as stats
from pyhht.emd import EMD
from sklearn.cluster import KMeans
import os
from datetime import datetime
from pyts.decomposition import SingularSpectrumAnalysis
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas_ta as pta
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler
import matplotlib.pyplot as plt
import webbrowser
import os
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler


def ad_clustering(X_train, k,metric="dtw", seed=42):
    import numpy as np
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.clustering import TimeSeriesKMeans, silhouette_score
    print(f"Using metric: {metric} for clustering.")
    model = TimeSeriesKMeans(
        n_clusters=k,
        n_init=2,
        metric=metric,
        verbose=True,
        max_iter_barycenter=5,
        tol=1e-1,
        random_state=seed,
        n_jobs=-1
    )
    y_pred = model.fit_predict(X_train)
    if len(np.unique(y_pred)) > 1:
        sil_score = silhouette_score(X_train, y_pred, metric=metric)
    else:
        sil_score = -1  
    return (model, y_pred, sil_score)



def preprocess(data, use_smote=True, resample_threshold=0.1, scaling_method='standard', shuffle=False, sort_by=None):
    input = data.copy()
    input = input.sort_index()
    input['Final_Target'] = input['Final_Target'].astype(int)
    '''
    
    for col in input.select_dtypes(include=['float64']).columns:
        input[col] = input[col].astype('float32')
    '''

    split_index = int(len(input) * 0.7)
    train = input[:split_index]
    test = input[split_index:]
    #

    # sorting the data
    if shuffle:
        train = train.sample(frac=1)
    else:
        if sort_by == 'date':
             train = train.sort_index(level=0)
        elif sort_by == 'index_ticker':
            # Sorting by both 'Date' and 'Ticker' indexes
            train = train.reset_index().sort_values(by=['index', 'Ticker']).set_index('index')
            train.index.name = None
    train.set_index(['Ticker'], append=True, inplace=True)
    test.set_index(['Ticker'], append=True, inplace=True)
    test_index = test.index
    # Separating features and target
    feature_names = train.drop(['Target', 'Quarter', 'Adj Close', 'Final_Target'], axis=1).columns.tolist()

    X_train = train.drop(['Target', 'Quarter','Adj Close','Final_Target'], axis=1)
    y_train = train['Final_Target']
    X_test = test.drop(['Target','Quarter','Adj Close','Final_Target'], axis=1)
    y_test = test['Final_Target']

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    if scaling_method == 'standard':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train) 
        X_test = scaler.transform(X_test)
    elif scaling_method == 'power':
        pt = PowerTransformer()
        X_train = pt.fit_transform(X_train)
        X_test = pt.transform(X_test)
    elif scaling_method == 'minmax':
        min_max_scaler = MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)
        X_test = min_max_scaler.transform(X_test)
    print("Count of label before resample'1':", np.sum(y_train == 1))
    print("Count of label  before resample '0':", np.sum(y_train == 0))
    # Use resampling only on the training
    if use_smote:
        resampler = SMOTE(sampling_strategy=resample_threshold)
    else:
        resampler = RandomOverSampler(sampling_strategy=resample_threshold)
    X_train, y_train = resampler.fit_resample(X_train, y_train)
    print("Count of label after resample'1':", np.sum(y_train == 1))
    print("Count of label  after resample '0':", np.sum(y_train == 0))
    return X_train, X_test, y_train, y_test,test_index,feature_names

def select_tickers_with_history(data, end_date, years_required, trading_days_per_year=352):
    """
    select_tickers_with_history(OHCL, '2024-01-01', 20)
    """
    data.index = pd.to_datetime(data.index) 
    end_date = pd.to_datetime(end_date)
    start_date = end_date - timedelta(days=years_required * 352) 

    selected_tickers = []
    for ticker in data['Ticker'].unique():
        series = data[(data['Ticker'] == ticker) & (data.index >= start_date) & (data.index < end_date)]['Adj Close']

        # Calculate the number of years with data
        num_years = series.notna().sum() / trading_days_per_year

        # Check if the series has at least 'years_required' years of non-missing data
        if num_years >= years_required:
            selected_tickers.append(ticker)
    
    return selected_tickers




def X_Years_full_data(data,year):
    '''
    252 napos trading evet feltetelezve szedjuk le azokat amelyeknel
    van x year full data
    
    '''
    selected_tickers = []
    for ticker in data.Ticker.unique():
        series = data[data['Ticker']==ticker]['Adj Close']

        num_years = series.notna().sum() / 252  

        if num_years >= year:
            selected_tickers.append(ticker)
    return selected_tickers


def EMD_Decomposition(data,price):
    ticker_features = {}

    for ticker in data.Ticker.unique():
        series = data[data.Ticker == ticker][price]
        decomposer = EMD(series)
        imfs = decomposer.decompose()
        
        # Find the index where the mean first significantly departs from zero using T-test
        significant_change_point = None
        for i in range(1, len(imfs)):
            t_stat, p_value = stats.ttest_1samp(np.cumsum(imfs[i-1]), 0)
            if p_value < 0.05:
                significant_change_point = i
                break
        
        if significant_change_point is not None:
            # Separate the components based on the significant change point
            high_frequency_imfs = imfs[:significant_change_point]
            
            # Calculate the high-frequency component
            high_frequency_component = np.sum(high_frequency_imfs, axis=0)
            
            # Extract features from the high-frequency component
            features = [np.mean(high_frequency_component), np.std(high_frequency_component)]
            
            # Store features in the dictionary
            ticker_features[ticker] = features

    X = np.array(list(ticker_features.values()))
    return X,ticker_features

'''
Furi transformacio.
sax vs dynamic wrapping

'''
'''
def SSA_decomposition(data, window_sizes, main_comp, save_components=False):
    ticker_features = {}
    data = data.reset_index().sort_values(by=['index', 'Ticker']).set_index('index')
    data.index.name = None  # Remove the index name

    for ticker in data['Ticker'].unique():
        series = data[data['Ticker'] == ticker]['Adj Close'].values
        ticker_features_list = []

        for window_size in window_sizes:
            ssa = SingularSpectrumAnalysis(window_size=window_size)  
            components = ssa.fit_transform(series.reshape(1, -1))[0]
            
            if save_components:
                # Save the raw components directly
                ticker_features_list.append(components[:main_comp])  # Save only the main components
            else:
                # Calculate the main trend from the first 'main_comp' components
                main_trend = np.sum(components[:main_comp], axis=0)
                std_trend = np.std(main_trend)
                mean_trend = np.mean(main_trend)
                
                ticker_features_list.extend([mean_trend, std_trend])

        ticker_features[ticker] = ticker_features_list

    X = np.array(list(ticker_features.values()))
    
    return X, ticker_features
'''



def SSA_decomposition(data,window_size=30, standardize=False):
        
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from pyts.decomposition import SingularSpectrumAnalysis

    data.index = pd.to_datetime(data.index)
    data = data.reset_index().sort_values(by=['index', 'Ticker']).set_index('index')
    data.index.name = None  # Remove the index name for clarity

    #groups = [np.arange(i, i + 5) for i in range(0, 11, 5)]
    ticker_features = {}
    scaler = StandardScaler() 

    for ticker in data['Ticker'].unique():
        series = data[data['Ticker'] == ticker]['Adj Close']
        weekly_series = series.resample('W').mean().dropna()  # Weekly mean, dropping NA for clean data


        ssa = SingularSpectrumAnalysis(window_size=window_size, groups='auto')
        components = ssa.fit_transform(weekly_series.values.reshape(1, -1))[0]
        first_group_components = components[0, :]  # Extract the first row (first group)

        # Decide whether to standardize the components based on the function parameter
        if standardize:
            standardized_components = scaler.fit_transform(first_group_components.reshape(-1, 1)).flatten()
            ticker_features[ticker] = standardized_components
        else:
            ticker_features[ticker] = first_group_components

    # Create a DataFrame from the dictionary of features
    feature_df = pd.DataFrame.from_dict(ticker_features, orient='index')

    return feature_df




def extract_data_for_tickers(data, tickers, years_back, desired_rows, base_year=2024):



    start_year = base_year - years_back

    quarters = [f"{year}Q{q}" for year in range(start_year, base_year) for q in range(1, 5)]

    final_data = pd.DataFrame()

    for ticker in tickers:
        last_12_quarters = quarters[:4]
        if not data.loc[(ticker,), last_12_quarters].isna().all().all():
            filtered_data = data.loc[(ticker, desired_rows), quarters]
            final_data = pd.concat([final_data, filtered_data])

    return final_data



def process_files_and_extract_tickers(directory, tickers):

    combined_data = pd.DataFrame()

    # Iterate over the files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".parquet"):
            file_path = os.path.join(directory, filename)
            
            # Read the parquet file
            data = pd.read_parquet(file_path, engine='pyarrow')

            # Find the intersecting tickers
            data_tickers = set(data['Ticker'].unique())
            valid_tickers = tickers.intersection(data_tickers)

            # Filter data for the intersecting tickers
            filtered_data = data[data['Ticker'].isin(valid_tickers)]

            # Concatenate the results
            combined_data = pd.concat([combined_data, filtered_data])

    return combined_data



def process_files_and_extract_tickers(directory, tickers, base_date, years_back):

    combined_data = pd.DataFrame()

    # Iterate over the files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".parquet"):
            file_path = os.path.join(directory, filename)
            
            # Read the parquet file
            data = pd.read_parquet(file_path, engine='pyarrow')
            data.index = pd.to_datetime(data.index)

            end_date = pd.to_datetime(base_date)
            start_date = end_date - pd.DateOffset(years=years_back)

            # Find the intersecting tickers
            data_tickers = set(data['Ticker'].unique())
            valid_tickers = tickers.intersection(data_tickers)

            # Filter data for the intersecting tickers
            filtered_data = data[data['Ticker'].isin(valid_tickers)& (data.index >= start_date) & (data.index < end_date)]

            # Concatenate the results
            combined_data = pd.concat([combined_data, filtered_data])

    return combined_data


def reconstruct_spy1500():
    data1=pd.read_csv('C:/Users/Attila/Desktop/ELTE/Master_thesis/sp-400-index-02-26-2024.csv')
    data2=pd.read_csv('C:/Users/Attila/Desktop/ELTE/Master_thesis/sp-500-index-02-26-2024.csv')
    data3=pd.read_csv('C:/Users/Attila/Desktop/ELTE/Master_thesis/sp-600-index-02-26-2024.csv')
    symbols1 = data1['Symbol'].tolist()
    symbols2 = data2['Symbol'].tolist()
    symbols3 = data3['Symbol'].tolist()

    # Concatenate the lists
    all_symbols = symbols1 + symbols2 + symbols3


    return all_symbols


def process_directories_and_tickers(directories, provided_tickers, end_date, years_required):
    intersected_tickers = set()
    missing_tickers = set(provided_tickers)
    filtered_data = pd.DataFrame()

    # Convert end_date to a datetime object and calculate the start date
    end_date = pd.to_datetime(end_date)
    start_date = end_date - timedelta(days=years_required * 352) 

    for directory in directories:
        directory_tickers = set()

        # Iterate over the files in the directory
        for filename in os.listdir(directory):
            if filename.endswith(".parquet"):
                file_path = os.path.join(directory, filename)

                data = pd.read_parquet(file_path, engine='pyarrow')
                data.index = pd.to_datetime(data.index, format='%m/%d/%Y')
                selected_tickers= select_tickers_with_history(data,end_date, 25)

                file_tickers = set(selected_tickers)
                directory_tickers.update(file_tickers)

        intersected_tickers.update(directory_tickers.intersection(provided_tickers))
        missing_tickers.difference_update(directory_tickers)

    # Process each directory again to filter data
    for directory in directories:
        for filename in os.listdir(directory):
            if filename.endswith(".parquet"):
                file_path = os.path.join(directory, filename)
                data = pd.read_parquet(file_path, engine='pyarrow')
                data.index = pd.to_datetime(data.index, format='%m/%d/%Y')

                # Filter for intersected tickers and the date range
                data = data[(data['Ticker'].isin(intersected_tickers)) & (data.index >= start_date) & (data.index <= end_date)]
                filtered_data = pd.concat([filtered_data, data])

    return list(intersected_tickers), list(missing_tickers), filtered_data


def process_directories(directories, provided_tickers, years_back, desired_rows, base_year=2024):
    '''
    Extract fundamental data by using extract_data_for_tickers
    '''
    combined_data = pd.DataFrame()

    for directory in directories:
        for filename in os.listdir(directory):
            if filename.endswith(".parquet"):
                file_path = os.path.join(directory, filename)
                data = pd.read_parquet(file_path)

                # Get tickers from the first level of the MultiIndex
                data_tickers = data.index.get_level_values(0).unique()

                # Intersect with provided tickers
                valid_tickers = set(provided_tickers).intersection(data_tickers)

                # Extract and filter data for valid tickers
                extracted_data = extract_data_for_tickers(data, valid_tickers, years_back, desired_rows, base_year)
                
                # Concatenate the results
                combined_data = pd.concat([combined_data, extracted_data])

    return combined_data


def DataCleaning(data):
    return




def process_directories(directories, provided_tickers, years_back, desired_rows, base_year=2024):
    combined_data = pd.DataFrame()

    for directory in directories:
        for filename in os.listdir(directory):
            if filename.endswith(".parquet"):
                file_path = os.path.join(directory, filename)
                print(file_path)
                data = pd.read_parquet(file_path)

                # Get tickers from the first level of the MultiIndex
                data_tickers = data.index.get_level_values(0).unique()

                # Intersect with provided tickers
                valid_tickers = set(provided_tickers).intersection(data_tickers)
                try:
                # Extract and filter data for valid tickers
                    extracted_data = extract_data_for_tickers(data, list(valid_tickers), years_back, desired_rows, base_year)
                    
                    # Concatenate the results
                    combined_data = pd.concat([combined_data, extracted_data])
                except:
                    print(f"No intersecting tickers found in file: {filename}")
                    continue  # Skip to the next file

    return combined_data




def Crash_Identification(data, threshold_percentage, time_window_trading_days):
    data_copy = data.copy()

    # Sort data if it's not already sorted
    data_copy.reset_index(inplace=True)
    data_copy.rename(columns={'index': 'Date'}, inplace=True)
    data_copy.sort_values(by=['Ticker', 'Date'], inplace=True)

    # Initialize a column for crash identification
    data_copy['Crash'] = False

    # Iterate over each ticker
    unique_tickers = data_copy['Ticker'].unique()
    for ticker in unique_tickers:
        # Select data for the current ticker
        ticker_data = data_copy[data_copy['Ticker'] == ticker]

        # Calculate percentage change over the specified window
        ticker_data['Pct_Change'] = ticker_data['Adj Close'].pct_change(periods=time_window_trading_days) * 100

        # Identify crashes
        crash_condition = ticker_data['Pct_Change'] < threshold_percentage
        crash_indices = ticker_data[crash_condition].index
        data_copy.loc[crash_indices, 'Crash'] = True

    return data_copy





def calculate_drawdowns(df):
    # Ensure 'Date' is the index
    data_copy=df.copy()
    data_copy.reset_index(inplace=True)
    data_copy.rename(columns={'index': 'Date'}, inplace=True)
    data_copy.set_index('Date',inplace=True)


    # Initialize a list to store drawdown data for each ticker
    drawdowns_list = []

    # Group by 'Ticker' and calculate drawdowns
    for ticker, group in df.groupby('Ticker'):
        # Identifying local maxima and minima in 'Adj Close'
        pmin_pmax = (group['Adj Close'].diff(-1) > 0).astype(int).diff()
        pmax = pmin_pmax[pmin_pmax == 1]
        pmin = pmin_pmax[pmin_pmax == -1]

        # Adjusting first and last points
        if not pmin.empty and not pmax.empty:
            if pmin.index[0] < pmax.index[0]:
                pmin = pmin.drop(pmin.index[0])
            if pmin.index[-1] < pmax.index[-1]:
                pmax = pmax.drop(pmax.index[-1])

            # Calculating drawdowns
            dd = (np.array(group['Adj Close'].loc[pmin.index]) - np.array(group['Adj Close'].loc[pmax.index])) / np.array(group['Adj Close'].loc[pmax.index])
            dur = [np.busday_count(p1.date(), p2.date()) for p1, p2 in zip(pmax.index, pmin.index)]

            # Creating a DataFrame for the ticker's drawdowns
            d = {'Date': pmax.index, 'Ticker': ticker, 'Drawdown': dd, 'Duration': dur}
            df_d = pd.DataFrame(d)
            drawdowns_list.append(df_d)

    # Combine all the drawdown data into a single DataFrame
    all_drawdowns_df = pd.concat(drawdowns_list)
    return all_drawdowns_df



def identify_swing_points(df, window=30):

    stock_data = df.copy()
    stock_data['Swing_High'] = stock_data['Adj Close'][argrelextrema(stock_data['Adj Close'].values, np.greater_equal, order=window)[0]]

    # Identify local minima (swing lows)
    stock_data['Swing_Low'] = stock_data['Adj Close'][argrelextrema(stock_data['Adj Close'].values, np.less_equal, order=window)[0]]


    return stock_data





def mark_threshold_cross(df, threshold):
    df['Threshold_Cross'] = False
    last_high_index = None

    for index, row in df.iterrows():
        # If there's a swing high, remember its index
        if not np.isnan(row['Swing_High']):
            last_high_index = index

        # If there's a swing low and we have seen a swing high before
        if not np.isnan(row['Swing_Low']) and last_high_index is not None:
            # Get the last swing high value
            last_high = df.at[last_high_index, 'Swing_High']

            # Check if the difference is below the threshold
            if (row['Swing_Low'] - last_high) / last_high < threshold:
                # Update the Threshold_Cross at the last high index
                df.at[last_high_index, 'Threshold_Cross'] = True

    return df


def identify_swing_points_and_mark_threshold_cross_combined(df, window=30, threshold=-0.1):
    """
    Combined function to identify swing points and mark threshold crosses 
    for each group of data per ticker.

    :param df: DataFrame containing stock data with a 'Ticker' column.
    :param window: Number of points to consider for identifying swing points.
    :param threshold: The specified threshold for the difference.
    :return: DataFrame with 'Swing_High', 'Swing_Low', and 'Threshold_Cross' columns added for each ticker.
    """
    # Define a function that applies both swing point identification and threshold marking
    # argrelextrema goes through all of the indexes and check the observations in a window
    def process_group(sub_df):
        sub_df = identify_swing_points(sub_df, window)
        sub_df = mark_threshold_cross(sub_df, threshold)
        return sub_df

    # Group by 'Ticker' and apply the combined function
    # But first make sure we have ordered data by date and ticker
    df = df.reset_index().sort_values(by=['index', 'Ticker']).set_index('index')
    df.index.name = None  # Remove the index name

    grouped = df.groupby('Ticker')
    return grouped.apply(process_group)



def plot_stock_data_with_swing_points(df, ticker, target):

    df.index = pd.to_datetime(df.index)
    df_ticker = df[df['Ticker'] == ticker]

    plt.figure(figsize=(15, 7))

    # Plotting Close Prices
    plt.plot(df_ticker.index, df_ticker['Close'], label='Close Price', color='blue')

    # Plot Swing Highs and Lows
    plt.scatter(df_ticker.index[df_ticker['Swing_High'].notna()],
                df_ticker['Swing_High'][df_ticker['Swing_High'].notna()],
                color='green', marker='^', label='Swing Highs')

    plt.scatter(df_ticker.index[df_ticker['Swing_Low'].notna()],
                df_ticker['Swing_Low'][df_ticker['Swing_Low'].notna()],
                color='red', marker='v', label='Swing Lows')

    # Highlight Threshold Cross where True
    threshold_cross_indices = df_ticker.index[df_ticker[target] == True]
    for idx in threshold_cross_indices:
        plt.axvline(x=idx, color='purple', linestyle='--', alpha=0.7)

    start, end = df_ticker.index.min(), df_ticker.index.max()
    years = pd.date_range(start=start, end=end, freq='YS')  # 'YS' stands for year start frequency
    plt.xticks(years, [year.strftime('%Y') for year in years], rotation=45)

    plt.title(f"Close Price with Swing Highs, Lows, and Threshold Cross for {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

def replace_nan_inf_with_mean(data, desired_rows):

    # Remove duplicate indices if any
    data = data[~data.index.duplicated()]

    for col in data.columns:
        for row in desired_rows:
            # Extract the specific data segment with its original index
            characteristic_data = data.xs(row, level=1, axis=0)[col]

            # Calculate the mean, excluding inf and -inf values
            mean_value = characteristic_data.replace([np.inf, -np.inf], np.nan).dropna().mean()

            # Replace inf, -inf, and nan values in the segment
            replaced_data = characteristic_data.replace([np.inf, -np.inf, np.nan], mean_value)

            # Ensure the replaced data segment retains the original index
            replaced_data.index = data.loc[(slice(None), row), col].index

            # Assign the modified data back to the DataFrame
            data.loc[(slice(None), row), col] = replaced_data

    return data


def calculate_macd(data, price_column, slow=26, fast=12, signal=9):
    """
    Calculate the Moving Average Convergence Divergence (MACD) for a given price column in a DataFrame.

    Parameters:
    data (pd.DataFrame): DataFrame containing the stock price data.
    price_column (str): Column name for the price data to use for MACD calculation.
    slow (int): The number of periods for the slow EMA.
    fast (int): The number of periods for the fast EMA.
    signal (int): The number of periods for the signal line.

    Returns:
    pd.DataFrame: DataFrame with MACD and Signal line.
    """
    # Select the price column from the DataFrame
    prices = data[price_column]

    # Calculate the fast and slow Exponential Moving Averages (EMA)
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()

    # Calculate the MACD line
    macd = exp1 - exp2

    # Calculate the Signal line
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    data['MACD'] = macd
    data['Signal_Line'] = signal_line

    return data


def calculate_rsi(data, price_column, period=14):
    """
    Calculate the Relative Strength Index (RSI) for a given price column in a DataFrame.

    Parameters:
    data (pd.DataFrame): DataFrame containing the stock price data.
    price_column (str): Column name for the price data to use for RSI calculation.
    period (int): The period to use for calculating RSI.

    Returns:
    pd.Series: RSI values.
    """
    # Select the price column from the DataFrame
    prices = data[price_column]

    # Calculate price changes
    delta = prices.diff()

    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate the Exponential Moving Average (EMA) for gains and losses
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    # Calculate the Relative Strength (RS)
    rs = avg_gain / avg_loss

    # Calculate the RSI
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi


    return data



def calculate_rolling_mean_changes(data, price, windows):
    """
    Calculate and append rolling mean changes for different window sizes to the original DataFrame.

    Parameters:
    data (pd.DataFrame): DataFrame with the stock data.
    price (str): Column name for the price data.
    windows (list of int): List of window sizes in days.

    Returns:
    pd.DataFrame: Original DataFrame with appended rolling mean changes for each window size.
    """
    # Ensure the DataFrame index is a DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    for window in windows:
        # Calculate rolling mean and append to the DataFrame
        data[f'{window}d_rolling_mean_{price}'] = data[price].rolling(window=window).mean()

    return data


def calculate_rolling_pct_change(data, windows):
    """
    Calculates rolling percentage change for specified window sizes for each ticker,
    applying the calculation only to numeric columns.

    Parameters:
    data (pd.DataFrame): DataFrame containing the data with 'Ticker', 'Quarter', and other columns.
    windows (list): List of integers representing window sizes for the rolling percentage change.

    Returns:
    pd.DataFrame: Original DataFrame with new rolling percentage change columns added.
    """

    # Convert 'Quarter' to a period (year and quarter)
    data['Quarter'] = pd.PeriodIndex(data['Quarter'], freq='Q')

    # Sort the DataFrame
    data = data.sort_values(by=['Ticker', 'Quarter'])

    # Filter numeric columns for rolling operation
    numeric_cols = data.select_dtypes(include='number').columns.tolist()

    # Dictionary to hold rolling percentage change results for each window size
    rolling_pct_changes = {window: pd.DataFrame() for window in windows}

    # Loop over each group of tickers
    for ticker, group in data.groupby('Ticker'):
        group = group.set_index('Quarter')  # Set 'Quarter' as the index for rolling calculation
        
        # Loop through each window size
        for window in windows:
            # Apply rolling percentage change only to numeric columns
            rolling_pct_change_ticker = group[numeric_cols].pct_change(periods=window)
            rolling_pct_change_ticker = rolling_pct_change_ticker.add_suffix(f'_RollingPctChangeWindow{window}')
            rolling_pct_change_ticker['Ticker'] = ticker  # Add ticker column back for later merge
            
            # Append the results
            rolling_pct_changes[window] = pd.concat([rolling_pct_changes[window], rolling_pct_change_ticker])

    # Merge the rolling percentage change results back to the original DataFrame
    for window in windows:
        data = data.merge(rolling_pct_changes[window], on=['Ticker', 'Quarter'], how='left')

    return data.reset_index(drop=True)


def plot_probability_vs_adjclose(data, ticker, shift_val):
    # Construct the probability column name based on the shift value
    prob_column = f'Probabilities_log_{shift_val}'

    # Filter for the specified ticker
    ticker_data = data[data['Ticker'] == ticker]

    # Create traces for the plot
    trace1 = go.Scatter(
        x=ticker_data.index,
        y=ticker_data[prob_column],  # Updated to use the constructed column name
        mode='lines',
        name='Probability of Crash',
        line=dict(color='Crimson')
    )

    trace2 = go.Scatter(
        x=ticker_data.index,
        y=ticker_data['Adj Close'],
        mode='lines',
        name='Adj Close',
        line=dict(color='RoyalBlue'),
        yaxis='y2'
    )

    # Define the layout
    layout = go.Layout(
        title=f'{ticker}: Probability of Crash vs Adj Close (Shift: {shift_val})',
        xaxis=dict(
            title='Date',
            tickformat='%b %Y',
            tickangle=-45
        ),
        yaxis=dict(title='Probability of Crash'),
        yaxis2=dict(
            title='Adj Close',
            overlaying='y',
            side='right'
        )
    )

    # Create the figure and show it
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    fig.show()

    # Save the plot as an HTML file
    file_path = f'{ticker}_plot_shift_{shift_val}.html'
    fig.write_html(file_path)

    # Open the HTML file in the default web browser
    webbrowser.open('file://' + os.path.realpath(file_path))




'''
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Downloading historical stock price data for Apple Inc. (BAC)
data = yf.download("BAC", start="2000-01-01", end="2023-01-01")

# Calculate daily percentage change
data['pct_change'] = data['Adj Close'].pct_change()

# Define the window size for comparing the changes
window_size = 30  # 30 trading days

# Calculate the percentage change over the specified window size
data['rolling_pct_change'] = data['Adj Close'].pct_change(periods=window_size)

# Identify where the percentage change is less than -20%
data['crash_signal'] = data['rolling_pct_change'] <= -0.20

# Plot the results with vertical lines for crash signals
plt.figure(figsize=(14, 7))
plt.plot(data['Adj Close'], label='Adjusted Close Price', color='blue')  # Plot the closing prices

# Add vertical lines for crash signals
for date in data[data['crash_signal']].index:
    plt.axvline(x=date, color='red', linestyle='--', label='Crash Signal' if date == data[data['crash_signal']].index[0] else "")

plt.title('BAC Stock Price and Crash Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

'''

'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Assuming your data is loaded into a DataFrame 'df'
data = {
    'Date': ['1999-11-28', '1999-12-05', '1999-12-12'],
    'A': [25.1225, 26.534286, 27.477143],
    'AA': [56.8275, 60.11, 64.642857],
    'AAON': [0.55845, 0.561457, 0.564243],
    'AAPL': [0.7173, 0.8117, 0.814343],
    'ABCB': [5.765, 5.718571, 5.737143],
    # Add other stocks...
}
df = pd.DataFrame(data)
df.set_index('Date', inplace=True)

# Standardize the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(pivot_df)

# Apply PCA
pca = PCA()
pca.fit(scaled_data)
scores_pca = pca.transform(scaled_data)

# KMeans clustering and WCSS plot
wcss = []
for i in range(1, 10):
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(scores_pca[:, :80])
    wcss.append(km.inertia_)

# Plotting the WCSS to find the optimal number of clusters
plt.figure(figsize=(20, 10))
plt.grid()
plt.plot(range(1, 10), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Determining Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
'''
'''

from tsfresh import extract_features
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

pivot_df=pd.read_csv('pivot_df.csv',index_col='Unnamed: 0', parse_dates=True)

df_long = pivot_df.reset_index().melt(id_vars=['index'], var_name='Ticker', value_name='Price')
df_long.rename(columns={'index': 'timestamp'}, inplace=True)


# Extract features
extracted_features = extract_features(
    df_long, 
    column_id='Ticker', 
    column_sort='timestamp'
)

extracted_features_cleaned=extracted_features
extracted_features_cleaned=extracted_features_cleaned.dropna(axis=1)

scaler = StandardScaler()
extracted_features_cleaned_std = scaler.fit_transform(extracted_features_cleaned)

pca = PCA()
pca.fit(extracted_features_cleaned_std)
scores_pca = pca.transform(extracted_features_cleaned_std)

# KMeans clustering and WCSS plot
wcss = []
for i in range(3, 6):
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(scores_pca[:, :80])
    wcss.append(km.inertia_)
    plt.figure(figsize=(20,10))
plt.grid()
plt.plot(range(1,10),wcss,marker='o',linestyle='--')
plt.xlabel('number of clusters')
plt.ylabel('WCSSS')
# kepet lementeni es a  extracted_features_cleaned_std
    '''