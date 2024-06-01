import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import requests
import warnings
warnings.filterwarnings("ignore")
from pyhht.emd import EMD
import requests
from pyhht.visualization import plot_imfs
import scipy.stats as stats
from sklearn.cluster import KMeans
import parquet
from Methods import select_tickers_with_history,X_Years_full_data,extract_data_for_tickers,process_files_and_extract_tickers,process_directories_and_tickers
from Methods import EMD_Decomposition,SSA_decomposition,process_directories,identify_swing_points,identify_swing_points_and_mark_threshold_cross_combined,plot_stock_data_with_swing_points
from Methods import calculate_rolling_mean_changes,calculate_rolling_pct_change,ad_clustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import plotly.graph_objects as go
import pandas_ta as pta


import pickle
from sklearn.metrics import adjusted_mutual_info_score as ami
from tqdm.notebook import tqdm


# get the spy1500 tickers
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

# Create an instance
spy_1500=reconstruct_spy1500()

# Create a dictionary including the OHCL data locations
directories_OHCL = [
    'C:/Users/Attila/Desktop/Financel Prep Data set/NYSE/OHCL',
    'C:/Users/Attila/Desktop/Financel Prep Data set/Amex/OHCL',
    'C:/Users/Attila/Desktop/Financel Prep Data set/Nasda/OHCL'
]

# get those tickers from spy_1500, which have 20 years trading days non missing Adj closing price
provided_tickers = set(spy_1500) 
intersected, missing,x = process_directories_and_tickers(directories_OHCL, provided_tickers,'2023-12-29',25)

# save it for later usage
#x.to_csv('intersected_tickers_OHCL_25.csv',index_label=False)

'''
THIS IS THE MAIN TESTING DATA. SPY1500 WITH 20 YEARS OHCL
WE NEED TO CLEAN THE FUNDAMENTAL DATA

'''
# OPEN THE MAIN OHCL FILE
intersected_tickers=pd.read_csv('intersected_tickers_OHCL_25.csv')
intersected_tickers_list=intersected_tickers.Ticker.unique().tolist()


# GRABING THE FUNDAMENTALS: WE NEED TO HANDLE THE MISSING VALUES
# Fundamental Dataset
desired_rows = [
'Operating Margin',
'Interest Coverage Ratio',
'Short Term Coverage Ratio',
'Price-to-Free-Cash-Flow',
'Debt-to-Equity Ratio',
'Return on Invested Capital',
'Free Cash Flow to Operating Cash Flow Ratio',
'Tangible Asset Value',
'Price-to-Earnings-Growth',
'Cash Conversion Cycle',
'Operating Cash Flow to Sales Ratio'
]  

# With Amex there is a data problem, but not too much data in there
directories = [
    r'C:\Users\Attila\Desktop\Financel Prep Data set\NYSE\Ratios',
    r'C:\Users\Attila\Desktop\Financel Prep Data set\Nasda\Ratios',
  
]
# get the fundamental data
# we get 25 years of data and droping those were there are only missing values in the beginning of the frame
combined_data = process_directories(directories, intersected_tickers_list, 25, desired_rows) # We drop anex


######## Price explanatory dataset ###########

#########################################################
#########  Dynamic clustering ###########
#####################################################
# only use the training size
# MAIN ###
input = intersected_tickers.copy()
input = input.sort_index()

#split_index_wrapping = int(len(input) * 0.7) # or training data int(len(input) * 0.7*0.4)
#train = input[:split_index_wrapping]
#test = input[split_index_wrapping:]

train=input.loc[:'2018-05-24'] # from notebook
test=input.loc['2018-05-24':] # from notebook


pivot_df = train.pivot_table(values='Adj Close', index=train.index, columns='Ticker')
pivot_df.index = pd.to_datetime(pivot_df.index) 
pivot_df = pivot_df.resample('W').mean() 
pivot_df.dropna(axis=0, how='any', inplace=True)  # Drop rows with any NaN values

scaler = TimeSeriesScalerMeanVariance()
normalized_data = scaler.fit_transform(pivot_df.T.values[:, :, np.newaxis])

# save pivot_dfpivot_df
pivot_df.to_csv('pivot_df.csv',index=True, header=True)

# lets use the silhouette score to decide the best cluster.
# we dont expect too much cause it is a simple model
# n_init=2: took many times to run, for 8 classes
# n_init 3 for 3
# We runned before on the whole data set. 3-5 models are ok
result_dict = {}
for k in tqdm(range(3, 6)): 
    model, y_pred, silhouette_avg = ad_clustering(normalized_data, k)
    result_dict[k] = (model, y_pred, silhouette_avg)
    print(f"For k = {k}, Silhouette Score = {silhouette_avg:.5f}")



ks = list(result_dict.keys())
silhouette_scores = [result_dict[k][2] for k in ks]

plt.figure(figsize=(10, 6))
plt.plot(ks, silhouette_scores, marker='o')
plt.title('Silhouette Score for Different Numbers of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

with open('Silhouette_dynamic_wrapping_class4', 'wb') as f:
    pickle.dump(result_dict, f)
## we have very low silhouette_scores ######
#n_clusters=8 , n_init=5 for all of the clusters
# default 1,its ok
# best cluster =3
model = TimeSeriesKMeans(n_clusters=3, metric="dtw", verbose=True, max_iter=10, random_state=42)
labels = model.fit_predict(normalized_data)
ticker_to_cluster = {ticker: label for ticker, label in zip(pivot_df.columns, labels)}

intersected_tickers['Cluster'] = intersected_tickers['Ticker'].map(ticker_to_cluster)

grouped=intersected_tickers.groupby(['Cluster'])['Ticker'].unique()
for i in grouped:
    print(len(i))


unique_clusters = intersected_tickers['Cluster'].unique()

# Plot mean prices for each cluster
plt.figure(figsize=(10, 6))
for cluster_id in unique_clusters:
    tickers_in_cluster = intersected_tickers[intersected_tickers['Cluster'] == cluster_id]['Ticker'].unique()
    cluster_data = pivot_df[tickers_in_cluster]
    mean_prices = cluster_data.mean(axis=1)
    plt.plot(mean_prices.index, mean_prices, label=f'Cluster {cluster_id}')

plt.title('Mean Prices Over Time for Each Cluster')
plt.xlabel('Date')
plt.ylabel('Mean Price')
plt.legend()
plt.show()
# save the output
intersected_tickers.to_pickle("clusters_best_3_dtw_normalized.pkl")

'''
# save the very obvious group as csv
cluster_2_tickers = intersected_tickers[intersected_tickers['Cluster'] == 2]['Ticker'].unique()
cluster_2_df = pd.DataFrame(cluster_2_tickers, columns=['Ticker'])
cluster_2_df.to_csv('cluster_2_tickers.csv', index=False)
'''

################ Other clustering methods
# Same logic, just we  use SSA 5 components
X=SSA_decomposition(train,15,standardize=False)
#X_array = X.to_numpy()

####
scaler = TimeSeriesScalerMeanVariance()
normalized_data_ssa = scaler.fit_transform(X.T.values[:, :, np.newaxis])
###

# Reshape the numpy array to add an additional dimension
#X_reshaped = X_array.reshape((676, 966, 1)) # need to add an additional dimension(features)
# Silhouette
result_dict_knn = {}
for k in tqdm(range(3, 6)): 
    model, y_pred, silhouette_avg = ad_clustering(normalized_data_ssa, k)
    result_dict_knn[k] = (model, y_pred, silhouette_avg)
    print(f"For k = {k}, Silhouette Score = {silhouette_avg:.5f}")


ks = list(result_dict_knn.keys())
silhouette_scores = [result_dict_knn[k][2] for k in ks]

plt.figure(figsize=(10, 6))
plt.plot(ks, silhouette_scores, marker='o')
plt.title('Silhouette Score for Different Numbers of Clusters:SSA')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score:DTW')
plt.grid(True)
plt.show()


#################
model_ssa = TimeSeriesKMeans(n_clusters=3, metric="dtw", verbose=True, max_iter=10, random_state=42)

labels_ssa = model_ssa.fit_predict(normalized_data_ssa)
ticker_to_cluster_ssa = {ticker: label for ticker, label in zip(pivot_df.columns, labels_ssa)}

intersected_tickers['Cluster'] = intersected_tickers['Ticker'].map(ticker_to_cluster_ssa)

grouped=intersected_tickers.groupby(['Cluster'])['Ticker'].unique()
for i in grouped:
    print(len(i))


unique_clusters = intersected_tickers['Cluster'].unique()

# Plot mean prices for each cluster
plt.figure(figsize=(10, 6))
for cluster_id in unique_clusters:
    tickers_in_cluster = intersected_tickers[intersected_tickers['Cluster'] == cluster_id]['Ticker'].unique()
    cluster_data = pivot_df[tickers_in_cluster]
    mean_prices = cluster_data.mean(axis=1)
    plt.plot(mean_prices.index, mean_prices, label=f'Cluster {cluster_id}')

plt.title('Mean Prices Over Time for Each Cluster')
plt.xlabel('Date')
plt.ylabel('Mean Price')
plt.legend()
plt.show()
# save the output
intersected_tickers.to_pickle("shifted_clustering.pkl")

##### KNN extract feature Ts learn #####
# https://aws.amazon.com/blogs/machine-learning/boost-your-forecast-accuracy-with-time-series-clustering/

from tsfresh import extract_features
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

extraced_time=np.load('extract_feature.npy')
pca = PCA()
pca.fit(extraced_time)
scores_pca = pca.transform(extraced_time)

plt.figure(figsize=(20,10))
plt.grid()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
# Around 80 features are responsible for 90%


distortions = []
silhouette_scores = []
K = range(2, 11)  # Test a range of cluster counts
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scores_pca[:, :80])
    distortions.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(scores_pca[:, :80], kmeans.labels_))

# Plotting the distortions to find the elbow
plt.figure(figsize=(8, 4))
plt.plot(K, distortions, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


plt.figure(figsize=(8, 4))
plt.plot(K, silhouette_scores, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Different k')
plt.show()

optimal_k = 5 
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels_pca = kmeans.fit_predict(scores_pca[:, :80])


unique_labels, counts = np.unique(labels_pca, return_counts=True)

# Print the count of each unique label
for label, count in zip(unique_labels, counts):
    print(f"Cluster {label}: {count} elements")


ticker_to_cluster_pca = {ticker: label for ticker, label in zip(pivot_df.columns, labels_pca)}

intersected_tickers['Cluster'] = intersected_tickers['Ticker'].map(ticker_to_cluster_pca)

grouped=intersected_tickers.groupby(['Cluster'])['Ticker'].unique()
for i in grouped:
    print(len(i))


unique_clusters = intersected_tickers['Cluster'].unique()

# Plot mean prices for each cluster
plt.figure(figsize=(10, 6))
for cluster_id in unique_clusters:
    tickers_in_cluster = intersected_tickers[intersected_tickers['Cluster'] == cluster_id]['Ticker'].unique()
    cluster_data = pivot_df[tickers_in_cluster]
    mean_prices = cluster_data.mean(axis=1)
    plt.plot(mean_prices.index, mean_prices, label=f'Cluster {cluster_id}')

plt.title('Mean Prices Over Time for Each Cluster')
plt.xlabel('Date')
plt.ylabel('Mean Price')
plt.legend()
plt.show()


#################################################################
### Continue the data prep #############
df_processed = identify_swing_points_and_mark_threshold_cross_combined(intersected_tickers, window=30, threshold=-0.2)
plot_stock_data_with_swing_points(df_processed, 'BAC','Threshold_Cross')


# DATA ENRICHING with Technical indicators ###
##############################

df=df_processed.copy()
windows=[5,15,30,45,65,130]
unique_tickers = df['Ticker'].unique()
concatenated_data = pd.DataFrame()

for ticker in unique_tickers:
    ticker_data = df[df['Ticker'] == ticker].copy()
    ticker_data.index = pd.to_datetime(ticker_data.index)

    # Calculate RSI, MACD, and segmented mean changes for prices and volumes
    ticker_data['RSI'] = pta.rsi(ticker_data['Adj Close'], length = 14)
    ticker_data[['MACD','MACD_H','MACD_S']]=pta.macd(ticker_data['Adj Close'],append=True)
    calculate_rolling_mean_changes(ticker_data, 'Adj Close', windows)
    calculate_rolling_mean_changes(ticker_data, 'Volume', windows)


    concatenated_data = pd.concat([concatenated_data, ticker_data])

concatenated_data.index = pd.to_datetime(concatenated_data.index)
concatenated_data['Quarter']= concatenated_data.index.to_period('Q').astype(str)

    

# Changed the data to other format for easier computation
# combined_data is the fundamental data
Fundamental_Final = combined_data.reset_index()
# Using melt to transform the DataFrame
Fundamental_Final= Fundamental_Final.rename(columns={'level_0': 'Ticker', 'level_1': 'Firm Characteristic'})
# Pivoting the table to get Firm Characteristics as separate columns
Fundamental_Final = Fundamental_Final.melt(id_vars=['Ticker', 'Firm Characteristic'], var_name='Quarter', value_name='Value')

Fundamental_Final = Fundamental_Final.pivot_table(index=['Ticker', 'Quarter'], columns='Firm Characteristic', values='Value').reset_index()
Fundamental_Final.columns.name = None
#
Fundamental_Final = Fundamental_Final.sort_values(by=['Ticker', 'Quarter'])
# lets use ffil from the 3rd columns
Fundamental_Final.iloc[:, 2:] = Fundamental_Final.groupby('Ticker').apply(lambda group: group.iloc[:, 2:].ffill())


#Enrich: Create PCT changes thorough windows
Fundamental_Final=calculate_rolling_pct_change(Fundamental_Final,windows=[2,4])


selected_columns = [col for col in Fundamental_Final.columns if 'Rolling' in col or col in ['Ticker', 'Quarter']]

filtered_fundamental = Fundamental_Final[selected_columns]



# Left join 

# change it to month based quarter

concatenated_data['Quarter'] = pd.to_datetime(concatenated_data['Quarter'].str.replace(r'(Q\d)', r'-\1'))
filtered_fundamental['Quarter'] = filtered_fundamental['Quarter'].astype(str)
filtered_fundamental['Quarter'] = pd.to_datetime(filtered_fundamental['Quarter'].str.replace(r'(Q\d)', r'-\1'))

# we need to shift back the quarters with 1. We are doing this cause  i.e the Q1 
# data is visible in the next quarter and we want to avoid look ahead bias.
filtered_fundamental['Quarter'] = filtered_fundamental['Quarter'] + pd.offsets.QuarterBegin(startingMonth=1)

concatenated_data.reset_index(inplace=True)
filtered_fundamental.reset_index(inplace=True)


Fundamental_Price = pd.merge(concatenated_data, filtered_fundamental, on=['Quarter', 'Ticker'], how='left')

#Fundamental_Price.to_csv('Fundamental_Price_Final_25.csv')
Fundamental_Price= pd.read_csv('Fundamental_Price_Final_25.csv')


ticker_columns=[ 'Threshold_Cross', 'RSI', 'MACD', 'MACD_H',
       'MACD_S', '5d_rolling_mean_Adj Close', '15d_rolling_mean_Adj Close',
       '30d_rolling_mean_Adj Close', '45d_rolling_mean_Adj Close',
       '65d_rolling_mean_Adj Close', '130d_rolling_mean_Adj Close',
       '5d_rolling_mean_Volume', '15d_rolling_mean_Volume',
       '30d_rolling_mean_Volume', '45d_rolling_mean_Volume',
       '65d_rolling_mean_Volume', '130d_rolling_mean_Volume','index_x','Adj Close']

final_columns=ticker_columns + selected_columns
FinalSet=Fundamental_Price[final_columns]
FinalSet = FinalSet.rename(columns={'Threshold_Cross': 'Target'})
FinalSet = FinalSet.sort_values(by=['Ticker', 'index_x'])
FinalSet.set_index('index_x',inplace=True)
FinalSet.index.name = None

#FinalSet.to_csv('FinalSet_before_target_Final_25.csv')
#FinalSet=pd.read_csv('FinalSet_before_target.csv')
FinalSet=pd.read_csv('FinalSet_before_target_Final_25.csv')

FinalSet.set_index('Unnamed: 0',inplace=True)
FinalSet.index.name = None

# Threshold_Cross is the peak value, the beginning of a crash. It is 
# the local max - local min if it is greater than 20%.
# we want to predict the beginning of a -20% crash so we shift back the target.
# It means we ll explain the target with the information which is available at the -shifted day.
FinalSet['Target'] = FinalSet.groupby('Ticker')['Target'].shift(-30)
FinalSet.replace([np.inf, -np.inf], np.nan, inplace=True)

FinalSet = FinalSet.dropna()

# drop additional columns and finalize
APPL_FinalSet=FinalSet[FinalSet.Ticker=='AAPL']
APPL_FinalSet['5d_rolling_mean_Adj Close'].plot()


####  LETS MAKE A DICTIONARY WITH SHIFTED VALUES ##########

FinalSet = pd.read_csv('FinalSet_before_target_Final_25.csv')
FinalSet.set_index('Unnamed: 0', inplace=True)
FinalSet.index.name = None
FinalSet.replace([np.inf, -np.inf], np.nan, inplace=True)

shift_values = [-30, -60]

shifted_datasets = {}

for shift_val in shift_values:
    temp_df = FinalSet.copy()
    temp_df['Final_Target'] = temp_df.groupby('Ticker')['Target'].shift(shift_val)
    temp_df = temp_df.dropna()  
    shifted_datasets[shift_val] = temp_df

##############

# SAVE THE FINAL DATA SET WITH COLUMNS

for shift_val, df in shifted_datasets.items():
    df.to_csv(f"shifted_dataset_{shift_val}.csv", index=True)

# save it as a pickle
with open('shifted_datasets_25_years.pkl', 'wb') as file:
    pickle.dump(shifted_datasets, file)

# DO THE LAGS TO ALLIGN A CORRECT FORM OF TRAINING

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# 5-day rolling mean of Adjusted Close
ax1.plot(APPL_FinalSet['5d_rolling_mean_Adj Close'], label='5d Rolling Mean of Adj Close')
ax1.set_title('5-day Rolling Mean of Adjusted Close')
ax1.legend()

# RSI
ax2.plot(APPL_FinalSet['RSI'], label='RSI', color='orange')
ax2.set_title('Relative Strength Index (RSI)')
ax2.legend()

# Adding vertical lines for Target = True
for date in APPL_FinalSet[APPL_FinalSet['Target']].index:
    ax1.axvline(x=date, color='green', linestyle='--', alpha=0.7)
    ax2.axvline(x=date, color='green', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

