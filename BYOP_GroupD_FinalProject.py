#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import seaborn as sns
#to ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[4]:


data = pd.read_csv("C:/Users/shrut/OneDrive/Desktop/Crypto_Historical_Dataset.csv/Crypto_Historical_Dataset.csv")
data


# In[5]:


#There are a total of 9 crypto currencies in the data set containing daily market prices for 6 years ranging from 8 Aug 2015 to 6 July 2021.

sym = data["Symbol"]
print (sym.describe())
print (data.columns.values)
print(data['Date'].head())
print(data['Date'].describe())


# In[6]:


#Basic descriptive statistics on the bitcoin prices used to validate against other publicly available sources.

btc = data[data.Symbol=='BTC']
display(btc.describe())


# In[7]:


#Plotting bitcoin price and volume movements within the entire data set.

btc_date = btc['Date']
btc_close = btc['Close']
btc_volume = btc['Volume']
btc_date.values
btc_close_by_date = pd.Series(btc_close.values, index=btc_date.values)
btc_volume_by_date = pd.Series(btc_volume.values, index=btc_date.values)


# In[8]:


btc_plot = pd.DataFrame(btc_close_by_date, columns=['btc_Close'])


# In[9]:


Start_date = btc_plot.index.values[0]
obs = btc_plot.index.nunique()


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


# In[11]:


#btc_plot.plot()
btc_plot.index = pd.to_datetime(btc_plot.index.values)
ax = btc_plot['btc_Close'].plot(title='Daily Bitcoin Prices', 
                                  figsize = (20,6))
ax.set_xlabel('Daily prices in each year')
ax.set_ylabel('Price in USD');
#ax.set_xticks(np.arange(2013,2018))
ax.xaxis.set_minor_formatter(mp.dates.DateFormatter('%m'))
#ax.xaxis.set_minor_formatter(mp.dates.AutoDateFormatter(mp.dates.AutoDateLocator()))


# In[12]:


btc_volume_plot = pd.DataFrame(btc_volume_by_date, columns=['btc_Volume'])
btc_volume_plot.index = pd.to_datetime(btc_volume_plot.index.values)
ax = btc_volume_plot.plot(figsize = (20,6), title = 'Daily bitcoin transaction volumes in 100,000s')
ax.set_xlabel('Daily transaction volumes in each year')
ax.set_ylabel('Number of transactions in 10 billions');
ax.xaxis.set_minor_formatter(mp.dates.DateFormatter('%m'))


# In[13]:


#The number of transactions in bitcoins show a similar pattern of movement to bitcoin prices, indicating a high correlation between the two.


# In[14]:


#Monthly distribution of BTC across different years


# In[15]:


btc_pv = pd.pivot_table(btc_plot, index=btc_plot.index.month, columns=btc_plot.index.year, values='btc_Close', aggfunc='mean')
btc_pv.index=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
ax = btc_pv.plot(title="Monthly plot: Bitcoin closing prices (mean)", 
                 figsize=(15,6), marker='*')
ax.set_xlabel('Month')
ax.set_ylabel('$')
ax.set_xticks(np.arange(0,12))
ax.set_xticklabels(btc_pv.index)


# In[16]:


fig = plt.figure(figsize=[20,3]);
# subplots
ax1 = plt.subplot2grid((1,4), (0,0))
ax2 = plt.subplot2grid((1,4), (0,1))
ax3 = plt.subplot2grid((1,4), (0,2))
ax4 = plt.subplot2grid((1,4), (0,3))
arr_ax = [ax1,ax2, ax3, ax4]
for i in range(2016,2020):
    btc_plot[str(i)].btc_Close.plot(ax=arr_ax[i-2016])


# In[17]:


btc_scatter = pd.DataFrame(btc_plot)
btc_scatter['btc_Volume'] = btc_volume_plot['btc_Volume']

fig = plt.figure(figsize=[20,3]);
# subplots
ax1 = plt.subplot2grid((1,4), (0,0))
ax2 = plt.subplot2grid((1,4), (0,1))
ax3 = plt.subplot2grid((1,4), (0,2))
ax4 = plt.subplot2grid((1,4), (0,3))
arr_ax = [ax1,ax2, ax3, ax4]
for i in range(2016,2020):
    btc_plot[str(i)].btc_Volume.plot(ax=arr_ax[i-2016])


# In[18]:


#Scatter plot of BTC Prices and Volumes


# In[19]:


ax = btc_scatter.plot.scatter(y='btc_Close', 
                             x='btc_Volume', 
                             alpha=1.0, 
                             figsize=(15,7), 
                             title='Close price plotted against Volume')
ax.set_xlabel('Volume')
ax.set_ylabel('Close price');


# In[20]:


#Calculate correlation coefficient of Bitcoin prices on any given day to it's volume


# In[21]:


btc_scatter['btc_Close'].corr(btc_scatter['btc_Volume'])


# In[22]:


#Plot the following top 5 cryptocurrencies by market cap in addition to BTC


# In[23]:


def plot_crypto(crypto, ax):
    btc = data[data.Symbol==crypto]
    
    btc_date = btc['Date']
    btc_close = btc['Close']
    btc_volume = btc['Volume']
    
    btc_close_by_date = pd.Series(btc_close.values, index=btc_date.values)
    btc_volume_by_date = pd.Series(btc_volume.values, index=btc_date.values)
    
    btc_plot = pd.DataFrame(btc_close_by_date, columns=['btc_close'])
    
    btc_plot.index = pd.to_datetime(btc_plot.index.values)
    ax = btc_plot['btc_close'].plot(title=str('Daily '+ crypto + ' Prices'), figsize = (20,6), ax=ax)
    ax.set_xlabel('Year')
    ax.set_ylabel('Thousands')
    #ax.set_xticks(np.arange(2013,2018))
    ax.xaxis.set_minor_formatter(mp.dates.DateFormatter('%m'))
    #ax.xaxis.set_minor_formatter(mp.dates.AutoDateFormatter(mp.dates.AutoDateLocator()))
    
    
fig = plt.figure(figsize=[20,6]);
# subplots
ax1 = plt.subplot2grid((2,2), (0,0))
ax2 = plt.subplot2grid((2,2), (0,1))
ax3 = plt.subplot2grid((2,2), (1,0))
ax4 = plt.subplot2grid((2,2), (1,1))
arr_ax = [ax1,ax2, ax3, ax4]

arr_cryp = ["ETH", "XRP", "DOGE", "LTC"]
for i in range(0, len(arr_ax)):
    plot_crypto( arr_cryp[i], arr_ax[i])
    
fig.tight_layout() 


# In[24]:


#Histogram plot by market value of all cryptos


# In[25]:


# select date
fig = plt.figure(figsize=[10,4]);
curr_date = "8/13/2018"
data_mcap = data[data.Date==curr_date]
ax = sns.distplot(data_mcap.Marketcap, hist = True, rug=True, kde=False, bins=10)
ax.set_xlabel("Marketcap Cap($)");
ax.set_ylabel("Number of Crypto Currencies")


# In[26]:


# top 20 cryptos by market value .


# In[27]:


fig = plt.figure(figsize=[10,4]);
df_mcap_by_name = pd.DataFrame(data_mcap, columns = {'Marketcap', 'Name'})
df_mcap_by_name.index = data_mcap.Name.values
df_mcap_by_name['mktshare'] = df_mcap_by_name['Marketcap'] *100 / df_mcap_by_name.Marketcap.sum()
subtotal = df_mcap_by_name.Marketcap.head(20).sum() * 100 / df_mcap_by_name.Marketcap.sum()
print("These top 20 cryptos account for " + str(round(subtotal,2)) + "% of the total market value of ALL cryptos")
display(df_mcap_by_name.mktshare.head(20))
ax = sns.distplot(df_mcap_by_name.mktshare.head(20), hist = True, rug=True, kde=False, bins=15)
ax.set_xlabel("% Share of Total Market Cap across all Cryptos")
ax.set_ylabel("Number of Crypto Currencies")


# In[28]:


#Scatter plot of Price vs Volume for ETH


# In[29]:


eth= data[data.Symbol=='ETH']
eth = eth.set_index(eth['Date'])
ax = eth.plot.scatter(y='Close', 
                             x='Volume', 
                             alpha=1.0, 
                             figsize=(15,7), 
                             title='Close price plotted against volume')
ax.set_xlabel('Volume')
ax.set_ylabel('Close price');


# In[30]:


eth['Close'].corr(eth['Volume'])
#ETH volumes to find correlation against bitcoin prices


# Consolidating findings from data analysis and forming ideas for modelling
# 
# So by this stage, We had a pretty good sense of the trend and variation in the top crypto prices and their volumes.
# 
# We also realized that there were a different of different predictors we could consider. To make this effort bounded and focus on a few learning objectives that could be conclusive, We decided to focus on one or two specific cryptoc - BTC and ETH (Bitcoin and Etherium).
# 
# We also decided to drop any analysis of exogenous predictor variables (outside of my chosen data set and domain of interest).
# 
# 1. Examine autocorrelation within cypto market prices, for BTC and ETH
# 2. Co-relation of BTC with other cryptos like ETH
# 3. Predictive models using linear regression and multiple regression
# 4. Residual analysis and modelling errors
# 5. Frameing hypothesis and calculating p-value
# 6. Calculate Akaike's information criteria - AIC and BIC
# 7. Analyse trend and seasonanlity in BTC prices
# 8. Use the above insights to design a more sophistical model like ARIMA an compare against a far simpler model like Naive Forecast
# 

# # Modelling of BTC prices using different forecasting techniques

# # Using linear regression to predict prices based on volume (based on a single predictor)

# In[31]:


import statsmodels.formula.api as sm

btc_training = btc_scatter['2017']
btc_validation = btc_scatter[ (btc_scatter.index >= '01-01-2015') & (btc_scatter.index <= '31-12-2015') ]

#btc_training['log_volumes'] = 

#eth_training = eth[ (eth.index >= '01-01-2017') & (eth.index <= '31-12-2017') ]

display(btc_training.describe())
btc_prices = btc_training['btc_Close']
btc_volumes = btc_training['btc_Volume']
result = sm.ols(formula="btc_prices ~ btc_volumes", data=btc_training.shift(1)).fit()
display( result.summary())
btc_training.plot.scatter(x='btc_Close', y='btc_Volume')


# While the R square values are very good at 0.867 and low P values, this may well be an example of a spurious regression with an unknown variable influencing both prices and volumes. So need to investigate the residual and ACF plots to check further.

# In[32]:


result.resid.plot()


# In[34]:


from statsmodels.graphics.tsaplots import month_plot, plot_acf, plot_pacf
x = plot_acf(result.resid, lags=20, zero=False)
x = plot_pacf(result.resid, zero=False, lags=20)
ax = plt.figure(figsize=(8,4))



# From the PACF plots of the in sample residuals, there is a significant auto-correlation at lags 3, 9 and 14. Thus it is likely that the residuals point to a yet unknown variable that is influencing both the prices and volumes. It does seem to be normally distributed around a zero mean. However, a spurious correlation cannot be ruled out.

# # Linear regression of BTC based on other Crypto Prices (multiple regression) 

# So, Let now see if the use of other predictors can help us avoid a possible spurious regression. We will now use other three other cyrpto prices (Etherium, Litecoin and XRP as possible predictors of BTC prices

# In[36]:


btc_training = btc_scatter['2017']
btc_validation = btc_scatter[ (btc_scatter.index >= '01-01-2015') & (btc_scatter.index <= '31-12-2015') ]

eth = data[data.Symbol=='ETH']
eth = eth.set_index(eth['Date'])

xrp = data[data.Symbol=='XRP']
xrp = xrp.set_index(xrp['Date'])

ltc = data[data.Symbol=='LTC']
ltc = ltc.set_index(ltc['Date'])

#btc_training['log_volumes'] = 

#eth_training = eth[ (eth.index >= '01-01-2017') & (eth.index <= '31-12-2017') ]

#print(btc_training.describe())
eth.index = pd.to_datetime(eth.index.values)
ltc.index = pd.to_datetime(ltc.index.values)
xrp.index = pd.to_datetime(xrp.index.values)

eth_prices = eth['2017']['Close'].shift(1)
xrp_prices = xrp['2017']['Close'].shift(1)
ltc_prices = ltc['2017']['Close'].shift(1)
  
result = sm.ols(formula="btc_prices ~ eth_prices + xrp_prices + ltc_prices", data=btc_training).fit()
display( result.summary())
#btc_training.plot.scatter(x='btc_close', y='btc_volume')


# In[37]:


x = plot_acf(result.resid, zero=False, lags=30) 


# In[38]:


x = plot_pacf(result.resid, zero=False, lags=45)


# In[39]:


sns.distplot(result.resid, kde=True, rug=True, hist = True)


# There is still a high degree of auto-correlation in the residuals which indicates that there are other factors which are determining the BTC prices

# # Examining auto-correlation within BTC prices

# To understand what's behind this auto-correlation, we decided to plot the BTC prices against itself with a lag of 1. We also wanted to checked if using a Log transformation would yield new insights.

# In[48]:


btc_scatter
btc_scatter['log_Volume'] = (btc_scatter['btc_Volume'].apply(np.log))
btc_scatter['2017'].plot.scatter(x='btc_Close', y='log_Volume')


# # Taking out the influence of any auto-correlation in my time series.
# 

# There are two possible approaches. One approach was to exploit any trend or seasonality in the data. The second approach was to remove any trend, seasonality and auto-correlation in the data, treating it purely as a white noise and then predicting from there.

# # Calculating 1 order differentials in the BTC time series

# In[49]:


btc_shift = btc_scatter.shift(1)


# In[50]:


btc_diff = btc_scatter - btc_shift
x = plot_acf(btc_diff['2017']['btc_Close'], lags=20, zero = False)


# This didn't help as the ACF plot above suggests. So trying a weekly difference instead of a daily difference.

# In[51]:


btc_diff_weekly = btc_diff.resample("W").mean()
x = plot_acf(btc_diff_weekly['2017']['btc_Close'], lags=20, zero = False)


# The weekly difference did remove auto-correlation as seen in the ACF plot above.

# In[52]:


sns.distplot(btc_diff_weekly['2017']['btc_Close'], kde=True, rug=True, hist = True)


# # So the weekly difference is indeed a stationary series. So exploiting this in forecasting.

# In[53]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
res_diff = sm.tsa.seasonal_decompose(btc_diff_weekly['2017':]['btc_Close'], freq=4, model='additive')
x = res_diff.plot()


# # Now extracting and using Trend and Seasonality in BTC weekly prices

# In[54]:


btc_weekly = btc_scatter.resample('W').mean()
# decompose time series into components
res = sm.tsa.seasonal_decompose(btc_weekly['2017':]['btc_Close'], freq=4, model='additive')
x = res.plot()


# In[55]:


sns.set_context("talk")
sns.set_style('darkgrid') 
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import display, HTML, Javascript
from matplotlib.pyplot import xlabel, ylabel, title, legend

# plot original data
ax = btc_scatter['2017':]['btc_Close'].plot(color="darkgrey", figsize=(12,6))
# plot trend component
res.trend.plot(ax=ax, color="red")
# formatting
title("BTC prices and trend line",fontsize=20)
#ylabel("Prices in $")
ax.legend(['Data', 'Trend-cycle']);
#plt.close()


# In[56]:


display(pd.infer_freq(btc_weekly.index))


# #  Portmanteau test for autocorrelation

# In[57]:


from statsmodels.stats.diagnostic import acorr_ljungbox
h = 10
# calculate Ljung-Box and Box-Pierce test
lbvalue, lbpvalue, bpvalue, bppvalue = acorr_ljungbox(btc_weekly['2017'].btc_Close.dropna(), lags=h,  boxpierce=True)
lbvalue[h-1], lbpvalue[h-1]
# display Ljung-Box test stats
data = pd.DataFrame(columns=['X-squared','p-value'], index = range(1))
data['X-squared'] = lbvalue[h-1]; data['p-value'] = lbpvalue[h-1];
display("Ljung-Box values before applying weekly differencing")
data


# The p-values are quite small indicating that there is still a high degree of auto-correlation present in the weekly prices. This is not surprising at all as the weekly prices are indeed auto-correlated with the previous week's prices, hence Ljung-Box test can indeed be used to look at any remaining auto-correlation in the residuals or differenced prices.

# In[58]:


h = 10
# calculate Ljung-Box and Box-Pierce test
lbvalue, lbpvalue, bpvalue, bppvalue = acorr_ljungbox(btc_diff_weekly['2017'].btc_Close.dropna(), lags=h,  boxpierce=True)
lbvalue[h-1], lbpvalue[h-1]
# display Ljung-Box test stats
data = pd.DataFrame(columns=['X-squared','p-value'], index = range(1))
data['X-squared'] = lbvalue[h-1]; data['p-value'] = lbpvalue[h-1];
display("Ljung-Box values after applying weekly differencing")
data


# The p-values are high indicating that there is little auto-correlation present first order differences of weekly prices. The differencing operation on weekly prices has made a tangible difference on removing the auto-correlation

# # Forecasting with a Simple Exponential Forecasting model (SES) using Insample and Out of Sample data sets

# An in-sample data set of two quarters of BTC price data from 01-Apr-2017 to 30-Sep-2017 is used below to create an SES model. The model thus derived is used to plot the forecast values in the same data set. Further tests below apply the same model to a previously unseen validation set of BTC price data.

# In[59]:


# Training (in-sample) accuracy
def accuracy(y_cap, y, y_train, is_ts=True, is_seasonal=True):
    e = y - y_cap
    ME = e.mean()
    RMSE = (e**2.).mean()**0.5
    MAE  = e.abs().mean()
    MAPE = (100.*e/y).abs().mean()
    freq = pd.infer_freq(y_train.index)
    display("Frequency of training data set is " + freq)
    q = e/((y_train-y_train.shift({'W':7,'D':1}[freq])).abs()).mean() if is_ts and is_seasonal else         e/((y_train-y_train.shift(1)).abs()).mean() if is_ts else         e/((y_train-y_train.mean()).abs()).mean()
    MASE = q.abs().mean()
    return ME, RMSE, MAE, MAPE, MASE


# In[60]:


from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

btc_training = btc_scatter['01-04-2017':'30-09-2017']

# fit simple exponential smoothing model
fit3 = SimpleExpSmoothing(btc_training.btc_Close).fit()

# also obtain other fit with Holt's method which extends SES with a trend function
#fit4 = Holt(btc_training.btc_close, exponential=False).fit(optimized=True)

# display forecast accuracy measures
results = pd.DataFrame(columns = ['ME','RMSE','MAE','MAPE','MASE'])
results.loc['results'] = accuracy(fit3.fittedvalues, btc_training.btc_Close, btc_training.btc_Close)


# In[61]:


display(results)
#btc_training.head()


# These error measures will be used to measure the performance of the forecast and compare against different forecast models that we will use further.

# In[62]:


display("Forecast values: " + str(fit3.fcastvalues))


# In[63]:


fit3.fittedvalues.median()


# # Plotting the in-sample forecasts from SES

# In[64]:


# plot original data
ax = btc_training.btc_Close.plot(color='black', figsize=(12,8))
# simple exponential smoothing plot
fit3.forecast(5).rename('Forecast').plot( ax=ax, color='blue', legend=True)
fit3.fittedvalues.plot( ax=ax, color='pink', label = 'Fitted')
# plot formatting
ax.set_xlabel('Day')
ax.set_ylabel('Bitcoin prices in USD')
ax.set_title('In Sample Forecasts from Simple exponential smoothing', fontsize = 18)
ax.legend(loc = 'upper left')
#plt.close()


# # Intrepreting the In-Sample forecast results from SES
# 1. The Mean Error is very low indicating a good in-sample forecast which is also visualized in the plot where the SES forecast plot closely mirrors the actual price plot among the in-sample values.
# 2. The Percentage Error (MAPE) is also extremely low at 3% indicating a good forecast performance. This being a unit free measure, we can use this to compare different forecast models as we try out different approaches
# 3. The Mean Absolute Scaled Error (MASE) is less than 1 indicating that this model provides a better forcast than an average naive forecast on the in-sample data.

# # Testing model performance with Out of Sample data using a Validation Set

# In[73]:


# testing accuracy (out-of-sample)
def test_accuracy(y_cap, y, is_ts=True, is_seasonal=True):
    e = y - y_cap
    ME = e.mean()
    RMSE = (e**2.).mean()**0.5
    MAE  = e.abs().mean()
    return ME, RMSE, MAE,


# # Forecasting with ARIMA

# In[65]:


from statsmodels.tsa.arima_model import ARIMA

# fit naive forecast (=ARIMA(0,0,0))
fit5 = ARIMA(btc_training.btc_Close, (2,1,2)).fit() 
#fit5 = ARIMA(btc_diff_weekly.btc_close, (1,0,1)).fit() #This is the quivalent of a naive forecast
#fcast5 = fit5.forecast(90)


# In[66]:


fit = ARIMA(btc_training.btc_Close, (2,1,2)).fit(trend='c') 
fit.summary()


# In[83]:


# plot seasonally adjusted data
ax = btc_training.btc_Close.plot(figsize=(12,8), color = 'black', label="In sample BTC closing prices")
# plot forecast and 95% CI
fit5.plot_predict(start='2017-10-01', end='2017-12-31', alpha=0.05, plot_insample=False, ax=ax)# 95% CI
#little hack to get the two confidence intervals 
fit5.plot_predict(start='2017-10-01', end='2017-12-31', alpha=0.15, plot_insample=False, ax=ax) # 85% CI
# format
ylabel("BTC Price in USD", fontsize = 14); xlabel("Date", fontsize = 14);
btc_validation.btc_Close.plot(ax=ax, color='blue', legend=True, label="Out of sample BTC closing prices")
ax.set_title('ARIMA forecasts of BTC prices',fontsize= 24);


# # Evaluating forecast accuracy of the ARIMA model

# The ARIMA fitted prediction object returns the differenced forecast values by default. Using the typ='levels' parameter below allowed me to get the levels of the prices instead of merely the differenced values.

# In[82]:


predicted_arima = fit5.predict(start='2017-10-01', end='2017-12-31', typ='levels')

btc_validation.loc[:,"btc_Close_ycap_ARIMA"] = predicted_arima.values
display(btc_validation.tail())


# The predicted results table above for the validation set now contains predicted values extracted via both SES and ARIMA. So, Checking which one is more accurate.

# In[86]:


results.loc['results-ARIMA'] = test_accuracy(btc_validation.btc_Close_ycap_ARIMA, btc_validation.btc_Close)
display(results)


# A comparative view of the error measures indicate that the SES Holt Trend method outperforms the ARIMA model in this particular instance.

# # Residual Analysis of ARIMA forecasts

# In[70]:



x = plot_acf(residuals_ARIMA, zero=False, lags=20)
x = plot_pacf(residuals_ARIMA, zero=False, lags=20)
ax = plt.figure(figsize=(10,6))


# # Comparing these results against a Naive method of forecasting as a benchmark

# In[87]:


btc_validation_naive = btc_scatter["2017-09-30":"2017-12-31"].shift(1) # shift one step ahead for naive forecasts
results.loc['results-Naive'] = test_accuracy(btc_validation_naive.btc_Close, btc_validation.btc_Close)
display(results)


# These results indicate that the Naive forecasts outperforms the other methods by a significant measure

# In[88]:


# plot seasonally adjusted data
ax = btc_training.btc_Close.plot(figsize=(12,8), color = 'black', label="In sample BTC closing prices")
# format
ylabel("BTC Price in USD", fontsize = 14); xlabel("Date", fontsize = 14);
btc_validation.btc_Close.plot(ax=ax, color='blue', legend=True, label="Out of sample BTC closing prices")
btc_validation_naive.btc_Close.plot(ax=ax, color='lightgreen', legend=True, label="Naive forecasted BTC closing prices (out of sample)")
ax.set_title('Naive forecasts of BTC prices',fontsize= 24);


# # Residual Analysis of Naive method

# In[89]:


residuals_Naive = (btc_validation_naive.btc_Close - btc_validation.btc_Close).dropna()
x = plot_acf(residuals_Naive, zero=True, alpha=.5, lags=10)
x = plot_pacf(residuals_Naive, zero=False, lags=20)
residuals_Naive.mean()


# With the trend information removed in the ACF plot above, resulting in the PACF plot, there does not appear to be any significant auto-correlation left in the residuals. It is normally distributed with a slight non zero mean.So, In our view, It is an an efficient forecast compared to the other methods.

# In[90]:


sns.distplot(residuals_Naive, kde=True, rug=True).set_title('Histogram of residuals');


# # Levelling the playing field - How does ARIMA perform on in-sample forecasts?

# In[91]:


arima_training = (btc_scatter['2017-08-01':].shift(1)).dropna()

fit = ARIMA(arima_training.btc_Close, (2,1,2)).fit(trend='c') 
display(fit.summary())
# plot seasonally adjusted data
ax = btc_validation.btc_Close.plot(figsize=(12,8), color = 'black', label="In sample BTC closing prices")
# plot forecast and 95% CI
fit.plot_predict(start='2017-10-01', end='2017-12-31', alpha=0.001, plot_insample=False, ax=ax)# 95% CI
#little hack to get the two confidence intervals 
#fit.plot_predict(start='2017-10-01', end='2017-12-31', alpha=0.15, plot_insample=True, ax=ax) # 85% CI
# format
ylabel("BTC Price in USD", fontsize = 14); xlabel("Date", fontsize = 14);
#btc_validation.btc_close.plot(ax=ax, color='blue', legend=True, label="Out of sample BTC closing prices")
ax.set_title('ARIMA forecasts of BTC prices - In sample within validation set',fontsize= 24);


# # Comparision of forecasting performance across models

# In[92]:


predicted_arima = fit.predict(start='2017-10-01', end='2017-12-31', typ='levels')

btc_validation.loc[:,"btc_Close_ycap_In_Sample_ARIMA"] = predicted_arima.values
results.loc['results-ARIMA-InSample'] = test_accuracy(btc_validation.btc_Close_ycap_In_Sample_ARIMA, btc_validation.btc_Close)
display(results)


# Naive wins hands down with the lowest Mean Error (ME), Root Mean Square Error (RMSE) and Mean Average Error (MAE). This is followed by ARIMA-In-Sample which yielded it's performance with a differencing of 1.
# The reasons for this is possibly because BTC daily prices resemble a white noise series. Given a short forecasting window like 1 day, the naive forecast yields the best results, as it simply uses the previous day's price as forecast. The ARIMA and SES forecast methods on the other hand also factor in the previous prices which probably increases the probability of higher forecast errors.

# # Conclusion and Further areas to explore

# While the Naive forecasts have performed the best at a daily forecasting level, as the forecasting window increases to a weekly or monthly level, the SES (Holt Trend) or ARIMA models have shown the ability to capture the trend (and seasonality, if any) in the data set. 
# 
# In the particular domain of interest chosen here, which is cryptocurrency price estimation, it is also doubtful whether weekly or monthly forecast will have any practical utility. However, for other kinds of time series which do not resemble a white noise series, We expect that these other forecasting methods will perform better, given the limitation of the Naive forecast to base itself on the last known value.
# 
# While there does appears to be some seasonality within the weekly BTC prices (as detected at a frequency=4 in ACF plot), this can potentially be exploited in a Seasonal Arima forecast.
