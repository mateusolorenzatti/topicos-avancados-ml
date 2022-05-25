"""
7 - Time Series

Author: Mateus Orlandin Lorenzatti (https://github.com/mateusolorenzatti)
Source: https://www.kaggle.com/datasets/kaushiksuresh147/top-10-cryptocurrencies-historical-dataset

Predição do valor da criptomoeda Cardano

"""

from time import time
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima

import platform
import os.path as path

def abrirArquivoLocal(source_file, with_date=False):
    file_path = ''
    if ( 'Linux' in platform.system() ):
        file_path = path.abspath(path.join(__file__ ,"../"*3)) + '/data/' + source_file
    elif ( 'Windows' in platform.system()):
        file_path = path.abspath(path.join(__file__ ,"../"*3)) + '\\data\\' + source_file

    if with_date:
        return pd.read_csv(file_path, parse_dates = ['Date'], index_col = 'Date', date_parser = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d') )
    else:
        return pd.read_csv(file_path)

def preparaBases(base):
    time_series = base['Close']
    # print(time_series.head())

    # plt.plot(time_series)
    # plt.savefig('out.png')

    # print(len(time_series))
    return time_series

def arima(time_series):
    dias_previsao = 9

    train = time_series[:len(time_series) - dias_previsao]
    test = time_series[len(time_series) - dias_previsao:]
    model = auto_arima(train, suppress_warnings=True)

    prediction = pd.DataFrame(model.predict(dias_previsao),index=test.index)
    prediction.columns=['Previsao']

    print(prediction)
    print(test)

    plt.figure(figsize=(8,5))
    plt.plot(train, label = 'Training')
    plt.plot(test, label = 'Test')
    plt.plot(prediction, label = 'Predictions')
    plt.legend()
    # plt.savefig('out.png')

def prophet(time_series):
    from fbprophet import Prophet

    time_series[['Date', 'Close']].rename(columns = {'Date': 'ds', 'Close': 'y'})
    time_series = time_series.sort_values(by = 'ds')

    model = Prophet()
    model.fit(time_series)

    future = model.make_future_dataframe(periods=24)
    forecast = model.predict(future)

    # model.plot(forecast, xlabel = 'Data', ylabel = 'Cardano').savefig('1.png')

def main():
    time_series = abrirArquivoLocal('cardano-value.csv', True)
    time_series = preparaBases(time_series)

    arima(time_series)
    prophet(time_series)

if __name__ == '__main__':
    main()