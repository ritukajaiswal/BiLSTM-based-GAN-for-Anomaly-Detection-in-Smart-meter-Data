import datetime as dt
import numpy as np
import pandas as pd
from pandas.tseries.offsets import Day

class Preprocessor:
    def __init__(self, path):
        self.df = self._read_csv(path)

    def _read_csv(self, path) -> pd.DataFrame:
        return pd.read_csv(path, skiprows=1, dayfirst=True, na_filter=False)

    def get_df(self) -> pd.DataFrame:
        return self.df
    
    def format_df(self, channel, epoch_time=True):
        unique_dates = self.df['date'].unique()
        start_date = unique_dates.min()
        end_date = unique_dates.max()
        date_indices = pd.date_range(start_date, end_date+Day(1), freq='30T', closed='left') 

        unique_customers = self.df['Customer'].unique()

        formatted_df = pd.DataFrame(columns=['timestamp'].extend([c for c in unique_customers]))
        formatted_df['timestamp'] = date_indices
        if epoch_time:
            formatted_df['timestamp'] = (formatted_df['timestamp'] - dt.datetime(1970,1,1)).dt.total_seconds()

        for customer in unique_customers:
            customer_data = self.df[self.df['Customer'] == customer]
            channel_filter = customer_data[customer_data['Consumption Category'] == channel]
            consumption_values = channel_filter.loc[:,'0:30':'0:00'].values.ravel()
            if len(consumption_values) < len(date_indices):
                continue
            else:  
                formatted_df[customer] = consumption_values
        
        self.formatted_df = formatted_df

    def get_formatted_df(self) -> pd.DataFrame:
        return self.formatted_df
