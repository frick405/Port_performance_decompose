import FinanceDataReader as fdr
import pandas as pd
import numpy as np

from datetime import datetime
from typing import *

class DataLoader:

    def __init__(self, tick_ls:List[str], st_date:str, ed_date:str=str(datetime.today())[:10]):
        self.tick_ls = tick_ls
        self.st_date = st_date
        self.ed_date = ed_date

    def get_securty_return(self, ) -> pd.DataFrame:

        return_df = pd.concat([fdr.DataReader(col, start=self.st_date, end=self.ed_date)[['Change']] for col in self.tick_ls], 1)
        return_df.columns = self.tick_ls

        return return_df

    def weight_loader(self, time_num:int, asset_num:int) -> pd.DataFrame:

        '''
        :return: pd.Dataframe, For making opportunity set, make random weight that matched to asset_num
        :description: For visualization, it makes random weight to make opportunity set
        '''

        weight_df:pd.DataFrame = pd.concat([pd.DataFrame(np.random.random(time_num)) for i in range(asset_num)], 1)
        weight_df = weight_df.apply(lambda x: x / x.sum(), 1)
        weight_df.columns = self.tick_ls
        weight_df.index = pd.date_range(self.st_date, self.ed_date, freq='D')
        return weight_df

    def load_bm_return(self, tick:str) -> pd.DataFrame:
        return fdr.DataReader(tick)[['Change']]
