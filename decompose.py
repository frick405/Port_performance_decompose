import pandas as pd
import numpy as np
import statsmodels.api as sm

from typing import *

class PortDecomposeWithCAPM:

    def __init__(self, port_return:pd.DataFrame, bm_return:pd.DataFrame, rf) -> pd.DataFrame:
        self.port_return = port_return
        self.bm_return = bm_return
        self.rf = rf

    def treynor_mazuy(self):
        '''
        :return:
        '''

        excess_port_ret = self.port_return - self.rf
        excess_mkt_ret = self.bm_return - self.rf

        all_df = pd.concat([excess_port_ret, excess_mkt_ret, excess_mkt_ret ** 2], 1)
        all_df.columns = ['Excess_port', 'Excess_mkt', 'Excess_mkt_square']

        model = sm.OLS(all_df[['Excess_port']], sm.add_constant(all_df[['Excess_mkt', 'Excess_mkt_square']]))
        res = model.fit()
        params_df = pd.DataFrame(res.params)
        pval_df = pd.DataFrame(res.pvalues)
        res_df = pd.concat([params_df, pval_df], 1)
        res_df = np.round(res_df, 3)
        res_df.columns = ['Coef', 'p-values']

        if ((res_df.loc['Excess_mkt_square', 'Coef'] > 0) and (res_df.loc['Excess_mkt_square', 'p-values'] < 0.01)):
            print('Have ability to timing')
        else:
            print('No ability to timing')
        return res_df

    def hm_model(self): # 감마에 대해서 하락장 베타와 상승장 베타를 따로 추정해서 베타 간의 차이가 유의한지 검정하는 것을 확인해야함

        excess_port_ret = self.port_return - self.rf
        excess_mkt_ret = self.bm_return - self.rf
        dummy = np.where(self.bm_return > self.rf, 1, 0) # 수정 가능

        excess_mkt_ret_dummy = dummy * excess_mkt_ret

        all_df = pd.concat([excess_port_ret, excess_mkt_ret, excess_mkt_ret_dummy], 1)
        all_df.columns = ['Excess_port', 'Excess_mkt', 'Excess_mkt_dummy']

        model = sm.OLS(all_df[['Excess_port']], sm.add_constant(all_df[['Excess_mkt', 'Excess_mkt_dummy']]))
        res = model.fit()
        params_df = pd.DataFrame(res.params)
        pval_df = pd.DataFrame(res.pvalues)
        res_df = pd.concat([params_df, pval_df], 1)
        res_df = np.round(res_df, 3)
        res_df.columns = ['Coef', 'p-values']

        return res_df

class DGTW:
    def __init__(self, port_univ_return, bm_univ_return, port_weight, bm_weight):

        self.port_univ_return = port_univ_return
        self.port_weight = bm_weight

        self.bm_univ_return = bm_univ_return
        self.bm_weight = bm_weight

    def dgtw(self): # t 검정 추가 및 내부 구조 변경 필요

        cs_measure = (self.port_weight.shift(1) * (self.port_univ_return - self.bm_univ_return)).sum(1)
        ct_measure = ((self.port_weight.shift(1) * self.port_univ_return.shift(1)) - (self.port_weight.shift(13) * self.port_univ_return.shift(13))).sum(1)
        as_measure = (self.port_weight.shift(13) * self.port_univ_return).sum(1)

        return pd.concat([cs_measure, ct_measure, as_measure], 1)

if __name__ == '__main__':

    port_return = pd.DataFrame(np.random.random(100) / 100)
    bm_return = pd.DataFrame(np.random.random(100) / 100)

    security_ret = pd.DataFrame([np.random.random(100) / 100 for i in range(5)]).T
    security_weight = pd.DataFrame([np.random.random(100) / 100 for i in range(5)]).T.apply(lambda x: x / x.sum(), 1)
    bm_ret = pd.DataFrame([np.random.random(100) / 100 for i in range(5)]).T
    bm_weight = pd.DataFrame([np.random.random(100) / 100 for i in range(5)]).T.apply(lambda x: x / x.sum(), 1)

    decomposer = PortDecomposeWithCAPM(port_return, bm_return, 0.0001)
    dgtw = DGTW(security_ret, security_weight, bm_ret, bm_weight)
    print(dgtw.dgtw())



    # print(decomposer.treynor_mazuy())
    # print(decomposer.hm_model())