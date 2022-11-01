from abc import ABCMeta, abstractmethod
from typing import List

import pandas as pd
import talib as ta


class Indicator(metaclass=ABCMeta):
    """
    所有indicator的基类
    """
    @classmethod
    @abstractmethod
    def compute(cls, data: pd.DataFrame, **kwargs) -> List[pd.Series]:
        return []


class ADX(Indicator):

    @classmethod
    def compute(cls, data: pd.DataFrame, window=14) -> List[pd.Series]:
        real = ta.ADX(data["High"], data["Low"], data["Close"], timeperiod=window)
        return [real]


class ADXR(Indicator):

    @classmethod
    def compute(cls, data: pd.DataFrame, window=14) -> List[pd.Series]:
        real = ta.ADXR(data["High"], data["Low"], data["Close"], timeperiod=window)
        return [real]


class BB(Indicator):
    """
                Adj_Ratio     Close      High       Low      Open       Volume  Upper BollingerBand  Lower BollingerBand
    Date
    1980-12-12   0.780522  0.100178  0.100614  0.100178  0.100178  469033600.0                  NaN                  NaN
    1981-02-25   0.780522  0.087983  0.088418  0.087983  0.087983   19488000.0                  NaN                  NaN
    1981-05-07   0.780522  0.096694  0.097130  0.096694  0.096694    9363200.0                  NaN                  NaN
    1981-07-20   0.780522  0.084063  0.084499  0.084063  0.084499   23654400.0                  NaN                  NaN
    1981-09-29   0.780522  0.052702  0.053138  0.052702  0.052702   94684800.0             0.121977             0.046671
    1981-12-09   0.780522  0.065769  0.066205  0.065769  0.065769   34272000.0             0.113134             0.041751
    1982-02-22   0.780522  0.064463  0.064898  0.064463  0.064898   26633600.0             0.107676             0.037801
    1982-05-04   0.780522  0.054881  0.055316  0.054881  0.054881   73987200.0             0.089194             0.039557
    1982-07-15   0.780522  0.044427  0.044863  0.044427  0.044427   65788800.0             0.074115             0.038782
    1982-09-24   0.780522  0.063156  0.063591  0.063156  0.063591  178192000.0             0.076461             0.040617
    """

    @classmethod
    def compute(cls, data: pd.DataFrame, window=14) -> List[pd.Series]:
        MA = pd.Series(data["Close"].rolling(window=window).mean())
        SD = pd.Series(data["Close"].rolling(window=window).std())

        b1 = MA + (2 * SD)
        B1 = pd.Series(b1, name=f'Upper BollingerBand({window})')

        b2 = MA - (2 * SD)
        B2 = pd.Series(b2, name=f'Lower BollingerBand({window})')

        return [B1, B2]


class CCI(Indicator):
    """
    一般小于-100 表示被低估

    Attributes  Adj_Ratio     Close      High       Low      Open       Volume         CCI
    Date
    1980-12-12   0.780522  0.100178  0.100614  0.100178  0.100178  469033600.0         NaN
    1981-02-25   0.780522  0.087983  0.088418  0.087983  0.087983   19488000.0         NaN
    1981-05-07   0.780522  0.096694  0.097130  0.096694  0.096694    9363200.0         NaN
    1981-07-20   0.780522  0.084063  0.084499  0.084063  0.084499   23654400.0         NaN
    1981-09-29   0.780522  0.052702  0.053138  0.052702  0.052702   94684800.0 -111.975400
    1981-12-09   0.780522  0.065769  0.066205  0.065769  0.065769   34272000.0  -43.607498
    1982-02-22   0.780522  0.064463  0.064898  0.064463  0.064898   26633600.0  -31.582973
    1982-05-04   0.780522  0.054881  0.055316  0.054881  0.054881   73987200.0  -51.008980
    1982-07-15   0.780522  0.044427  0.044863  0.044427  0.044427   65788800.0  -90.727455
    1982-09-24   0.780522  0.063156  0.063591  0.063156  0.063591  178192000.0   34.347050
    """

    @classmethod
    def compute(cls, data: pd.DataFrame, window=14) -> List[pd.Series]:
        data = data.copy()
        TP = (data['High'] + data['Low'] + data['Close']) / 3.  # typical price
        ret = pd.Series((TP - TP.rolling(window=window).mean()) / (0.015 * TP.rolling(window=window).std()),
                        name=f"CCI({window})")
        return [ret]


class MINUS_DI(Indicator):

    @classmethod
    def compute(cls, data: pd.DataFrame, window=14) -> List[pd.Series]:
        real = ta.MINUS_DI(data["High"], data["Low"], data["Close"], timeperiod=window)

        return [real]


class MINUS_DM(Indicator):

    @classmethod
    def compute(cls, data: pd.DataFrame, window=14) -> List[pd.Series]:
        real = ta.MINUS_DM(data["High"], data["Low"], timeperiod=window)

        return [real]


class PLUS_DI(Indicator):

    @classmethod
    def compute(cls, data: pd.DataFrame, window=14) -> List[pd.Series]:
        real = ta.PLUS_DI(data["High"], data["Low"], data["Close"], timeperiod=window)

        return [real]


class PLUS_DM(Indicator):

    @classmethod
    def compute(cls, data: pd.DataFrame, window=14) -> List[pd.Series]:
        real = ta.PLUS_DM(data["High"], data["Low"], timeperiod=window)

        return [real]


class EVM_MA(Indicator):
    """
    一般和0比
                Adj_Ratio     Close      High       Low      Open       Volume       EVM
    Date
    1980-12-12   0.780522  0.100178  0.100614  0.100178  0.100178  469033600.0       NaN
    1981-02-25   0.780522  0.087983  0.088418  0.087983  0.087983   19488000.0       NaN
    1981-05-07   0.780522  0.096694  0.097130  0.096694  0.096694    9363200.0       NaN
    1981-07-20   0.780522  0.084063  0.084499  0.084063  0.084499   23654400.0       NaN
    1981-09-29   0.780522  0.052702  0.053138  0.052702  0.052702   94684800.0       NaN
    1981-12-09   0.780522  0.065769  0.066205  0.065769  0.065769   34272000.0 -0.000002
    1982-02-22   0.780522  0.064463  0.064898  0.064463  0.064898   26633600.0  0.000003
    1982-05-04   0.780522  0.054881  0.055316  0.054881  0.054881   73987200.0 -0.000006
    1982-07-15   0.780522  0.044427  0.044863  0.044427  0.044427   65788800.0 -0.000003
    1982-09-24   0.780522  0.063156  0.063591  0.063156  0.063591  178192000.0  0.000001
    """

    @classmethod
    def compute(cls, data: pd.DataFrame, window=14) -> List[pd.Series]:
        dm = ((data['High'] + data['Low']) / 2.) - ((data['High'].shift(1) + data['Low'].shift(1)) / 2.)
        br = (data['Volume'] / 100000000) / (data['High'] - data['Low'])
        evm = dm / br
        ret = pd.Series(evm.rolling(window).mean(), name=f"EVM({window})")
        return [ret]


class FI(Indicator):
    """
                Adj_Ratio     Close      High       Low      Open       Volume  ForceIndex(5)
    Date
    1980-12-12   0.780522  0.100178  0.100614  0.100178  0.100178  469033600.0            NaN
    1981-02-25   0.780522  0.087983  0.088418  0.087983  0.087983   19488000.0            NaN
    1981-05-07   0.780522  0.096694  0.097130  0.096694  0.096694    9363200.0            NaN
    1981-07-20   0.780522  0.084063  0.084499  0.084063  0.084499   23654400.0            NaN
    1981-09-29   0.780522  0.052702  0.053138  0.052702  0.052702   94684800.0            NaN
    1981-12-09   0.780522  0.065769  0.066205  0.065769  0.065769   34272000.0  -1.179275e+06
    1982-02-22   0.780522  0.064463  0.064898  0.064463  0.064898   26633600.0  -6.264286e+05
    1982-05-04   0.780522  0.054881  0.055316  0.054881  0.054881   73987200.0  -3.093651e+06
    1982-07-15   0.780522  0.044427  0.044863  0.044427  0.044427   65788800.0  -2.607584e+06
    1982-09-24   0.780522  0.063156  0.063591  0.063156  0.063591  178192000.0   1.862729e+06
    """

    @classmethod
    def compute(cls, data: pd.DataFrame, window=14) -> List[pd.Series]:
        FI = pd.Series(data['Close'].diff(window) * data['Volume'], name=f'ForceIndex({window})')
        return [FI]


class MA(Indicator):
    """
    Attributes  Adj_Ratio     Close      High       Low      Open       Volume  MA5_Close
    Date
    1980-12-12   0.780522  0.100178  0.100614  0.100178  0.100178  469033600.0        NaN
    1981-02-25   0.780522  0.087983  0.088418  0.087983  0.087983   19488000.0        NaN
    1981-05-07   0.780522  0.096694  0.097130  0.096694  0.096694    9363200.0        NaN
    1981-07-20   0.780522  0.084063  0.084499  0.084063  0.084499   23654400.0        NaN
    1981-09-29   0.780522  0.052702  0.053138  0.052702  0.052702   94684800.0   0.084324
    1981-12-09   0.780522  0.065769  0.066205  0.065769  0.065769   34272000.0   0.077442
    1982-02-22   0.780522  0.064463  0.064898  0.064463  0.064898   26633600.0   0.072738
    1982-05-04   0.780522  0.054881  0.055316  0.054881  0.054881   73987200.0   0.064376
    1982-07-15   0.780522  0.044427  0.044863  0.044427  0.044427   65788800.0   0.056448
    1982-09-24   0.780522  0.063156  0.063591  0.063156  0.063591  178192000.0   0.058539
    """

    @classmethod
    def compute(cls, data: pd.DataFrame, window=14) -> List[pd.Series]:
        ma = pd.Series(data["Close"].rolling(window=window).mean(), name=f"MA({window})")
        return [ma]


class EMA(Indicator):
    """
    Attributes  Adj_Ratio     Close      High       Low      Open       Volume  EWMA5_Close
    Date
    1980-12-12   0.780522  0.100178  0.100614  0.100178  0.100178  469033600.0     0.100178
    1981-02-25   0.780522  0.087983  0.088418  0.087983  0.087983   19488000.0     0.096113
    1981-05-07   0.780522  0.096694  0.097130  0.096694  0.096694    9363200.0     0.096307
    1981-07-20   0.780522  0.084063  0.084499  0.084063  0.084499   23654400.0     0.092226
    1981-09-29   0.780522  0.052702  0.053138  0.052702  0.052702   94684800.0     0.079051
    1981-12-09   0.780522  0.065769  0.066205  0.065769  0.065769   34272000.0     0.074624
    1982-02-22   0.780522  0.064463  0.064898  0.064463  0.064898   26633600.0     0.071237
    1982-05-04   0.780522  0.054881  0.055316  0.054881  0.054881   73987200.0     0.065785
    1982-07-15   0.780522  0.044427  0.044863  0.044427  0.044427   65788800.0     0.058666
    1982-09-24   0.780522  0.063156  0.063591  0.063156  0.063591  178192000.0     0.060162
    """

    @classmethod
    def compute(cls, data: pd.DataFrame, window=14) -> List[pd.Series]:
        ema = pd.Series(data["Close"].ewm(span=window, adjust=False).mean(), name=f"EMA({window})")
        return [ema]


class MACD(Indicator):

    @classmethod
    def compute(cls, data: pd.DataFrame, fastperiod=12, slowperiod=26, signalperiod=9) -> List[pd.Series]:
        macd, macdsignal, macdhist = ta.MACD(data["Close"], fastperiod=fastperiod, slowperiod=slowperiod,
                                             signalperiod=signalperiod)
        macd.name = f"macd_{fastperiod}_{slowperiod}_{signalperiod}"
        macdsignal.name = f"macdsignal_{fastperiod}_{slowperiod}_{signalperiod}"
        macdhist.name = f"MACDhist_{fastperiod}_{slowperiod}_{signalperiod}"
        return [macd, macdsignal, macdhist]


class ROC(Indicator):
    """
    ROC上升 看涨 反之 看跌
                Adj_Ratio     Close      High       Low      Open       Volume     ROC_5
    Date
    1980-12-12   0.780522  0.100178  0.100614  0.100178  0.100178  469033600.0       NaN
    1981-02-25   0.780522  0.087983  0.088418  0.087983  0.087983   19488000.0       NaN
    1981-05-07   0.780522  0.096694  0.097130  0.096694  0.096694    9363200.0       NaN
    1981-07-20   0.780522  0.084063  0.084499  0.084063  0.084499   23654400.0       NaN
    1981-09-29   0.780522  0.052702  0.053138  0.052702  0.052702   94684800.0       NaN
    1981-12-09   0.780522  0.065769  0.066205  0.065769  0.065769   34272000.0 -0.343480
    1982-02-22   0.780522  0.064463  0.064898  0.064463  0.064898   26633600.0 -0.267328
    1982-05-04   0.780522  0.054881  0.055316  0.054881  0.054881   73987200.0 -0.432429
    1982-07-15   0.780522  0.044427  0.044863  0.044427  0.044427   65788800.0 -0.471500
    1982-09-24   0.780522  0.063156  0.063591  0.063156  0.063591  178192000.0  0.198493
    """

    @classmethod
    def compute(cls, data: pd.DataFrame, window=10) -> List[pd.Series]:
        N = data['Close'].diff(window)
        D = data['Close'].shift(window)
        ROC = pd.Series(N / D, name=f'ROC({window})')
        return [ROC]


class APO(Indicator):

    @classmethod
    def compute(cls, data: pd.DataFrame, fastperiod=12, slowperiod=26, matype=0) -> List[pd.Series]:
        real = ta.APO(data["Close"], fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)

        return [real]


class AROON(Indicator):

    @classmethod
    def compute(cls, data: pd.DataFrame, window=14) -> List[pd.Series]:
        aroondown, aroonup = ta.AROON(data["High"], data["Low"], timeperiod=window)
        aroondown.name = f"aroondown({window})"
        aroonup.name = f"aroonup({window})"
        return [aroondown, aroonup]


class AROONOSC(Indicator):

    @classmethod
    def compute(cls, data: pd.DataFrame, window=14) -> List[pd.Series]:
        real = ta.AROONOSC(data["High"], data["Low"], timeperiod=window)

        return [real]


class BOP(Indicator):

    @classmethod
    def compute(cls, data: pd.DataFrame, **kwargs) -> List[pd.Series]:
        real = ta.BOP(data["Open"], data["High"], data["Low"], data["Close"])

        return [real]


class CMO(Indicator):

    @classmethod
    def compute(cls, data: pd.DataFrame, window=14) -> List[pd.Series]:
        real = ta.CMO(data["Close"], timeperiod=window)

        return [real]


class DX(Indicator):

    @classmethod
    def compute(cls, data: pd.DataFrame, window=14) -> List[pd.Series]:
        real = ta.DX(data["High"], data["Low"], data["Close"], timeperiod=window)

        return [real]


class MFI(Indicator):

    @classmethod
    def compute(cls, data: pd.DataFrame, window=14) -> List[pd.Series]:
        real = ta.MFI(data["High"], data["Low"], data["Close"], data["Volume"], timeperiod=window)

        return [real]


class MOM(Indicator):

    @classmethod
    def compute(cls, data: pd.DataFrame, window=14) -> List[pd.Series]:
        real = ta.MOM(data["Close"], timeperiod=window)

        return [real]


class PPO(Indicator):

    @classmethod
    def compute(cls, data: pd.DataFrame, fastperiod=12, slowperiod=26, matype=0) -> List[pd.Series]:
        real = ta.PPO(data["Close"], fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)

        return [real]


class ROCP(Indicator):

    @classmethod
    def compute(cls, data: pd.DataFrame, window=10) -> List[pd.Series]:
        real = ta.ROCP(data["Close"], timeperiod=window)

        return [real]


class ROCR(Indicator):

    @classmethod
    def compute(cls, data: pd.DataFrame, window=10) -> List[pd.Series]:
        real = ta.ROCR(data["Close"], timeperiod=window)

        return [real]


class RSI(Indicator):

    @classmethod
    def compute(cls, data: pd.DataFrame, window=14) -> List[pd.Series]:
        real = ta.RSI(data["Close"], timeperiod=window)

        return [real]


class STOCH(Indicator):

    @classmethod
    def compute(cls, data: pd.DataFrame, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3,
                slowd_matype=0) -> List[pd.Series]:
        slowk, slowd = ta.STOCH(data["High"], data["Low"], data["Close"],
                                fastk_period=fastk_period, slowk_period=slowk_period,
                                slowk_matype=slowk_matype, slowd_period=slowd_period, slowd_matype=slowd_matype)
        slowk.name = f"STOCH_slowk_{fastk_period}_{slowk_period}_{slowd_period}"
        slowd.name = f"STOCH_slowd_{fastk_period}_{slowk_period}_{slowd_period}"
        return [slowk, slowd]


class STOCHF(Indicator):

    @classmethod
    def compute(cls, data: pd.DataFrame, fastk_period=5, fastd_period=3, fastd_matype=0) -> List[pd.Series]:
        fastk, fastd = ta.STOCHF(data["High"], data["Low"], data["Close"],
                                 fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=fastd_matype)
        fastk.name = f"STOCHF_fastk_{fastk_period}_{fastd_period}"
        fastd.name = f"STOCHF_fastd_{fastk_period}_{fastd_period}"
        return [fastk, fastd]


class STOCHRSI(Indicator):

    @classmethod
    def compute(cls, data: pd.DataFrame, window=14, fastk_period=5, fastd_period=3, fastd_matype=0) -> List[pd.Series]:
        fastk, fastd = ta.STOCHRSI(data["Close"],
                                   timeperiod=window, fastk_period=fastk_period,
                                   fastd_period=fastd_period, fastd_matype=fastd_matype)
        fastk.name = f"STOCHRSI_fastk_{window}_{fastk_period}_{fastd_period}"
        fastd.name = f"STOCHRSI_fastd_{window}_{fastk_period}_{fastd_period}"
        return [fastk, fastd]


class TRIX(Indicator):

    @classmethod
    def compute(cls, data: pd.DataFrame, window=30) -> List[pd.Series]:
        real = ta.TRIX(data["Close"], timeperiod=window)

        return [real]


class WILLR(Indicator):

    @classmethod
    def compute(cls, data: pd.DataFrame, window=14) -> List[pd.Series]:
        real = ta.WILLR(data["High"], data["Low"], data["Close"], timeperiod=window)

        return [real]


class AD(Indicator):

    @classmethod
    def compute(cls, data: pd.DataFrame, **kwargs) -> List[pd.Series]:
        real = ta.AD(data["High"], data["Low"], data["Close"], data["Volume"])

        return [real]


class ADOSC(Indicator):

    @classmethod
    def compute(cls, data: pd.DataFrame, fastperiod=3, window=10) -> List[pd.Series]:
        real = ta.ADOSC(data["High"], data["Low"], data["Close"], data["Volume"],
                        fastperiod=fastperiod, slowperiod=window)

        return [real]


class OBV(Indicator):

    @classmethod
    def compute(cls, data: pd.DataFrame, **kwargs) -> List[pd.Series]:
        real = ta.OBV(data["Close"], data["Volume"])

        return [real]


class ATR(Indicator):

    @classmethod
    def compute(cls, data: pd.DataFrame, window=14) -> List[pd.Series]:
        real = ta.ATR(data["High"], data["Low"], data["Close"], timeperiod=window)

        return [real]


class NATR(Indicator):

    @classmethod
    def compute(cls, data: pd.DataFrame, window=14) -> List[pd.Series]:
        real = ta.NATR(data["High"], data["Low"], data["Close"], timeperiod=window)

        return [real]


class TRANGE(Indicator):

    @classmethod
    def compute(cls, data: pd.DataFrame, **kwargs) -> List[pd.Series]:
        real = ta.TRANGE(data["High"], data["Low"], data["Close"])

        return [real]
