import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

"""

Welcome to custom pandas functions

"""
def outliers_std(x,columns,OneD=False):
    if columns is None:
        if OneD:
            x = np.where(x > 3*x.std(), 3*x.std(), x)
            x = np.where(x < -3*x.std(), -3*x.std(), x)
            return x
        else:
            columns = x.columns
    for i in columns:
        x[i] = np.where(x[i] > 3*x[i].std(),3*x[i].std(),x[i])
        x[i] = np.where(x[i] < -3*x[i].std(), -3*x[i].std(), x[i])
    return x

def outliers_treatment(x,columns=None,OneD = False):
    """
    This method will remove the outliers by sticking to upper and lower boundaries of the boxplot.
    Time Complxity : O(n)
    n - number of columns
    """
    if columns is None:
        if OneD:
            q1 = x.quantile(0.25)
            q3 = x.quantile(0.75)
            iqr = q3 - q1
            upper = q3 + (1.5 * iqr)
            lower = q1 - (1.5 * iqr)
            x = np.where(x > upper, upper, x)
            x = np.where(x < lower, lower, x)
            return x
        else:
            columns = x.columns
    for i in columns:
        q1 = x[i].quantile(0.25)
        q3 = x[i].quantile(0.75)
        iqr = q3-q1
        upper = q3 + (1.5*iqr)
        lower = q1 - (1.5*iqr)
        x[i] = np.where(x[i] > upper,upper,x[i])
        x[i] = np.where(x[i] < lower, lower, x[i])
    return x

def get_max(temp, l):
    #     temp = temp.reset_index()
    temp = temp.sort_values(ascending=False).reset_index()
    j = 0
    while True:
        if temp['index'][j] in l:
            j += 1
            continue
        else:
            return temp['index'][j], temp[0][j]


def vif_pro(x, Y, max_vif=5, min_change=0.05, verbose=True):
    """
    This is for VIF. Just give few params and relax.
    Inputs:
    x - array of independent variables
    Y - array of dependent variable
    max_vif - optional - up to what limit you want to reduce the VIF
    min_change - optional - min change req to keep a column
    Returns:
    l - list of high VIF and imp columns
    x - the dataframe after the process
    """
    l = []
    i = 0
    while True:
        temp = pd.Series([vif(x, i) for i in range(x.shape[1])], index=x.columns)
        res1 = sm.OLS(Y, x)
        mod1 = res1.fit()
        a = mod1.rsquared
        col, m = get_max(temp, l)
        if m < max_vif:
            break
        x1 = x.drop(col, axis=1)
        res2 = sm.OLS(Y, x1)
        mod2 = res2.fit()
        b = mod2.rsquared
        i += 1
        if verbose:
            print("Cycle :", i, ", col:", col, "max:", m, ", r-sq:", b, ", diff:", a - b)
        if -min_change < (a - b) < min_change:
            x = x.drop(col, axis=1)
        else:
            l.append(col)
    return l, x


def red_mem(df, cat=None, verbose=True):
    """
    This beautiful function can be used to modify the column types and hence reduce the size and the burden on
    dataframes.

    Input:
    df = dataframe
    cat = default []
          any categorical columns to be changed data type
    verbose = default True,
              To inform you the effect percentage

    Output:
    df - data type changed and reduced memory and high performance DF is yours.
    """
    numerics = ['int16', 'int32', 'int64', 'float32', 'float64']
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if cat is not None and col in cat:
            df[col] = df[col].astype("category")
            continue
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    if verbose: print(
        'Mem.usage reduced to {:5.2f} MB {:.1f}% reduction'.format(end_mem, 100 * ((start_mem - end_mem) / start_mem)))
    return df


def to_cat(df, filter=10, exclude=None):
    """
      Converts your dataframe columns to Categorical based on the filter condition.

      Input:
         dataframe
         filter : max number of unique values in the data to qualify for the transformation.
         exclude : excludes these cols even if they met the cond.

      Output:
         dataframe
    """

    for i in df.columns:
        if (max(df[i].value_counts().reset_index().index) <= filter) and (i not in exclude):
            df[i] = df[i].astype("category")
    return df
