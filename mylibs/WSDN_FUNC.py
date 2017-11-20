#coding:utf-8 
import pandas as pd 

def transform_df(df):
    df = pd.DataFrame(df)
    df = df.sort_values(by=['date'], ascending=[False])
    df = df.reset_index(drop=True)
    df = df.drop_duplicates(subset=['msno'], keep='first')
    return df

def transform_df2(df):
    df = df.sort_values(by=['date'], ascending=[False])
    df = df.reset_index(drop=True)
    df = df.drop_duplicates(subset=['msno'], keep='first')
    return df

def user_log_count(df):
    """
        user log count
    """
    df["date"] = pd.to_datetime(df["date"], format='%Y%m%d')
    df = df.sort_values(by=['date'], ascending=[True])
    df = df.set_index("date").truncate(before = '2017-01-01').reset_index()
    num_cols = [col for col in df.columns if col not in ["msno", "date"]]
    df_group = df.groupby("msno")
    df_group_sum = df_group[num_cols].sum().reset_index()
    df_group_mean = df_group[num_cols].mean().reset_index(drop=True)

    cols_map = dict([[col, col+"_sum"] for col in df_group_sum.columns if col != "msno"])
    df_group_sum = df_group_sum.rename(columns=cols_map)
    cols_map = dict([[col, col+"_mean"] for col in df_group_mean.columns if col != "msno"])
    df_group_mean = df_group_mean.rename(columns=cols_map)

    df_group_concat = pd.concat([df_group_sum, df_group_mean], axis=1)

    return df_group_concat

def user_log_count2(df):
    sum_cols = [col for col in df.columns if col.endswith("sum")]
    mean_cols = [col for col in df.columns if col.endswith("mean")]
    df_group = df.groupby("msno")
    df_group_sum = df_group[sum_cols].sum().reset_index()
    df_group_mean = df_group[mean_cols].mean().reset_index(drop=True)
    df_group_concat = pd.concat([df_group_sum, df_group_mean], axis=1)

    return df_group_concat

def user_log_trend(df):
    """
        user log trend.
    """
    def trend_op(data):
        cumsum = 0
        for i in range(len(data) - 1):
            x = data.iloc[i]
            y = data.iloc[i+1]
            cumsum += 1 if x <= y else -1
        return cumsum
    df["date"] = pd.to_datetime(df["date"], format='%Y%m%d')
    df = df.sort_values(by=['date'], ascending=[True])
    df = df.set_index("date").truncate(before = '2017-01-01').reset_index()

    num_cols = [col for col in df.columns if col not in ["msno", "date"]]
    df_group = df.groupby("msno")
    df_trend = df_group[num_cols].agg(trend_op).reset_index()
    cols_map = dict([[col, col+"_trend"] for col in df_trend.columns if col != "msno"])
    df_trend = df_trend.rename(columns=cols_map)
    return df_trend

def user_log_trend2(df):
    def trend_op2(data):
        cumsum = 0
        for item in data:
            cumsum += item
        return cumsum
    df_group = df.groupby("msno")
    df_trend = df_group.agg(trend_op2).reset_index()
    return df_trend

def user_log_num_count(df):
    df["date"] = pd.to_datetime(df["date"], format='%Y%m%d')
    df = df.sort_values(by=['date'], ascending=[True])
    df = df.set_index("date").truncate(before = '2017-02-01').reset_index()
    df = df["msno"].value_counts().reset_index().rename(columns={"index":"msno", "msno":"log_count"})
    return df

def user_log_num_count2(df):
    df = df.groupby("msno")["log_count"].sum().reset_index()
    return df

def square(a): return a ** 2

if __name__ == "__main__":
	from multiprocessing import Pool, cpu_count
	p = Pool(cpu_count())
	l = range(100)
	square_l = p.map(square, l)
	p.close()
	p.join()
	print(square_l)