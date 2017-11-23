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

def sklearn_kernel(parallel_data):
    """
    sklearn_kernel.
    """
    import gc

    clf = parallel_data['clf']
    predict_method = parallel_data['predict_method']
    round_i = parallel_data['round_i']
    test_index = parallel_data['test_index']
    train_data_x = parallel_data['train_data_x']
    train_data_y = parallel_data['train_data_y']
    training_test_data_x = parallel_data['training_test_data_x']
    training_test_data_y = parallel_data['training_test_data_y']
    test_x = parallel_data['test_x']
    training_prediction = parallel_data['training_prediction']
    test_prediction = parallel_data['test_prediction']

    if 'parallel_data_name' in parallel_data:
        parallel_data_name = parallel_data['parallel_data_name']
        globalDict = globals()
        print(globalDict)
        print(parallel_data_name in globalDict)
        if parallel_data_name in globalDict:
            del globalDict[parallel_data_name]
            gc.collect()
            print("all local variables are invisible now.")
    clf.fit(train_data_x, train_data_y)
    del train_data_x
    del train_data_y
    gc.collect()
    print("round%d training finished." % round_i)

    if predict_method == "proba":
        training_prediction[test_index] = clf.predict_proba(training_test_data_x)
        test_prediction[round_i, :] = clf.predict_proba(test_x)
    elif predict_method == "log_proba":
        training_prediction[test_index] = clf.predict_log_proba(training_test_data_x)
        test_prediction[round_i, :] = clf.predict_log_proba(test_x)
    else:
        training_prediction[test_index] = clf.predict(training_test_data_x).reshape(-1, 1)
        test_prediction[round_i, :] = clf.predict(test_x).reshape(-1, 1)    

    del training_test_data_x
    del training_test_data_y
    del test_index
    gc.collect()

def square(a): return a ** 3

if __name__ == "__main__":
	from multiprocessing import Pool, cpu_count
	p = Pool(cpu_count())
	l = range(100)
	square_l = p.map(square, l)
	p.close()
	p.join()
	print(square_l)