from pyspark.sql import SparkSession
import pandas as pd
import os

def main():
    spark = SparkSession.builder.appName("ParseData").getOrCreate()

    hdfs_directory = "/phase2/data"
    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
    path = spark._jvm.org.apache.hadoop.fs.Path(hdfs_directory)
    files = fs.listStatus(path)
    
    pandas_dfs, paths = [], []
    for file in files:
        path = file.getPath().toString()
        paths.append(path)
        spark_df = read_file(spark, path)
        if spark_df is not None:
            pandas_dfs.append(spark_df.toPandas())

    universal_mapping = {
            '1': 1,
            '0': 0,
            '-1': -1,
    }

    col_names = ['content', 'sentiment']
    test_names = ['content']

    train = pd.DataFrame(columns=col_names)
    valid_labels = pd.DataFrame(columns=col_names)
    valid_none = pd.DataFrame(columns=test_names)
    test = pd.DataFrame(columns=test_names)

    split_ratio = 0.8
    key = 0
    
    # deals with 1_test.csv
    pandas_dfs[key] = pandas_dfs[key].rename(columns={'Tweets': col_names[0]})
    valid_none = pd.concat([valid_none, pandas_dfs[key]], ignore_index=True)
    key += 1

    # deals with 1_train.csv
    mapping = {
            'Positive emotion': 1,
            'No emotion towards brand or product': 0,
            'Negative emotion': -1
    }
    temp_cols = ['tweet_text', 'is_there_an_emotion_directed_at_a_brand_or_product']
    pandas_dfs[key] = pandas_dfs[key].drop(columns=['emotion_in_tweet_is_directed_at'])
    pandas_dfs[key][temp_cols[1]] = (pandas_dfs[key][temp_cols[1]].map(mapping))
    pandas_dfs[key] = pandas_dfs[key].rename(columns={temp_cols[0]: col_names[0], temp_cols[1]: col_names[1]})
    train = pd.concat([train, pandas_dfs[key]], ignore_index=True)
    key += 1

    # deals with 2.csv
    pandas_dfs[key] = pandas_dfs[key].drop(columns=['textID', 'selected_text'])
    pandas_dfs[key] = pandas_dfs[key].rename(columns={'text': 'content'})
    pandas_dfs[key][col_names[1]] = (pandas_dfs[key][col_names[1]].map(universal_mapping))
    split = int(split_ratio * len(pandas_dfs[key]))
    t = pandas_dfs[key].iloc[:split]
    v = pandas_dfs[key].iloc[split:]
    train = pd.concat([train, t], ignore_index=True)
    valid_labels = pd.concat([valid_labels, v], ignore_index=True)
    key += 1

    # deals with 3.csv
    t = pandas_dfs[key]
    pandas_dfs[key] = t[t['sentiment'] != '2']
    pandas_dfs[key] = pandas_dfs[key].drop(columns=['tweetid'])
    pandas_dfs[key] = pandas_dfs[key].rename(columns={'message': col_names[0]})
    pandas_dfs[key][col_names[1]] = (pandas_dfs[key][col_names[1]].map(universal_mapping))
    split = int(split_ratio * len(pandas_dfs[key]))
    t = pandas_dfs[key].iloc[:split]
    v = pandas_dfs[key].iloc[split:]
    train = pd.concat([train, t], ignore_index=True)
    valid_labels = pd.concat([valid_labels, v], ignore_index=True)
    key += 1

    # deals with 4.csv
    pandas_dfs[key] = pandas_dfs[key].rename(columns={'clean_text': col_names[0], 'category': col_names[1]})
    pandas_dfs[key][col_names[1]] = (pandas_dfs[key][col_names[1]].map(universal_mapping))
    split = int(split_ratio * len(pandas_dfs[key]))
    t = pandas_dfs[key].iloc[:split]
    v = pandas_dfs[key].iloc[split:]
    train = pd.concat([train, t], ignore_index=True)
    valid_labels = pd.concat([valid_labels, v], ignore_index=True)
    key += 1

    # deals with 6_test.csv
    mapping = {
            '0': -1,
            '1': 1,
            '2': 0
    }
    pandas_dfs[key]['label'] = (pandas_dfs[key]['label'].map(mapping))
    pandas_dfs[key] = pandas_dfs[key].rename(columns={'text': col_names[0], 'label': col_names[1]})
    valid_labels = pd.concat([valid_labels, pandas_dfs[key]], ignore_index=True)
    key += 1

    # deals with 6_train.csv
    pandas_dfs[key]['label'] = (pandas_dfs[key]['label'].map(mapping))
    pandas_dfs[key] = pandas_dfs[key].rename(columns={'text': col_names[0], 'label': col_names[1]})
    train = pd.concat([train, pandas_dfs[key]], ignore_index=True)
    key += 1

    # deals with 71.csv
    mapping = {
            0.0: -1,
            1.0: 1,
            2.0: 0
    }
    pandas_dfs[key][col_names[1]] = (pandas_dfs[key][col_names[1]].map(mapping))
    pandas_dfs[key][col_names[1]] = pandas_dfs[key][col_names[1]].fillna(0)
    pandas_dfs[key] = pandas_dfs[key].drop(pandas_dfs[key].columns[0], axis=1)
    pandas_dfs[key] = pandas_dfs[key].rename(columns={'tweet': col_names[0]})
    split = int(split_ratio * len(pandas_dfs[key]))
    t = pandas_dfs[key].iloc[:split]
    v = pandas_dfs[key].iloc[split:]
    train = pd.concat([train, t], ignore_index=True)
    valid_labels = pd.concat([valid_labels, v], ignore_index=True)
    key += 1

    # deals with 72.csv
    pandas_dfs[key][col_names[1]] = (pandas_dfs[key][col_names[1]].map(mapping))
    pandas_dfs[key][col_names[1]] = pandas_dfs[key][col_names[1]].fillna(0)
    pandas_dfs[key] = pandas_dfs[key].drop(pandas_dfs[key].columns[0], axis=1)
    pandas_dfs[key] = pandas_dfs[key].rename(columns={'tweet': col_names[0]})
    split = int(split_ratio * len(pandas_dfs[key]))
    t = pandas_dfs[key].iloc[:split]
    v = pandas_dfs[key].iloc[split:]
    train = pd.concat([train, t], ignore_index=True)
    valid_labels = pd.concat([valid_labels, v], ignore_index=True)
    key += 1

    # deals with 8.csv
    pandas_dfs[key][col_names[1]] = (pandas_dfs[key][col_names[1]].map(universal_mapping))
    pandas_dfs[key] = pandas_dfs[key].rename(columns={'text': col_names[0]})
    pandas_dfs[key] = pandas_dfs[key].dropna(subset=[col_names[1]])
    split = int(split_ratio * len(pandas_dfs[key]))
    t = pandas_dfs[key].iloc[:split]
    v = pandas_dfs[key].iloc[split:]
    train = pd.concat([train, t], ignore_index=True)
    valid_labels = pd.concat([valid_labels, v], ignore_index=True)
    key += 1

    # deals with 9.csv
    mapping = {
            'Positive': 1,
            'Negative': -1
    }
    pandas_dfs[key] = pandas_dfs[key].drop(columns=['polarity', 'subjectivity'])
    pandas_dfs[key] = pandas_dfs[key][pandas_dfs[key]['Sentiment'].isin(['Positive', 'Negative'])]
    pandas_dfs[key] = pandas_dfs[key].rename(columns={'Tweet': col_names[0], 'Sentiment': col_names[1]})
    pandas_dfs[key][col_names[1]] = (pandas_dfs[key][col_names[1]].map(mapping))
    split = int(split_ratio * len(pandas_dfs[key]))
    t = pandas_dfs[key].iloc[:split]
    v = pandas_dfs[key].iloc[split:]
    train = pd.concat([train, t], ignore_index=True)
    valid_labels = pd.concat([valid_labels, v], ignore_index=True)
    key += 1

    # deals with tweets.json (the file Dr. Rao gave us)
    test[test_names[0]] = pandas_dfs[key]['full_text']

    print("Train shape: ",train.shape)
    print("Validation with labels shape: ", valid_labels.shape)
    print("Validation no labels shape: ", valid_none.shape)
    print("Test shape: ", test.shape)

    print(pandas_dfs[1].keys())

    # was used in debugging when building the script
    # print(paths[key])
    # print(pandas_dfs[key].keys())
    # print(pandas_dfs[key].head())

def read_file(spark, file_path):
    ext = file_path.split('.')[-1].lower()

    if ext == 'csv':
        return spark.read.csv(file_path, header=True, inferSchema=True)
    elif ext == 'json':
        return spark.read.json(file_path)
    else:
        print(f"Unsupported file type: {ext}")
        return None

if __name__ == '__main__':
    main()
