from datetime import datetime
from xpms_file_storage.file_handler import XpmsResourceFactory, XpmsResource, LocalResource
import pandas as pd
import numpy as np


def process_value(x):
    if len(set(x)) == 1:
        return list(x)
    else:
        return list(x)


def preprocess_aggregated_df(df):
    df["CAF"] = [x if type(x) == list else [x] for x in df['CAF']]
    df["CFE"] = [x if type(x) == list else [x] for x in df['CFE']]
    df["AGGREGATED_CAF"] = df["CAF"]
    df["AGGREGATED_CFE"] = df["CFE"]
    df["CAF"] = 1 - df.AGGREGATED_CFE.apply(max)
    df["CFE"] = df.AGGREGATED_CFE.apply(max)

    df["benefit_error"] = [x if type(x) == list else [x] for x in df['benefit_error']]
    df["cob_error"] = [x if type(x) == list else [x] for x in df['cob_error']]
    df["provider_error"] = [x if type(x) == list else [x] for x in df['provider_error']]
    df["service_error"] = [x if type(x) == list else [x] for x in df['service_error']]
    df["subscriber_error"] = [x if type(x) == list else [x] for x in df['subscriber_error']]

    df["aggregated_benefit_error"] = df["benefit_error"]
    df["aggregated_cob_error"] = df["cob_error"]
    df["aggregated_provider_error"] = df["provider_error"]
    df["aggregated_service_error"] = df["service_error"]
    df["aggregated_subscriber_error"] = df["subscriber_error"]

    df["index_choice"] = [x.index(max(x)) for x in df["AGGREGATED_CFE"]]
    df["benefit_error"] = [x[df['index_choice'][i]] for i, x in enumerate(df['aggregated_benefit_error'])]
    df["cob_error"] = [x[df['index_choice'][i]] for i, x in enumerate(df['aggregated_cob_error'])]
    df["provider_error"] = [x[df['index_choice'][i]] for i, x in enumerate(df['aggregated_provider_error'])]
    df["service_error"] = [x[df['index_choice'][i]] for i, x in enumerate(df['aggregated_service_error'])]
    df["subscriber_error"] = [x[df['index_choice'][i]] for i, x in enumerate(df['aggregated_subscriber_error'])]

    df["TOT PROV CHARGE"] = [x[0] for x in df["TOT PROV CHARGE"]]

    return df


def hma_post_process_2(config=None, **obj):
    batch_name = config['context']['batch_name']
    start_time = config['context']['start_time']
    threshold = config['context']['threshold']
    file_name = config['context']['input_file_name']

    file_path = config["context"]["source_file_path"]
    local_csv_path = config["context"]["local_source_file_path"]
    xr1 = XpmsResource()
    minio_resource = xr1.get(urn=file_path)
    local_res = LocalResource(key=local_csv_path)
    minio_resource.copy(local_res)
    df = pd.read_csv(local_csv_path)

    result_paths = []
    for key in obj:
        try:
            result_paths.append(obj[key]['result_path'])
        except KeyError as e:
            print('KeyError')

    # Result 1
    file_path = result_paths[0]
    local_csv_path = "/tmp/ vmAuditTest.csv"
    # file_path = config["context"]["source_file_path"]
    # local_csv_path = config["context"]["local_source_file_path"]
    xr1 = XpmsResource()
    minio_resource = xr1.get(urn=file_path)
    local_res = LocalResource(key=local_csv_path)
    minio_resource.copy(local_res)
    df1 = pd.read_csv(local_csv_path)
    df1 = df1.apply(lambda x: (round(x * 100, 3)) / 100)

    file_path = result_paths[1]
    local_csv_path = "/tmp/ vmAuditTest.csv"
    xr1 = XpmsResource()
    minio_resource = xr1.get(urn=file_path)
    local_res = LocalResource(key=local_csv_path)
    minio_resource.copy(local_res)
    df2 = pd.read_csv(local_csv_path)
    df2 = df2.apply(lambda x: (round(x * 100, 3)) / 100)

    if len(df1.columns) == 3:
        final_df = pd.concat([df, df1.drop(df1.columns[0], axis=1), df2.drop(df2.columns[0], axis=1)], axis=1)
    else:
        final_df = pd.concat([df, df2.drop(df2.columns[0], axis=1), df1.drop(df1.columns[0], axis=1)], axis=1)

    final_df.fillna(0, inplace=True)
    aggregated_df = final_df.groupby('CLAIM NUMBER').agg(lambda x: process_value(x)).reset_index()

    aggregated_df = preprocess_aggregated_df(aggregated_df)

    claim_error_columns = ['benefit_error', 'cob_error',
                           'provider_error', 'service_error', 'subscriber_error']

    # aggregated_df["Audit Result"] = aggregated_df[["CAF", "CFE"]].idxmax(axis=1)
    aggregated_df["Audit Result"] = aggregated_df["CAF"].apply(
        lambda x: "CAF" if x >= float(threshold) / 100 else "CFE")
    aggregated_df["Error Result"] = aggregated_df[claim_error_columns].idxmax(axis=1)

    final_df["Audit Result"] = [i for i in range(final_df.shape[0])]
    final_df["Error Result"] = [i for i in range(final_df.shape[0])]
    for index in range(len(aggregated_df)):
        for i in range(len(df1)):
            if aggregated_df["CLAIM NUMBER"][index] == final_df["CLAIM NUMBER"][i]:
                final_df["Audit Result"][i] = aggregated_df["Audit Result"][index]
                final_df["Error Result"][i] = aggregated_df["Error Result"][index]

    return {
        'data': {
            "data_format": "list",
            "values": [aggregated_df.columns.values.tolist()] + aggregated_df.values.tolist(),
            "line_level_data": [final_df.columns.values.tolist()] + final_df.values.tolist()
        }
    }