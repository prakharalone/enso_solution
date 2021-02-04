from datetime import datetime
from xpms_file_storage.file_handler import XpmsResourceFactory, XpmsResource, LocalResource
import pandas as pd
import numpy as np


def hma_post_process(config=None, **obj):
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
    df1 = df1.apply(lambda x: (round(x * 100, 2)) / 100)

    file_path = result_paths[1]
    local_csv_path = "/tmp/ vmAuditTest.csv"
    # file_path = config["context"]["source_file_path"]
    # local_csv_path = config["context"]["local_source_file_path"]
    xr1 = XpmsResource()
    minio_resource = xr1.get(urn=file_path)
    local_res = LocalResource(key=local_csv_path)
    minio_resource.copy(local_res)
    df2 = pd.read_csv(local_csv_path)
    df2 = df2.apply(lambda x: (round(x * 100, 2)) / 100)

    if len(df1.columns) == 3:
        final_df = pd.concat([df, df1.drop(df1.columns[0], axis=1), df2.drop(df2.columns[0], axis=1)], axis=1)
    else:
        final_df = pd.concat([df, df2.drop(df2.columns[0], axis=1), df1.drop(df1.columns[0], axis=1)], axis=1)

    final_df.fillna('', inplace=True)
    aggregated_df = final_df.groupby('CLAIM NUMBER').agg(lambda x: list(x)[0] if len(set(x)) == 1 else list(x)).replace(
        'nan', '').reset_index()
    lcols = ['CAF', 'CFE', 'benefit_error', 'cob_error', 'provider_error', 'service_error', 'subscriber_error']
    for col in lcols:
        aggregated_df[col] = aggregated_df[col].apply(lambda x: x if type(x) == list else [x])
    aggregated_df['index_choice'] = aggregated_df['CFE'].apply(lambda x: x.index(max(x)))
    for col in lcols:
        aggregated_df[col + '_confidence'] = aggregated_df.apply(lambda row: row[col][row['index_choice']], axis=1)

    rename_cols = {"CAF": "aggregated_caf",
                   "CFE": "aggregated_cfe",
                   "benefit_error": "aggregated_benefit_error",
                   "cob_error": "aggregated_cob_error",
                   "provider_error": "aggregated_provider_error",
                   "service_error": "aggregated_service_error",
                   "subscriber_error": "aggregated_subscriber_error",
                   "CAF_confidence": "CAF",
                   "CFE_confidence": "CFE",
                   "benefit_error_confidence": "benefit_error",
                   "cob_error_confidence": "cob_error",
                   "provider_error_confidence": "provider_error",
                   "service_error_confidence": "service_error",
                   "subscriber_error_confidence": "subscriber_error"
                   }
    aggregated_df.rename(columns=rename_cols, inplace=True)

    return {
        'data': {
            "data_format": "list",
            "values": [aggregated_df.columns.values.tolist()] + aggregated_df.values.tolist(),
        }
    }
