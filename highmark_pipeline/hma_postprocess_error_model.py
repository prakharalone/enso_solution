from datetime import datetime
from xpms_file_storage.file_handler import XpmsResourceFactory, XpmsResource, LocalResource
import pandas as pd
import numpy as np
from xpms_storage.db_handler import DBProvider
import json
from datetime import datetime
import time
import requests
import copy
import pickle
from xpms_storage.utils import get_env


def hma_postprocess_error_model(config=None, **obj):

    NAMESPACE = get_env("NAMESPACE", "claims-audit", False)
    DOMAIN_NAME = get_env("DOMAIN_NAME", "enterprise.xpms.ai", False)
    ENV_DATABASE = get_env('DATABASE_PARAPHRASE', None, True)

    batch_name = config['context']['batch_name']
    start_time = int(config['context']['start_time'])
    threshold = float(config['context']['threshold'])
    file_name = config["context"]["cfe_source_file"].split('/')[-1]

    converted_start_time = datetime.utcfromtimestamp(start_time)

    file_path = config["context"]["cfe_source_file"]
    local_csv_path = "/tmp/cfe_data.csv"

    xr1 = XpmsResource()
    minio_resource = xr1.get(urn=file_path)
    local_res = LocalResource(key=local_csv_path)
    minio_resource.copy(local_res)
    df = pd.read_csv(local_csv_path)

    # Result 1
    result_path = obj['result_path']
    local_csv_path = "/tmp/vmAuditTest.csv"
    # file_path = config["context"]["source_file_path"]
    # local_csv_path = config["context"]["local_source_file_path"]
    xr1 = XpmsResource()
    minio_resource = xr1.get(urn=result_path)
    local_res = LocalResource(key=local_csv_path)
    minio_resource.copy(local_res)
    df1 = pd.read_csv(local_csv_path)
    df1 = df1.apply(lambda x: (round(x * 100, 2)) / 100)

    final_df = pd.concat([df, df1.drop(df1.columns[0], axis=1)], axis=1)

    labels = {'0': 'Coding Error', '1': 'Frequency - Claim Error',
              '2': 'Frequency - Money Claim Error', '3': 'Internal Error'
             }

    final_df.rename(columns=labels, inplace=True)

    aggregated_df = final_df.groupby('CLAIM_NUMBER_Mask').agg(
        lambda x: list(x) if len(set(x)) == 1 else list(x)).replace(
        'nan', '').reset_index()
    lcols = ['Coding Error', 'Frequency - Claim Error', 'Frequency - Money Claim Error', 'Internal Error'
        ]

    #     for col in lcols:
    #         aggregated_df[col] = aggregated_df[col].apply(lambda x: x if type(x) == list else [x])

    aggregated_df['index_choice'] = [x[0] for x in aggregated_df["index_choice"]]
    #
    for col in lcols:
        aggregated_df[col + '_confidence'] = aggregated_df.apply(lambda row: row[col][row['index_choice']], axis=1)
    rename_cols = {"Coding Error": "aggregated_Coding Error",
                   "Frequency - Claim Error": "aggregated_Frequency - Claim Error",
                   "Frequency - Money Claim Error": "aggregated_Frequency - Money Claim Error",
                   "Internal Error": "aggregated_Internal Error",

                   "Coding Error_confidence": "Coding Error",
                   "Frequency - Claim Error_confidence": "Frequency - Claim Error",
                   "Frequency - Money Claim Error_confidence": "Frequency - Money Claim Error",
                   "Internal Error_confidence": "Internal Error",

                   }
    aggregated_df.rename(columns=rename_cols, inplace=True)
    aggregated_df["AFV_PRV_CRG_AMT_1"] = [sum(x) for x in aggregated_df["AFV_PRV_CRG_AMT_1"]]
    claim_error_columns = ['Coding Error', 'Frequency - Claim Error', 'Frequency - Money Claim Error', 'Internal Error'
        ]

    aggregated_df["Error Result"] = aggregated_df[claim_error_columns].idxmax(axis=1)
    aggregated_df["error_bucket"] = aggregated_df[claim_error_columns].to_dict(orient='records')

    final_df["Error Result"] = final_df['CLAIM_NUMBER_Mask'].map(aggregated_df.set_index('CLAIM_NUMBER_Mask')['Error Result'])

    aggregated_df_copy = aggregated_df.copy(deep=True)
    final_df_copy = final_df.copy(deep=True)
    rename_c = {
        "CLAIM_NUMBER_Mask": "CLAIM NUMBER",
        "AFV_PRV_CRG_AMT_1": "TOT PROV CHARGE",
    }
    aggregated_df_copy.rename(columns=rename_c, inplace=True)
    aggregated_df_copy.drop(columns=claim_error_columns, inplace=True)

    # final_df_copy.rename(columns=rename_c,inplace=True)

    claim_ids = df["CLAIM_NUMBER_Mask"].unique().tolist()
    try:
        db = DBProvider.get_instance(db_name=ENV_DATABASE)
        is_claim_present = db.find(table="claims_data", filter_obj={"data.CLAIM NUMBER": {'$in': claim_ids}})
        is_line_level_claim_present = db.find(table="line_level_claims_data",
                                              filter_obj={"data.CLAIM_NUMBER_Mask": {'$in': claim_ids}})

        for claim in is_claim_present:
            claim["data"]["Error Result"] = aggregated_df_copy[aggregated_df_copy['CLAIM NUMBER'] == claim["data"]["CLAIM NUMBER"]]["Error Result"].iloc[0]
            claim["data"]["error_bucket"] = aggregated_df_copy[aggregated_df_copy['CLAIM NUMBER'] == claim["data"]["CLAIM NUMBER"]]['error_bucket'].iloc[0]

        for claim in is_line_level_claim_present:
            claim["data"]["Error Result"] = final_df_copy[final_df_copy['CLAIM_NUMBER_Mask'] == claim["data"]["CLAIM_NUMBER_Mask"]]['Error Result'].iloc[0]

        r1 = db.delete(table='claims_data', filter_obj={"data.CLAIM NUMBER": {'$in': claim_ids}})
        r2 = db.delete(table='line_level_claims_data', filter_obj={"data.CLAIM_NUMBER_Mask": {'$in': claim_ids}})

        if r1 and r2:
            s1 = db.insert(table='claims_data', rows=is_claim_present)
            s3 = db.insert(table='line_level_claims_data', rows=is_line_level_claim_present)

        notification = {
            "group": "batch_status",
            "message": {
                "batch_name": batch_name,
                "current_status": "in-progress",
                "previous_status": "to-do"
            },
            "created_timestamp": int(time.time())
        }

        s = db.insert(table='notifications', rows=[notification])

        celery_batch_url = "https://claimsaudit-be.{0}.{1}/celery/batch-ingested-calculation".format(NAMESPACE,DOMAIN_NAME)

        payload = {}
        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.request("GET", celery_batch_url, headers=headers, data=payload)

        url = "https://claimsaudit-be.{0}.{1}/send_notification".format(NAMESPACE,DOMAIN_NAME)
        headers = {
            'Content-Type': 'application/json'
        }

        resp = requests.request("POST", url, headers=headers, data=json.dumps(notification, default=str))

        return {
            "status": "completed",
            "claims_db_inserted": s1,
            "line_level_claims_data_inserted": s3,
            "notification_resp": resp.text,
        }

    except Exception as e:
        return {
            "status": "failed",
            "claims_db_inserted": False,
            "line_level_claims_data_inserted": False,
            "error_message": str(e)
        }