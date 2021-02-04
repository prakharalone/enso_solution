from datetime import datetime
from xpms_file_storage.file_handler import XpmsResourceFactory, XpmsResource, LocalResource
import pandas as pd
import numpy as np
from xpms_storage.db_handler import DBProvider
import json
from datetime import datetime
import time
import requests


def hma_optimized_postprocess_insert_db(config=None, **obj):
    batch_name = config['context']['batch_name']
    start_time = int(config['context']['start_time'])
    threshold = float(config['context']['threshold'])
    file_name = config['context']['input_file_name']
    converted_start_time = datetime.utcfromtimestamp(start_time)

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
    local_csv_path = "/tmp/vmAuditTest.csv"
    # file_path = config["context"]["source_file_path"]
    # local_csv_path = config["context"]["local_source_file_path"]
    xr1 = XpmsResource()
    minio_resource = xr1.get(urn=file_path)
    local_res = LocalResource(key=local_csv_path)
    minio_resource.copy(local_res)
    df1 = pd.read_csv(local_csv_path)
    df1 = df1.apply(lambda x: (round(x * 100, 2)) / 100)

    file_path = result_paths[1]
    local_csv_path = "/tmp/vmAuditTest.csv"
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
    labels = {'0': 'Coding Error', '1': 'Frequency - Claim Error',
              '2': 'Frequency - Money Claim Error', '3': 'Internal Error',
              }

    final_df.rename(columns=labels, inplace=True)
    aggregated_df = final_df.groupby('CLAIM_NUMBER_Mask').agg(
        lambda x: list(x) if len(set(x)) == 1 else list(x)).replace(
        'nan', '').reset_index()

    lcols = ['CAF', 'CFE', 'Coding Error', 'Frequency - Claim Error', 'Frequency - Money Claim Error', 'Internal Error'
        ]

    #     for col in lcols:
    #         aggregated_df[col] = aggregated_df[col].apply(lambda x: x if type(x) == list else [x])

    aggregated_df['index_choice'] = aggregated_df['CFE'].apply(lambda x: x.index(max(x)))

    for col in lcols:
        aggregated_df[col + '_confidence'] = aggregated_df.apply(lambda row: row[col][row['index_choice']], axis=1)
    rename_cols = {"CAF": "aggregated_caf",
                   "CFE": "aggregated_cfe",
                   "Coding Error": "aggregated_Coding Error",
                   "Frequency - Claim Error": "aggregated_Frequency - Claim Error",
                   "Frequency - Money Claim Error": "aggregated_Frequency - Money Claim Error",
                   "Internal Error": "aggregated_Internal Error",
                   "CAF_confidence": "CAF",
                   "CFE_confidence": "CFE",
                   "Coding Error_confidence": "Coding Error",
                   "Frequency - Claim Error_confidence": "Frequency - Claim Error",
                   "Frequency - Money Claim Error_confidence": "Frequency - Money Claim Error",
                   "Internal Error_confidence": "Internal Error",
                   }
    aggregated_df.rename(columns=rename_cols, inplace=True)
    aggregated_df["AFV_PRV_CRG_AMT_1"] = [sum(x) for x in aggregated_df["AFV_PRV_CRG_AMT_1"]]
    claim_error_columns = ['Coding Error', 'Frequency - Claim Error', 'Frequency - Money Claim Error', 'Internal Error'
       ]

    aggregated_df["Audit Result"] = aggregated_df["CAF"].apply(
        lambda x: "CAF" if x >= float(threshold) / 100 else "CFE")
    aggregated_df["Error Result"] = aggregated_df[claim_error_columns].idxmax(axis=1)

    final_df["Audit Result"] = final_df['CLAIM_NUMBER_Mask'].map(aggregated_df.set_index('CLAIM_NUMBER_Mask')['Audit Result'])
    final_df["Error Result"] = final_df['CLAIM_NUMBER_Mask'].map(aggregated_df.set_index('CLAIM_NUMBER_Mask')['Error Result'])

    clean_df = aggregated_df
    line_level_df = final_df

    #     clean_df["Audit Result"] = clean_df[["CAF", "CFE"]].idxmax(axis=1)
    clean_df["Audit Result"] = clean_df["CAF"].apply(lambda x: "CAF" if x >= float(threshold) / 100 else "CFE")
    clean_df["Error Result"] = clean_df[claim_error_columns].idxmax(axis=1)

    clean_df.rename(columns={"CLAIM_NUMBER_Mask":"CLAIM NUMBER"}, inplace=True)
    clean_df.rename(columns={"AFV_PRV_CRG_AMT_1":"TOT PROV CHARGE"}, inplace=True)


    clean_df["error_bucket"] = clean_df[claim_error_columns].to_dict(orient='records')
    clean_df.drop(columns=claim_error_columns, inplace=True)
    clean_df["system_recommended_result"] = clean_df[["CAF", "CFE"]].to_dict(orient='records')

    clean_ob = json.loads(clean_df.to_json(orient='records'))
    line_level_ob = json.loads(line_level_df.to_json(orient='records'))

    clean_ob_lst = [
        {'batch_name': batch_name, 'data': item, 'start_time': start_time, "converted_start_time": converted_start_time,
         'threshold': float(threshold / 100),
         'file_name': file_name, 'flag': 'untrained'} for item in clean_ob]
    line_level_lst = [
        {'batch_name': batch_name, 'data': item, 'start_time': start_time, "converted_start_time": converted_start_time,
         'threshold': float(threshold / 100),
         'file_name': file_name, 'flag': 'untrained'} for item in line_level_ob]

    try:
        db = DBProvider.get_instance(db_name='ca_be')
        s1 = db.insert(table='claims_data', rows=clean_ob_lst)
        s3 = db.insert(table='line_level_claims_data', rows=line_level_lst)

        update_ob = {"$set": {'audit_not_needed': clean_df[clean_df['CAF'] >= threshold / 100].shape[0],
                              'audit_needed': clean_df[clean_df['CAF'] < threshold / 100].shape[0],
                              "status": "in-progress"}}
        filter_ob = {"batch_name": batch_name}

        s2 = db.update(table='batch_metadata', update_obj=update_ob, filter_obj=filter_ob)

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

        celery_batch_url = "https://claimsaudit-be.claims-audit.enterprise.xpms.ai/celery/batch-ingested-calculation"

        payload = {}
        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.request("GET", celery_batch_url, headers=headers, data=payload)

        print(response.text.encode('utf8'))

        url = "https://claimsaudit-be.claims-audit.enterprise.xpms.ai/send_notification"
        headers = {
            'Content-Type': 'application/json'
        }

        resp = requests.request("POST", url, headers=headers, data=json.dumps(notification, default=str))

        return {
            "status": "completed",
            "claims_db_inserted": s1,
            "line_level_claims_data_inserted": s3,
            "batch_metadata_updated": s2,
            "notification_resp": resp.text
        }
    except Exception as e:
        return {
            "status": "failed",
            "claims_db_inserted": False,
            "line_level_claims_data_inserted": False,
            "batch_metadata_updated": False,
            "error_message": str(e)
        }



