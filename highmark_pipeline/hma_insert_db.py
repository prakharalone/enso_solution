import pandas as pd
import time
from xpms_storage.db_handler import DBProvider
import json


def hma_insert_db(config=None, **obj):
    clean_lst = obj['data']["values"]
    line_level_data = obj["data"]["line_level_data"]

    clean_df = pd.DataFrame(clean_lst[1:], columns=clean_lst[0])
    line_level_df = pd.DataFrame(line_level_data[1:], columns=line_level_data[0])

    batch_name = config['context']['batch_name']
    start_time = config['context']['start_time']
    threshold = float(config['context']['threshold'])
    file_name = config['context']['input_file_name']

    claim_error_columns = ['benefit_error', 'cob_error',
                           'provider_error', 'service_error', 'subscriber_error']

    #     clean_df["Audit Result"] = clean_df[["CAF", "CFE"]].idxmax(axis=1)
    clean_df["Audit Result"] = clean_df["CAF"].apply(lambda x: "CAF" if x >= float(threshold) / 100 else "CFE")
    clean_df["Error Result"] = clean_df[claim_error_columns].idxmax(axis=1)
    clean_df["error_bucket"] = clean_df[claim_error_columns].to_dict(orient='records')
    clean_df.drop(columns=claim_error_columns, inplace=True)
    clean_df["system_recommended_result"] = clean_df[["CAF", "CFE"]].to_dict(orient='records')

    clean_ob = json.loads(clean_df.to_json(orient='records'))
    line_level_ob = json.loads(line_level_df.to_json(orient='records'))

    clean_ob_lst = [
        {'batch_name': batch_name, 'data': item, 'start_time': start_time, 'threshold': float(threshold / 100),
         'file_name': file_name, 'flag': 'untrained'} for item in clean_ob]
    line_level_lst = [
        {'batch_name': batch_name, 'data': item, 'start_time': start_time, 'threshold': float(threshold / 100),
         'file_name': file_name, 'flag': 'untrained'} for item in line_level_ob]

    try:
        notification = {
            "group": "batch_status",
            "message": {
                "batch_name": batch_name,
                "current_status": "in-progress",
                "previous_status": "to-do"
            }
        }
        db = DBProvider.get_instance(db_name='hma_ca')
        s = db.insert(table='notifications', rows=[notification])
    except Exception as e:
        print("error is " + str(e))

    try:
        db = DBProvider.get_instance(db_name='hma_ca')

        s1 = db.insert(table='claims_data', rows=clean_ob_lst)
        s3 = db.insert(table='line_level_claims_data', rows=line_level_lst)

        update_ob = {"$set": {'audit_not_needed': clean_df[clean_df['CAF'] >= threshold / 100].shape[0],
                              'audit_needed': clean_df[clean_df['CAF'] < threshold / 100].shape[0],
                              "status": "in-progress"}}
        filter_ob = {"batch_name": batch_name}
        s2 = db.update(table='batch_metadata', update_obj=update_ob, filter_obj=filter_ob)
        return "Successfully Inserted into DB " + str(s1) + str(s2) + str(s3)
    except Exception as e:
        print(e)
        return 'e is ' + str(e)

