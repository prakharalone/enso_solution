from datetime import datetime
import pandas as pd
from xpms_storage.db_handler import DBProvider
import json
import time
import requests
import numpy as np
from xpms_storage.utils import get_env

def hma_fetch_retrain_data(config=None, **objects):
    NAMESPACE = get_env("NAMESPACE", "claims-audit", False)
    DOMAIN_NAME = get_env("DOMAIN_NAME", "enterprise.xpms.ai", False)
    ENV_DATABASE = get_env('DATABASE_PARAPHRASE', None, True)
    try:

        db = DBProvider.get_instance(db_name=ENV_DATABASE)
        objects = db.find(table="line_level_claims_data", columns={
            "exclude": ["data.CAF", "data.CFE",
                        ]},
                          filter_obj={"flag": "untrained", "manual_audit": {"$exists": 1}},
                          limit=200000)

        data = [claim_data['data'] for claim_data in objects]
        if len(data) > 0:
            df = pd.DataFrame(data)

            #         for claim_data in objects:
            #             temp_df = pd.DataFrame(claim_data["data"])
            #             df = df.append(temp_df, ignore_index=True)

            manual_audit_result = [item['manual_audit']["manual_audit_result"] for item in objects]
            manual_audit_error_bucket = [item['manual_audit']["manual_audit_error_bucket"] for item in objects]
            manual_audit_claim_id = [item['data']["CLAIM_NUMBER_Mask"] for item in objects]

            manual_audit_df = pd.DataFrame(
                data=list(zip(manual_audit_claim_id, manual_audit_result, manual_audit_error_bucket)),
                columns=["CLAIM_NUMBER_Mask", "manual_audit_result", "manual_audit_error_bucket"])

            #             df = df.merge(manual_audit_df,on="CLAIM NUMBER")
            df = pd.concat([df.reset_index(drop=True),
                            manual_audit_df[["manual_audit_result", "manual_audit_error_bucket"]].reset_index(
                                drop=True)], axis=1)
            df["manual_audit_result"] = df["manual_audit_result"].apply(lambda x: "CAF" if x == "clean" else "CFE")
            notification = {
                "group": "retrain_status",
                "message": {
                    "batch_name": 'Retrain Pipeline',
                    "current_status": "started",
                    "previous_status": "Nil"
                },
                "created_timestamp": int(time.time())
            }
            db = DBProvider.get_instance(db_name=ENV_DATABASE)
            s = db.insert(table='notifications', rows=[notification])

            if s:
                url = "https://claimsaudit-be.{0}.{1}/send_notification".format(NAMESPACE,DOMAIN_NAME)
                headers = {
                    'Content-Type': 'application/json'
                }

                requests.request("POST", url, headers=headers, data=json.dumps(notification, default=str))

            # return generic_df
            return {
                "batch_length": len(df),
                "data": [df.columns.values.tolist()] + df.values.tolist()
            }

        else:
            return {
                "batch_length": 0,
                "data": []
            }

    except Exception as e:
        return {
            "batch_length": 0,
            "data": [],
            "error": str(e)
        }
