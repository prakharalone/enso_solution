from xpms_file_storage.file_handler import XpmsResourceFactory, XpmsResource, LocalResource
import pandas as pd
import time
from xpms_storage.db_handler import DBProvider
import uuid
from datetime import datetime
import json
import requests
from xpms_storage.utils import get_env



def hma_metadata(config=None, **objects):
    NAMESPACE = get_env("NAMESPACE", "claims-audit", False)
    DOMAIN_NAME = get_env("DOMAIN_NAME", "enterprise.xpms.ai", False)
    ENV_DATABASE = get_env('DATABASE_PARAPHRASE', None, True)

    file_path = objects["document"][0]["metadata"]["properties"]["file_metadata"]["file_path"]
    local_csv_path = "/tmp/" + objects["document"][0]["metadata"]["properties"]["filename"]
    minio_resource = XpmsResource.get(urn=file_path)
    local_res = LocalResource(key=local_csv_path)
    minio_resource.copy(local_res)

    dataset = pd.read_csv(local_csv_path)

    start_time = int(time.time())
    batch_name = "batch_{0}_{1}".format(str(uuid.uuid4())[:8], start_time)
    converted_start_time = datetime.utcfromtimestamp(start_time)
    input_source = objects["document"][0]["metadata"]["properties"]["extension"]
    status = "to-do"
    batch_volume = dataset.groupby("CLAIM_NUMBER_Mask").grouper.shape[0]

    db = DBProvider.get_instance(db_name=ENV_DATABASE)
    try:
        data = db.find(table='global_settings')
        threshold = data[0]['confidence_score']
        config['context']['threshold'] = threshold

    except Exception as e:
        threshold = 50
        config['context']['threshold'] = threshold

    config["context"]["batch_name"] = batch_name
    config["context"]["start_time"] = start_time
    config["context"]["input_file_name"] = objects["document"][0]["metadata"]["properties"]["filename"]

    batch_ob = {
        "batch_name": batch_name,
        "input_source": input_source,
        "batch_volume": batch_volume,
        "audit_needed": None,
        "audit_not_needed": None,
        "batch_start_date": start_time,
        "converted_start_time": converted_start_time,
        "status": status,
        "threshold": threshold,
        "file_name": objects["document"][0]["metadata"]["properties"]["filename"]
    }

    try:
        db = DBProvider.get_instance(db_name=ENV_DATABASE)
        s = db.insert(table='batch_metadata', rows=batch_ob)
    except Exception as e:
        return 'e is ' + str(e)

    try:
        notification = {
            "group": "batch_status",
            "message": {
                "batch_name": batch_name,
                "current_status": "to-do",
                "previous_status": "started"
            },
            "created_timestamp": start_time
        }
        db = DBProvider.get_instance(db_name=ENV_DATABASE)
        s = db.insert(table='notifications', rows=[notification])

        if s:
            url = "https://claimsaudit-be.{0}.{1}/send_notification".format(NAMESPACE,DOMAIN_NAME)
            headers = {
                'Content-Type': 'application/json'
            }

            resp = requests.request("POST", url, headers=headers, data=json.dumps(notification, default=str))


    except Exception as e:
        return "error is " + str(e)

    return objects
