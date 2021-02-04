from datetime import datetime
import numpy as np
from xpms_storage.db_handler import DBProvider
import time
import requests
import json
from xpms_storage.utils import get_env


def hma_feedback_notification_inprogress(config=None, **objects):

    NAMESPACE = get_env("NAMESPACE", "claims-audit", False)
    DOMAIN_NAME = get_env("DOMAIN_NAME", "enterprise.xpms.ai", False)
    ENV_DATABASE = get_env('DATABASE_PARAPHRASE', None, True)

    file_name = objects["document"][0]["metadata"]["properties"]["filename"]
    db = DBProvider.get_instance(db_name=ENV_DATABASE)
    notifications = {
        "group": "feedback_status",
        "message": {
            "file_name": file_name,
            "current_status": "in-progress",
            "previous_status": "started"
        },
        "created_timestamp": int(time.time())
    }

    s = db.insert(table='notifications', rows=[notifications])
    if s:
        url = "https://claimsaudit-be.{0}.{1}/send_notification".format(NAMESPACE,DOMAIN_NAME)
        headers = {
            'Content-Type': 'application/json'
        }

        resp = requests.request("POST", url, headers=headers, data=json.dumps(notifications, default=str))

    return objects
