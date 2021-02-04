from xpms_storage.db_handler import DBProvider
import json
import time
import requests
from xpms_storage.utils import get_env

def hma_retrain_completed_notification(**obj):
    NAMESPACE = get_env("NAMESPACE", "claims-audit", False)
    DOMAIN_NAME = get_env("DOMAIN_NAME", "enterprise.xpms.ai", False)
    ENV_DATABASE = get_env('DATABASE_PARAPHRASE', None, True)

    notification = {
        "group": "retrain_status",
        "message": {
            "batch_name": "Retrain Pipeline",
            "current_status": "completed",
            "previous_status": "started"
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

    return {"status":"completed"}