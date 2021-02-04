from datetime import datetime
import json
from xpms_file_storage.file_handler import XpmsResourceFactory, XpmsResource, LocalResource
from xpms_storage.utils import get_env
from xpms_storage.db_handler import DBProvider
import json
import time
import requests

def hma_move_audit_train_to_completed(config=None, **objects):
    NAMESPACE = get_env("NAMESPACE", "claims-audit", False)
    DOMAIN_NAME = get_env("DOMAIN_NAME", "enterprise.xpms.ai", False)
    ENV_DATABASE = get_env('DATABASE_PARAPHRASE', None, True)

    file_path = config["context"]["source_file_path"]
    # file_path = "minio://claims-audit/aclaimsauditpoc/highmark/highmark_inputs_backup/1601454901_hma.csv"
    file_name = file_path.split("/")[-1]
    if objects["model"]["status"].lower() == "trained":
        xrm = XpmsResource()
        mr = xrm.get(urn=file_path)
        print(mr)
        backup_path = "minio://{0}/claimsaudit-ingestfiles/audit-train-batches-completed".format(NAMESPACE)
        backup_filename = str(int(datetime.now().timestamp())) + '_' + file_name
        backup_rm = XpmsResource()
        backup_mr = backup_rm.get(urn=backup_path + '/' + backup_filename)
        if mr.exists():
            mr.copy(backup_mr)
            mr.delete()
            notification = {
                "group": "batch_train_status",
                "message": {
                    "batch_name": 'Batch Train for audit model using {}'.format(file_name),
                    "current_status": "completed",
                    "previous_status": "started"
                },
                "created_timestamp": int(time.time())
            }
            db = DBProvider.get_instance(db_name=ENV_DATABASE)
            s = db.insert(table='notifications', rows=[notification])

            if s:
                url = "https://claimsaudit-be.{0}.{1}/send_notification".format(NAMESPACE, DOMAIN_NAME)
                headers = {
                    'Content-Type': 'application/json'
                }

                requests.request("POST", url, headers=headers, data=json.dumps(notification, default=str))
            return {
                "status": "moved"
            }
        else:
            return {
                "status": "file not exist"
            }
    else:
        return {
            "status": "not moved"
        }