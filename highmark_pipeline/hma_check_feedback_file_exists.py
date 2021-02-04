from xpms_file_storage.file_handler import XpmsResourceFactory, XpmsResource, LocalResource
import pandas as pd
from datetime import datetime
import os
from xpms_storage.utils import get_env

def hma_check_feedback_file_exists(config=None, **objects):
    NAMESPACE = get_env("NAMESPACE", "claims-audit", False)

    file_path = "minio://{0}/claimsaudit-ingestfiles/feedback-inputs".format(NAMESPACE)
    xr = XpmsResource()
    minio_resource = xr.get(urn=file_path)
    if minio_resource.exists():
        all_files_list = minio_resource.list()
        files_list = [(path.filename) for path in all_files_list if ".csv" in path.fullpath]
        if len(files_list) == 0:

            return {
                "file_path": "na"
            }
        # elif len(files_list) > 1:
        #     return {
        #         'message': 'More than one file present in ' + file_path
        #     }
        else:
            file_name = files_list[0]
            local_path = '/tmp/local_' + file_name
            lr = LocalResource(key=local_path)
            xrm = XpmsResource()
            mr = xrm.get(urn=file_path + '/' + file_name)
            mr.copy(lr)
            backup_path = "minio://{0}/claimsaudit-ingestfiles/feedback-inprogress".format(NAMESPACE)

            backup_filename = str(int(datetime.now().timestamp())) + '_' + file_name

            filename, file_extension = os.path.splitext(file_name)

            backup_rm = XpmsResource()
            backup_mr = backup_rm.get(urn=backup_path + '/' + backup_filename)
            mr.copy(backup_mr)
            mr.delete()
            return {

                "file_path": backup_path + '/' + backup_filename

            }

    else:
        return {

            "file_path": "na"
        }
