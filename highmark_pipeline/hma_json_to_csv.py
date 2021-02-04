import time
import uuid
import json
import numpy as np
import pandas as pd
from xpms_storage.utils import get_env
from xpms_file_storage.file_handler import XpmsResourceFactory, XpmsResource, LocalResource


def hma_json_to_csv(config=None, **objects):
    dataset = None
    if objects.get('dataset'):  # My payload has data inside dataset key
        if objects['dataset']['data_format'] == 'json':
            dataset = pd.DataFrame(
                objects['dataset']['value'][1:],
                columns=objects['dataset']['value'][0])
    # creating file and saving in the minio
    dataset = dataset.replace(r'^\s*$', np.nan, regex=True)
    NAMESPACE = get_env("NAMESPACE", "claims-audit", False)
    start_time = int(time.time())
    file_name = "{0}_{1}_json_to_csv.csv".format(str(uuid.uuid4())[:8], start_time)
    csv_minio_urn = "minio://{0}/".format(NAMESPACE) +"/claimsaudit-ingestfiles/batches-input/"+ file_name
    local_csv_path = "/tmp/" + file_name
    minio_resource = XpmsResource.get(urn=csv_minio_urn)
    # saving the csv
    dataset.to_csv(local_csv_path,na_rep='NaN', index=False, header=True)
    local_res = LocalResource(key=local_csv_path)
    local_res.copy(minio_resource)
    return {"csv_urn": csv_minio_urn}
