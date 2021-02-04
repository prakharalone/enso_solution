from xpms_storage.db_handler import DBProvider
import pandas as pd
import numpy as np
from xpms_file_storage.file_handler import XpmsResource, LocalResource


def create_batch_feedbacks(config=None, **objects):
    db = DBProvider.get_instance(db_name='hma_ca')
    batches = ["batch_84bc0ec2_1602829805", "batch_16530822_1602828307", "batch_b9fd1e7e_1602828007",
               "batch_eb2534b5_1602827405", "batch_8a61e678_1602827708", "batch_46125d63_1602828904",
               "batch_b5d8efb3_1602828605", "batch_8469135d_1602826808", "batch_37859980_1602826505"]

    for batch in batches:
        objects = db.find(table="claims_data", filter_obj={"batch_name": batch, })

        claims_ids = [item["data"]["CLAIM NUMBER"] for item in objects]

        df = pd.DataFrame(data=list(zip(claims_ids)), columns=["claim_id"])
        df["manual_audit_result"] = np.random.choice(["clean", "error"],
                                                     len(claims_ids))
        df["manual_audit_error_bucket"] = np.random.choice(["subscriber", "service", "cob", "benefit", "provider"],
                                                           len(claims_ids))
        df["manual_audit_error_bucket"] = np.where(df['manual_audit_result'] == 'clean', "",
                                                   df['manual_audit_error_bucket'])
        df.to_csv("/tmp/feedback_" + batch + ".csv", index=False)

        csv_minio_urn = "minio://claims-audit/aclaimsauditpoc/batch_feedbacks/" + "feedback_" + batch + ".csv"
        local_csv_path = "/tmp/feedback_" + batch + ".csv"

        minio_resource = XpmsResource.get(urn=csv_minio_urn)
        local_res = LocalResource(key=local_csv_path)
        local_res.copy(minio_resource)

    return {
        "status": "done"
    }

