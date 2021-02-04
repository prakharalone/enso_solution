import pandas as pd
import time
from xpms_storage.db_handler import DBProvider
import json
import numpy as np
from datetime import datetime
from xpms_storage.utils import get_env

def hma_update_audit_claims_retrain_flag(config=None, **obj):
    claims_list = config['context']['audit_claim_ids']
    ENV_DATABASE = get_env('DATABASE_PARAPHRASE', None, True)

    if all([str(obj['model']['status']).lower() == 'trained',
            # obj['model']['is_published'],
            # obj['model']['is_default'],
            # obj['model']['is_enabled'],
            obj['success']]
           ):
        try:
            db = DBProvider.get_instance(db_name=ENV_DATABASE)
            update_ob = {"$set": {'flag': 'retrained'}}
            filter_ob_1 = {"data.CLAIM NUMBER": {'$in': claims_list}}
            filter_ob_2 = {"data.CLAIM_NUMBER_Mask": {'$in': claims_list}}
            s = db.update(table='claims_data', update_obj=update_ob, filter_obj=filter_ob_1)
            s = db.update(table='line_level_claims_data', update_obj=update_ob, filter_obj=filter_ob_2)
            return {"update_id_flag": s}
        except Exception as e:

            return str(e)

    else:

        return {"update_id_flag": False}
