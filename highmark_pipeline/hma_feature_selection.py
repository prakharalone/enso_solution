from datetime import datetime
import pandas as pd
from xpms_file_storage.file_handler import XpmsResourceFactory, XpmsResource, LocalResource
import json
import numpy as np


def hma_feature_selection(config=None, **objects):
    df_lst = objects["data"]
    headers = df_lst.pop(0)
    df = pd.DataFrame(df_lst, columns=headers)
    df = df.drop(columns=[
        'billingprovider',
        'claimid',
        'createddate',
        'claimfinalizationdate',
        'claimpaiddate',
        'claimsubmissiondate',
        'adjudicationdate',
        'occurrencedate',
        'providerregid',
        'policynumberid',
        'name',
        'firstname',
        'middlename',
        'lastname',
        'dateofbirth',
        'personalid',
        'address1',
        'address2',
        'zip',
        'telephone',
        'country',
        'employername',
        'admissiondate',
        'groupnumbercode',
        'groupname',
        'dischargedate',
        'billstartdate',
        'billenddate',
        'subscriberid',
        'subscriberfirstname',
        'subscriberlastname',
        'medicarebeneficiaryid',
        'medicareid',
        'medicaidid',
        'coverageeffectivedate',
        'othercoverageterminationdate',
        'othercoverageeffectivedate',
        'othercoverageterminationdate',
        'othercoveragememberid',
        'providerid',
        'providername',
        'nationalproviderid',
        'taxid',
        'referringprovider',
        'referringprovidername',
        'provideraddress1',
        'provideraddress2',
        'providerzip',
        'providercountry',
        'providercity',
        'providerstate',
        'drugname',
        'drugcode',
        'providerlicensecode',
        'providersignaturecode',
        'regioncode',
        'facilityname',
        'providernetworkname',
        'providernetworkcode',
        'startdateofservice',
        'enddateofservice',
        'memberproductid',
        'pharmacyname',
        'pharmacyaddress1',
        'pharmacyaddress2',
        'pharmacycity',
        'pharmacystate',
        'pharmacyzip',
        'pharmacytype',
        'dayssupply',
        'gender_code_M',
        'relation_code_2',
        'provider_class_code_11'
        ])
    return {
        "dataset":{
            "data_format":"list",
            "value":[df.columns.values.tolist()] + df.values.tolist()
        }
    }
