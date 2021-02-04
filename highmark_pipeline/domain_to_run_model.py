from datetime import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
import pandas as pd
from xpms_file_storage.file_handler import XpmsResourceFactory, XpmsResource, LocalResource


def domain_to_run_model(config=None, **objects):
    source_dataset = {
        "value": objects["domain"],
        "dataset_format": "csv"
    }
    data = domain_object_to_df(source_dataset, **objects)

    return data


def domain_object_to_df(source_dataset, **objects):
    columns = []
    rows = []
    for domain in source_dataset['value']:
        row_value = {}
        if 'children' in domain and domain['children']:
            for d_object in domain['children']:
                for entity in d_object['children']:
                    attribute_path_list = [d_object['name'], entity['name']]
                    construct_key_path(entity, attribute_path_list, columns, row_value)
        rows.append(row_value)

    # df = pd.DataFrame(data=rows, columns=columns)

    df = pd.DataFrame.from_dict(row_value)
    rename_keys = {}
    for key in df.columns.tolist():
        if key not in rename_keys:
            rename_keys[key] = key.split('.')[-1]

    df.rename(columns=rename_keys, inplace=True)

    diag_code_list = [
        'Z23', 'E119', 'E785', 'I10', 'Z85828', 'M810', 'C61', 'C7951', 'J849',
        'H2511', 'K5000', 'E782', 'M170', 'M8580', 'R55', 'R4182', 'G309', 'C50919',
        'D801', 'J45909', 'C44319', 'C155', 'I872'
    ]
    rejection_reason_list = ['L5049', 'T6032', 'R5295', 'X5018', 'L5096']
    modifer_code_list = [
        '0', '26', 'WS', 'WJ', 'TC', 25, 'RR', 'AI', 'LT', 76, 'QK', 'GA', 'QS']
    history_source_code_list = ['O']
    procedure_code_list = [
        '731', '99213', '99214', '276', '96365', '360', '36415', 'G0008',
        '99223', '370', '17311', 'J1568', '17312', '71260', '93015', '66821',
        '710', 'J3111', '270', '90662', 'J2785', 'J3357', '74177', '96372',
        '29580', 'A9555', 'E1390', '85025', '78492', '636', '250']
    place_of_service_list = [10, 22, 30, 40]
    provider_speciality_list = [
        '008', '022', '014', '024', '020', '087', '000', '029', '021']
    provider_cls_list = ['01', '11']
    relationship_code_list = [1, 2]
    gender_list = ['M', 'F']
    claim_status_list = ['A']
    freq_type_list = [1]
    assignment_of_benefits_list = ['Y']
    data_source_list = ['EDW']

    feature_scaling_cols = ['provideramount', 'lineamount', 'days_of_service', 'age']

    df_diag = pd.DataFrame(columns=['diagnosis_code_' + str(dc) for dc in sorted(diag_code_list)])
    for code in diag_code_list:
        mask = df[["diagnosiscode0", "diagnosiscode1", "diagnosiscode2", "diagnosiscode3"]].isin([code]).any(axis=1)
        df_diag['diagnosis_code_' + str(code)] = [1 if code else 0 for code in mask.values]

    df_rejec = pd.DataFrame(columns=['rejection_code_' + str(dc) for dc in sorted(rejection_reason_list)])
    for code in rejection_reason_list:
        mask = df[["rejectioncode"]].isin([code]).any(axis=1)
        df_rejec['rejection_code_' + str(code)] = [1 if code else 0 for code in mask.values]

    df_mcode = pd.DataFrame(columns=['modifier_code_' + str(dc) for dc in (modifer_code_list)])
    for code in modifer_code_list:
        mask = df[['proceduremodifiercode1', 'proceduremodifiercode2', 'proceduremodifiercode3']].isin([code]).any(
            axis=1)
        df_mcode['modifier_code_' + str(code)] = [1 if code else 0 for code in mask.values]

    df_hiscode = pd.DataFrame(columns=['history_source_code_' + str(dc) for dc in sorted(history_source_code_list)])
    for code in history_source_code_list:
        mask = df[["historyflag"]].isin([code]).any(axis=1)
        df_hiscode['history_source_code_' + str(code)] = [1 if code else 0 for code in mask.values]

    df_pro_code = pd.DataFrame(columns=['procedure_code_' + str(dc) for dc in sorted(procedure_code_list)])
    for code in procedure_code_list:
        mask = df[["proceedurecode"]].isin([code]).any(axis=1)
        df_pro_code['procedure_code_' + str(code)] = [1 if code else 0 for code in mask.values]

    df_pos_code = pd.DataFrame(columns=['pos_code_' + str(dc) for dc in sorted(place_of_service_list)])
    for code in place_of_service_list:
        mask = df[["placeofservice"]].isin([code]).any(axis=1)
        df_pos_code['pos_code_' + str(code)] = [1 if code else 0 for code in mask.values]

    df_pspl_code = pd.DataFrame(
        columns=['provider_speciality_code_' + str(dc) for dc in sorted(provider_speciality_list)])
    for code in provider_speciality_list:
        mask = df[["providerspeciality13"]].isin([code]).any(axis=1)
        df_pspl_code['provider_speciality_code_' + str(code)] = [1 if code else 0 for code in mask.values]

    df_pclass_code = pd.DataFrame(columns=['provider_class_code_' + str(dc) for dc in sorted(provider_cls_list)])
    for code in provider_cls_list:
        mask = df[["providerclassification"]].isin([code]).any(axis=1)
        df_pclass_code['provider_class_code_' + str(code)] = [1 if code else 0 for code in mask.values]

    df_rel_code = pd.DataFrame(columns=['relation_code_' + str(dc) for dc in sorted(relationship_code_list)])
    for code in relationship_code_list:
        mask = df[["relationshipwithsubscriber"]].isin([code]).any(axis=1)
        df_rel_code['relation_code_' + str(code)] = [1 if code else 0 for code in mask.values]

    df_gender_code = pd.DataFrame(columns=['gender_code_' + str(dc) for dc in sorted(gender_list)])
    for code in gender_list:
        mask = df[["gender"]].isin([code]).any(axis=1)
        df_gender_code['gender_code_' + str(code)] = [1 if code else 0 for code in mask.values]

    df_claim_status_code = pd.DataFrame(columns=['claim_status_code_' + str(dc) for dc in sorted(claim_status_list)])
    for code in claim_status_list:
        mask = df[["claimstatuscode"]].isin([code]).any(axis=1)
        df_claim_status_code['claim_status_code_' + str(code)] = [1 if code else 0 for code in mask.values]

    df_freq_code = pd.DataFrame(columns=['freq_type_code_' + str(dc) for dc in sorted(freq_type_list)])
    for code in freq_type_list:
        mask = df[["frequencycode"]].isin([code]).any(axis=1)
        df_freq_code['freq_type_code_' + str(code)] = [1 if code else 0 for code in mask.values]

    df_aob_code = pd.DataFrame(columns=['aob_code_' + str(dc) for dc in sorted(assignment_of_benefits_list)])
    for code in assignment_of_benefits_list:
        mask = df[["assignmentsofbenefits"]].isin([code]).any(axis=1)
        df_aob_code['aob_code_' + str(code)] = [1 if code else 0 for code in mask.values]

    df_datas_code = pd.DataFrame(columns=['data_source_code_' + str(dc) for dc in sorted(data_source_list)])
    for code in data_source_list:
        mask = df[["sourceindicator"]].isin([code]).any(axis=1)
        df_datas_code['data_source_code_' + str(code)] = [1 if code else 0 for code in mask.values]

    df.drop(columns=[
        "diagnosiscode0", "diagnosiscode1", "diagnosiscode2", "diagnosiscode3",
        "rejectioncode", 'proceduremodifiercode1', 'proceduremodifiercode2',
        'proceduremodifiercode3', "historyflag", "proceedurecode",
        "placeofservice", "providerspeciality13", "providerclassification",
        "relationshipwithsubscriber", "gender", "claimstatuscode",
        "frequencycode", "assignmentsofbenefits", "sourceindicator"
    ], axis=1, inplace=True)
    code_df = pd.concat([
        df, df_diag, df_rejec, df_mcode, df_hiscode, df_pro_code, df_pos_code,
        df_pspl_code, df_pclass_code, df_rel_code, df_gender_code,
        df_claim_status_code, df_freq_code, df_aob_code, df_datas_code
    ], axis=1)

    # start_date_of_service, end_date_of_service, date_of_birth, final_date
    code_df['days_of_service'] = (
            pd.to_datetime(code_df['enddateofservice'])
            - pd.to_datetime(code_df['startdateofservice']))
    code_df['age'] = pd.Timestamp('now') - pd.to_datetime(code_df['dateofbirth'])
    code_df['age'] = code_df['age'] / np.timedelta64(1, 'D')
    code_df['days_of_service'] = code_df['days_of_service'] / np.timedelta64(1, 'D')

    sc = StandardScaler()
    code_df[feature_scaling_cols] = sc.fit_transform(code_df[feature_scaling_cols])

    code_df.eobflag = code_df.eobflag.map({"y": 1, "n": 0})

    code_df = code_df.drop(columns=[
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

    file_name = objects["document"][0]["metadata"]["properties"]["filename"]
    csv_minio_urn = "minio://claims-audit/aclaimsauditpoc/ml_input/mapped_" + file_name
    local_csv_path = "/tmp/mapped_" + file_name
    minio_resource = XpmsResource.get(urn=csv_minio_urn)
    code_df.to_csv(local_csv_path, index=False)
    local_res = LocalResource(key=local_csv_path)
    local_res.copy(minio_resource)

    return {
        "dataset": {
            "data_format": "csv",
            "value": csv_minio_urn
        }
    }


def construct_key_path(data, attribute_path_list, columns, values):
    for attribute in data['children']:
        sub_items = deepcopy(attribute_path_list)
        sub_items.append(attribute['name'])
        if 'children' in attribute and attribute['children']:
            construct_key_path(attribute, sub_items, columns, values)
        else:
            key_path = ".".join(sub_items)
            if key_path not in columns:
                columns.append(key_path)
            if key_path not in values:
                values[key_path] = [attribute['value']]
            else:
                values[key_path].append(attribute['value'])



