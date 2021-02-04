from datetime import datetime
import pandas as pd
from xpms_file_storage.file_handler import XpmsResourceFactory, XpmsResource, LocalResource
import json
from sklearn.preprocessing import StandardScaler
import numpy as np
import json


def hma_mapper_preprocess_feature(config=None, **objects):
    mapper = {
        "claimid": "CLAIM_NUMBER_Mask",
        "familytmoop": None,
        "totalmaximumoutofpockettmoop": None,
        "familycoinsuranceamount": None,
        "individualcoinsuranceamount": None,
        "familyoop": None,
        "individualoop": None,
        "familydeductibleamount": None,
        "allowanceamount": None,
        "oplworkercompensation": None,
        "paypercent": None,
        "costshare": None,
        # "adjustmentamount": "ADJ NET AMT",
        "adjustmentamount": "ABL_NET_AMT",  # No data
        "sumofcharges": None,
        "memberpaidamount": "AB7_FEE_PAID_AMT",
        "rejectioncode": "AC1_RSN_CODE",
        "servicetax": None,
        "medicaidamount": None,
        "medicareamount": None,
        "deductibleaspersecondarpayer": None,
        "deductibleasperprimarypayer": None,
        "deductiblenotmetasperplan": None,
        "deductibleasperdeductibleplan": None,
        "patientdeductibleamount": None,
        "providerreductionamount": None,
        "provideramount": "AFV_PRV_CRG_AMT",
        "surchargeamount": None,
        "copaydiscountamount": None,
        "oplamount": None,
        "cobamounttype": None,
        "cobamount": None,
        "rejectionreason": None,
        "rejectionamount": None,
        "subscriberpaymentamount": "AB7_FEE_PAID_AMT",
        "totalpaidamount": "ABL_NET_AMT",
        "noncoveredamount": None,
        "totalclaimamount": "AB7_TOT_BILL_AMT",

        # "proceduremodifiercode5": "MODIFIER CODE(5)",
        # "proceduremodifiercode4": "MODIFIER CODE(4)",
        "proceduremodifiercode7": "AGD_CODE_7",  # No data
        "proceduremodifiercode6": "AGD_CODE_6",  # No data
        "proceduremodifiercode5": "AGD_CODE_5",  # No data
        "proceduremodifiercode4": "AGD_CODE_4",  # No data
        "proceduremodifiercode3": "AGD_CODE_3",
        "proceduremodifiercode2": "AGD_CODE_2",
        "proceduremodifiercode1": "AGD_CODE_1",
        "diagnosiscode": "ADB_CODE",
        "duplicatelineflag": "ABR_RSN_CODE",
        "serverityassessmentcodesac": None,
        "servicevisitslimit": None,
        "servicelinestatus2": "ABK_STA_CODE_2",
        "servicelinestatus1": "ABK_STA_CODE_1",
        "medicallyunlikelyeditsmue": None,
        "prospectivepaymentsystemppstypecode": None,
        "inquirytime": None,
        "priorauthcode": None,
        "caselineidforsavings": None,
        "historyflag": "ABK_HIS_SCE_CODE",
        "encounterflag": None,
        # "xraytype": "X-RAY TYPE",
        "xraytype": None,  # No data
        "reversedflag": None,
        "authorizationcode": None,
        "copayamount": None,
        "coinsuranceamount": None,
        "deductibleamount": None,
        "notcoveredamount": None,
        "adjudicatoryamount": None,
        "usualandcustomaryamountucramount": None,
        "allowedamount": None,
        "billedamount": "AB7_TOT_BILL_AMT",
        "distancebyambulance": None,
        "adjustmentcode16": None,
        "typeofservice": None,
        "revenuecode": "AGD_UVFY_CODE",
        "quantity": None,
        "patientreasonforvisitprvcode": None,
        "principlesurgicalprocedure": "AGD_UVFY_CODE",
        "principlediagnosis": "ADB_CODE",
        "submittedcharges": "AB7_TOT_BILL_AMT",
        "proceedurecode": "AGD_UVFY_CODE",
        "secondaydiagnosiscode19": "ADB_CODE",
        "unit": None,
        "currentservicetype": None,
        "lineamount": "AFV_PRV_CRG_AMT",
        "erindicator": None,
        "enddateofservice": "HSCBMP_SVCE_END_DT",
        "startdateofservice": "HSCBMP_SVCE_BGN_DT",
        "placeofservice": "AFV_BCBSA_PL_CODE",
        "linenumber": "AFV_ID",

        "secondarypayer": None,
        "primarypayer": None,
        "institutionalpricing": None,
        "memberproductid": "PR_ID",

        "dayssupply": None,
        "pharmacytype": None,
        "pharmacyzip": None,
        "pharmacystate": None,
        "pharmacycity": None,
        "pharmacyaddress2": None,
        "pharmacyaddress1": None,
        "pharmacyname": None,

        "orderingprovider": None,
        "renderingprovider": None,
        "servicingprovider": None,
        "billingprovider": "BILLING PROVIDER NUMBER",
        "performingprovider": None,
        "multispecialityflag": None,
        "onninnflag": None,
        "parnonparflag": None,
        "ppoflag": None,
        "providernetworkcode": None,
        "providernetworkname": None,
        "participationnetworkflag": None,
        "primarycarephysicianpcpflag": None,
        "facilityname": None,
        "regioncode": None,
        "providersignaturecode": None,
        "providerlicensecode": None,
        "drugcode": None,
        "drugname": None,
        # "providercountry":"REFERRING PROVIDER COUNTY",
        "providercountry": None,  # no data
        "providerzip": "SBMD_HSCRMPRV_ZIP_AD",
        "providerstate": "SBMD_HSCRMPRV_STE_AD",
        "providercity": "SBMD_HSCRMPRV_CTY_AD",
        "provideraddress2": None,
        "provideraddress1": None,
        "referringprovidername": None,
        "referringprovider": "ADO_REF_BY_PRV_ID",
        "providerspeciality13": "ANE_PFN_PRV_SPL_CODE",
        "providerclassification": "AB7_CLS_CODE",
        "taxid": None,
        "nationalproviderid": "SBIL_NPI_MVEI_ID",
        "providername": None,
        "providerid": "AFV_BILL_PRV_UVFY_ID",
        "providertype": "AB7_TYPE_CODE",

        "dualcredentialisedflag": None,
        "credentialisedflag": None,
        "cobprimarypayercopay": None,
        "cobprimarypayeramountpaid": None,
        "membercobtype": None,
        "othercoveragememberid": None,
        "othercoveragegroup": None,
        "othercoverageterminationdate": None,
        "othercoverageeffectivedate": None,
        "othercoveragetype": None,
        "othercoverageplan": None,
        "coverageterminationdate": None,
        "coverageeffectivedate": None,
        "accountstatus": None,
        "medicaidid": None,
        "medicareid": None,
        "medicarebeneficiaryid": None,
        "subscriberlastname": None,
        "subscriberfirstname": None,
        "subscriberid": None,
        "employeeflag": None,
        "dispositionto": None,
        "dischargestatus": None,
        "billenddate": None,
        "billstartdate": None,
        "dischargedate": "HSCBMP_SVCE_END_DT",
        "responsibilityflag": None,
        "groupname": None,
        "groupnumbercode": "ABK_VFY_ENR_GRP_ID",
        "admissiondate": "AFV_BGN_02_DATE",
        "admissiontype": None,
        "relationshipwithsubscriber": "ABK_PNT_REL_APP_CODE",
        "employername": None,
        "country": None,
        "telephone": None,
        "zip": None,
        "state": None,
        "city": None,
        "address2": "ABK_APP_FGN_LN_2_ADDR",
        "address1": "ABK_APP_FGN_LN_1_ADDR",
        "personalid": "ABK_PNT_PERS_ID",
        "dateofbirth": "ABK_PNT_BTH_DATE",
        "gender": "ABK_PNT_SEX_CODE",
        "lastname": "ABK_APP_SUB_LAST_NAME",
        "middlename": "ABK_APP_SUB_MID_NAME",
        "firstname": "ABK_APP_SUB_FST_NAME",
        "name": None,
        "policynumberid": "AB7_PNT_ACC_ID",
        "providerregid": 'AFV_BILL_CLM_PRV_ID',
        "type": None,

        "imageclaimflag": "CCJ_IMG_APL_ID",
        "eobflag": "AB7_EOB_CODE",
        "claimstatuscode2": "ABK_STA_CODE_2",
        "claimstatuscode1": "ABK_STA_CODE_1",
        "claimstatus2": "AFV_FNL_STA_CODE_2",
        "claimstatus1": "AFV_FNL_STA_CODE_1",
        "bcbsflag": None,
        "fepsecondaryflag": None,
        "primaryblueflag": None,
        "autoclaimflag": None,
        "oplauto": None,
        "oplflag": None,
        "activecoverageflag": "ABK_SUB_ENR_CLS_CODE",
        "lob": None,
        "occurrencespan": None,
        "occurrencedate": None,
        "occurrencecode": None,
        "additionalinfoflag": None,
        "billtype": None,
        "facilityflag": None,
        "adjudicationdate": "ABK_FNL_DATE",
        # "adjudicationcode":"ADJ TYP CODE",
        "adjudicationcode5": 'PBSC_SAE_CD_5',
        "adjudicationcode4": 'PBSC_SAE_CD_4',
        "adjudicationcode3": 'PBSC_SAE_CD_3',
        "adjudicationcode2": 'PBSC_SAE_CD_2',
        "adjudicationcode1": 'PBSC_SAE_CD_1',  # No data
        "documentcontrolnumber": None,
        "memberinfoconsent": None,
        "claimsubmissiondate": None,
        "frequencycode": "NS_FTP_CD",
        "placeoftreatment": None,
        # "claimamount":"CLAIM AMT",
        "claimamount": 'ABE_RCP_CLM_AMT',
        "specialprogramcode": None,
        "payersresponsibility": None,
        "assignmentsofbenefits": "AB7_SUB_ASG_BEN_CODE",
        "medicaidflag": None,
        "medicareflag": None,
        "signatureonfileflag": None,
        "claimpaiddate": None,
        "attachmentindicator": None,
        "claimfinalizationdate": "ABK_FNL_DATE",
        "duplicateflag": None,
        "typeofadmission": None,
        "createddate": None,
        "sourceindicator": "PBS_CLM_ORIG_CD",  # source_indicator
        "claimtype": "ABK_TYPE_CODE",
    }
    file_path = objects["document"][0]["metadata"]["properties"]["file_metadata"]["file_path"]
    local_csv_path = "/tmp/ " + objects["document"][0]["metadata"]["properties"]["filename"]

    config["context"]["source_file_path"] = file_path
    config["context"]["local_source_file_path"] = local_csv_path
    config["context"]["doc_id"] = objects["document"][0]["doc_id"]
    config["context"]["solution_id"] = objects["document"][0]["solution_id"]
    config["context"]["root_id"] = objects["document"][0]["root_id"]

    mapped_file_path = objects["document"][0]["metadata"]["properties"]["file_metadata"]["file_path"].replace(
        objects["document"][0]["metadata"]["properties"]["filename"],
        "mapped_" + objects["document"][0]["metadata"]["properties"]["filename"])

    mapped_local_csv_path = "/tmp/" + "mapped_" + objects["document"][0]["metadata"]["properties"]["filename"]
    minio_resource = XpmsResource.get(urn=file_path)
    local_res = LocalResource(key=local_csv_path)
    minio_resource.copy(local_res)

    minio_resource2 = XpmsResource.get(urn=mapped_file_path)
    mapped_local_res = LocalResource(key=mapped_local_csv_path)

    hma_df = pd.read_csv(local_csv_path)
    hma_df.columns = [x.lower().strip() for x in hma_df.columns]

    generic_df = pd.DataFrame(columns=(mapper.keys()))

    for k, v in (mapper.items()):
        if v is not None:
            generic_df[k] = hma_df[v.lower()]
        else:
            generic_df[k] = 0

    df = generic_df

    # -------------------------------------------------------------------------
    # Feature selection
    # -------------------------------------------------------------------------
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

    code_df.eobflag = code_df.eobflag.map(dict(Y=1, N=0))

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

