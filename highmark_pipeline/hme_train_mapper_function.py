from datetime import datetime
import pandas as pd
from xpms_file_storage.file_handler import XpmsResourceFactory, XpmsResource, LocalResource
import json


def hme_train_mapper_function(config=None, **objects):
    mapper = {
        "claimid": "CLAIM NUMBER",
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
        "adjustmentamount": None,  # No data
        "sumofcharges": None,
        "memberpaidamount": None,
        "rejectioncode": "REJECTION REASON CODE",
        "servicetax": None,
        "medicaidamount": None,
        "medicareamount": None,
        "deductibleaspersecondarpayer": None,
        "deductibleasperprimarypayer": None,
        "deductiblenotmetasperplan": None,
        "deductibleasperdeductibleplan": None,
        "patientdeductibleamount": None,
        "providerreductionamount": None,
        "provideramount": "TOT PROV CHARGE",
        "surchargeamount": None,
        "copaydiscountamount": None,
        "oplamount": None,
        "cobamounttype": None,
        "cobamount": None,
        "rejectionreason": None,
        "rejectionamount": None,
        "subscriberpaymentamount": None,
        "totalpaidamount": None,
        "noncoveredamount": None,
        "totalclaimamount": None,

        # "proceduremodifiercode5": "MODIFIER CODE(5)",
        # "proceduremodifiercode4": "MODIFIER CODE(4)",
        "proceduremodifiercode5": None,  # No data
        "proceduremodifiercode4": None,  # No data

        "proceduremodifiercode3": "MODIFIER CODE(3)",
        "proceduremodifiercode2": "MODIFIER CODE(2)",
        "proceduremodifiercode1": "MODIFIER CODE(1)",
        "diagnosiscode0": "DIAG_CODE_0",
        "diagnosiscode1": "DIAG_CODE_1",
        "diagnosiscode2": "DIAG_CODE_2",
        "diagnosiscode3": "DIAG_CODE_3",
        "duplicatelineflag": None,
        "serverityassessmentcodesac": None,
        "servicevisitslimit": None,
        "servicelinestatus": None,
        "medicallyunlikelyeditsmue": None,
        "prospectivepaymentsystemppstypecode": None,
        "inquirytime": None,
        "priorauthcode": None,
        "caselineidforsavings": None,
        "historyflag": "HISTORY SOURCE CODE",
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
        "billedamount": None,
        "distancebyambulance": None,
        "adjustmentcode16": None,
        "typeofservice": None,
        "revenuecode": None,
        "quantity": None,
        "patientreasonforvisitprvcode": None,
        "principlesurgicalprocedure": None,
        "principlediagnosis": None,
        "submittedcharges": None,
        "proceedurecode": None,
        "secondaydiagnosiscode19": None,
        "unit": None,
        "currentservicetype": None,
        "lineamount": "PROVIDER CHARGE",
        "erindicator": None,
        "enddateofservice": "END DT OF SERVICE",
        "startdateofservice": "SERVICE BEGIN DATE",
        "placeofservice": "PLACE OF SERVICE",
        "linenumber": "LINE NUMBER",

        "secondarypayer": None,
        "primarypayer": None,
        "institutionalpricing": None,
        "memberproductid": "PROD IDEN",

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
        "providerzip": "REFERRRING PROVIDER ZIP",
        "providerstate": "REFERRRING PROVIDER STATE",
        "providercity": None,
        "provideraddress2": None,
        "provideraddress1": None,
        "referringprovidername": None,
        "referringprovider": "REFERRING PROVIDER ID",
        "providerspeciality13": "PROVIDER SPECIALTY CODE",
        "providerclassification": "PROV CLS CD",
        "taxid": None,
        "nationalproviderid": "NPI NO",
        "providername": None,
        "providerid": None,
        "providertype": None,

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
        "dischargedate": None,
        "responsibilityflag": None,
        "groupname": None,
        "groupnumbercode": "VERIFIED GRP NO",
        "admissiondate": None,
        "admissiontype": None,
        "relationshipwithsubscriber": "RELATIONSHIP CODE",
        "employername": None,
        "country": None,
        "telephone": None,
        "zip": None,
        "state": None,
        "city": None,
        "address2": None,
        "address1": None,
        "personalid": None,
        "dateofbirth": "ABK_PNT_BTH_DATE",
        "gender": "ABK_PNT_SEX_CODE",
        "lastname": None,
        "middlename": None,
        "firstname": None,
        "name": None,
        "policynumberid": None,
        "providerregid": None,
        "type": None,

        "imageclaimflag": "IMAGE INDICATOR",
        "eobflag": "EOB REMIT",
        "claimstatuscode": "CLAIM STATUS CODE",
        "claimstatus": None,
        "bcbsflag": None,
        "fepsecondaryflag": None,
        "primaryblueflag": None,
        "autoclaimflag": None,
        "oplauto": None,
        "oplflag": None,
        "activecoverageflag": None,
        "lob": None,
        "occurrencespan": None,
        "occurrencedate": None,
        "occurrencecode": None,
        "additionalinfoflag": None,
        "billtype": None,
        "facilityflag": None,
        "adjudicationdate": None,
        # "adjudicationcode":"ADJ TYP CODE",
        "adjudicationcode": None,  # No data
        "documentcontrolnumber": None,
        "memberinfoconsent": None,
        "claimsubmissiondate": None,
        "frequencycode": "FREQ TYPE CODE",
        "placeoftreatment": None,
        # "claimamount":"CLAIM AMT",
        "claimamount": None,
        "specialprogramcode": None,
        "payersresponsibility": None,
        "assignmentsofbenefits": "ASSIGN OF BENEFITS",
        "medicaidflag": None,
        "medicareflag": None,
        "signatureonfileflag": None,
        "claimpaiddate": None,
        "attachmentindicator": None,
        "claimfinalizationdate": "FINAL DATE",
        "duplicateflag": None,
        "typeofadmission": None,
        "createddate": None,
        "sourceindicator": "compute_0014",  # source_indicator
        "claimtype": "CLAIM TYPE",
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

    if "audit result" in hma_df:
        generic_df["audit_result"] = hma_df["audit result"]

        return {
            "batch_length": len(generic_df),
            "data": [generic_df.columns.values.tolist()] + generic_df.values.tolist()
        }

    elif "error bucket" in hma_df:
        generic_df["error_bucket"] = hma_df["error bucket"]
        return {
            "batch_length": len(generic_df),
            "data": [generic_df.columns.values.tolist()] + generic_df.values.tolist()
        }