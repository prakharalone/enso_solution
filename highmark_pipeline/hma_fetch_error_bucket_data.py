from datetime import datetime
import pandas as pd


def hma_fetch_error_bucket_data(config=None, **objects):
    if len(objects["data"]) > 0:
        cols = objects["data"][0]
        vals = objects["data"][1:]
        df = pd.DataFrame(data=vals, columns=cols)
        # print(df.columns)
        if "Error Result" in df.columns:
            df = df.loc[:, df.columns != 'Audit Result']
            df = df.loc[(df["manual_audit_error_bucket"] != 'NA')]
            # df = df.loc[df["Error Result"] != df["manual_audit_error_bucket"]]
            config["context"]["error_claim_ids"] = df["CLAIM_NUMBER_Mask"].tolist()
            df.drop(["CLAIM_NUMBER_Mask", "Error Result", "manual_audit_result"], axis=1, inplace=True)
            df.rename(columns={"manual_audit_error_bucket": 'ec_dscr'}, inplace=True)
            df["ec_dscr"] = df["ec_dscr"].map({'Coding Error': 0, 'Frequency - Claim Error': 1, 'Frequency - Money Claim Error': 2, 'Internal Error': 3})
            return {
                "dataset": {
                    "data_format": "list",
                    "value": [df.columns.values.tolist()] + df.values.tolist(),
                    "size": int(df.shape[0])
                }
            }

        else:
            return {
                "dataset": {
                    "data_format": "list",
                    "value": [],
                    "size": int(0)
                }
            }
    else:
        return {
            "dataset": {
                "data_format": "list",
                "value": [],
                "size": int(0)
            }
        }


