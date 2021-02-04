from datetime import datetime
import pandas as pd


def hma_fetch_caf_cfe_data(config=None, **objects):
    if len(objects["data"]) > 0:
        cols = objects["data"][0]
        vals = objects["data"][1:]
        df = pd.DataFrame(data=vals, columns=cols)

        df = df.loc[:, df.columns != 'Error Result']
        # df = df.loc[df["Audit Result"] != df["manual_audit_result"]]

        config["context"]["audit_claim_ids"] = df["CLAIM_NUMBER_Mask"].tolist()

        df.drop(["CLAIM_NUMBER_Mask", "Audit Result", "manual_audit_error_bucket"], axis=1, inplace=True)
        df.rename(columns={"manual_audit_result": 'cas_stus_dscr'}, inplace=True)
        df["cas_stus_dscr"]= df["cas_stus_dscr"].map({"CAF":1, "CFE":0})

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
                "size": 0
            }
        }