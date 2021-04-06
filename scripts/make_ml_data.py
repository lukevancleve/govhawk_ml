import os
import pickle

import pandas as pd
from sklearn.preprocessing import LabelEncoder

METADATA_DIR = "references/external/"
DERIVED_DIR = 'references/derived/'

bv = pd.read_csv(METADATA_DIR + "bill_version.csv", sep=";", encoding="latin1", parse_dates=True)
sessions =  pd.read_csv(METADATA_DIR + "session.csv", sep=";", encoding="latin1", parse_dates=True)
bills =  pd.read_csv(METADATA_DIR + "bill.csv", sep=";", encoding="latin1", parse_dates=True)
divs =  pd.read_csv(METADATA_DIR + "division.csv", sep=";", encoding="latin1", parse_dates=True)
bill_leg_vote =  pd.read_csv(METADATA_DIR + "bill_legislator_vote.csv", sep=";", encoding="latin1", parse_dates=True)
vote_rec_type =  pd.read_csv(METADATA_DIR + "vote_recorded_type.csv", sep=";", encoding="latin1", parse_dates=True)

pl = pd.read_csv(DERIVED_DIR + "/partisan_lean.csv", sep=",", encoding="latin1", parse_dates=True)

if 'DATA_VOL' not in os.environ:
    # Manually set:
    raise("DATA_VOL should be set in the Docker image.")
    #DATA_VOL = '/datavol/'
else:
    DATA_VOL = os.environ['DATA_VOL']

print(bv.head())
version_exists = []
for i in bv['id']:

    fn = DATA_VOL + 'clean/' + str(i) + '.txt'
    #print(fn)
    if os.path.exists(fn):
        version_exists.append(True)
    else:
        version_exists.append(False)

print(f'Number of total possible bill_versions:{ len(bv) }')
print(f"Number of bill versions in DATA_VOL/clean/ dir:{sum(version_exists)}")
print(f"Percentage:{sum(version_exists)/len(bv)*100:.2f}%")

# Use only the ones properly cleaned:
bv = bv[version_exists]


# Data Cleaning:
# One bill version has no bill_id
bills = bills[~bills.chamber_id.isna()]
bills.chamber_id = bills.chamber_id.astype(int)


for col in bills[['signed', 'passed_lower', 'passed_upper' ]]:
    mask = bills[col].isna()
    bills.loc[mask, col] = 0
    mask = bills[col]!=0
    bills.loc[mask, col] = 1


# Number of unique bills per state. How many passed each chamber.
df = bills.merge(divs, left_on="division_id", right_on="id")
df[['abbr', "signed", "passed_upper", "passed_lower", "has_data"]].groupby("abbr").agg(["count"]).sort_values('abbr').head()

df = df[['abbr', 'passed_lower', 'passed_upper', 'signed']]
df = df.fillna(0)
for col in df.columns[1:]:
    mask = df[col]!=0
    df.loc[mask, col] = 1
df = df.astype({"abbr":str, "signed": int, "passed_upper": int, "passed_lower":int})
passage_agg = df.groupby('abbr').agg(["mean", "count"])
sm = passage_agg['signed']['mean'].sort_values(axis=0)


a = bv
a['version_number'] = a[['id', 'bill_id', 'updated_at']].groupby('bill_id').cumcount()+1
a[['id', 'bill_id', 'version_number', 'updated_at']].sort_values('bill_id').tail(n=100)


ml_data = a[['id', 'version_number', 'bill_id']].merge(bills[['id', 'signed', 'passed_upper', 'passed_lower', 'session_id', 'chamber_id']].rename(columns={'id':"bill_id"}), on='bill_id')
ml_data = ml_data.merge(pl[['session_id', 'chamber_id', 'partisan_lean']], on=['session_id', 'chamber_id'])
ml_data = ml_data.fillna(0)
ml_data['sc_id'] = ml_data['session_id'].astype(str) + "-" + ml_data['chamber_id'].astype(str)
ml_data['passed'] = ml_data['passed_lower']*(ml_data['chamber_id']==1) + ml_data['passed_upper']*(ml_data['chamber_id']==2)
ml_data = ml_data[['id', 'version_number', 'bill_id', 'signed', 'passed', 'partisan_lean',  'sc_id']]
s1 = ml_data[['bill_id', 'id']].groupby('bill_id').sample(1)
ml_data = ml_data.merge(s1, on = ['id', 'bill_id'])



# Encode all the labels before (potentially) reducing the dataset.
sc_id_encoder = LabelEncoder()
ml_data['sc_id_cat'] = sc_id_encoder.fit_transform(ml_data['sc_id'])  
pickle.dump( sc_id_encoder, open( "models/encoder_production.pkl", "wb" ) )


ml_data.to_csv(DERIVED_DIR + 'ml_data.csv', index=False)

print(f"ml_data.csv shape:{ml_data.shape}")