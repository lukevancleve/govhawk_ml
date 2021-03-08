# Create the partisan lean feature
# The feature is calculated from the metadata. At runtime an ETL job will need to lookup the
# partisan lean for a specific session-chamber combination

import os
import pandas as pd

METADATA_DIR = "references/external/"  # Raw stuff
DERIVED_DIR = "references/derived/"    # Calculated from metadata

if not os.path.exists(DERIVED_DIR):
    os.makedirs(DERIVED_DIR)

# Load in all the metadata
bills =  pd.read_csv(METADATA_DIR + "/bill.csv", sep=";", encoding="latin1", parse_dates=True).rename(columns={'id': 'bill_id'})
bv = pd.read_csv(METADATA_DIR + "/bill_version.csv", sep=";", encoding="latin1", parse_dates=True)
sessions =  pd.read_csv(METADATA_DIR + "/session.csv", sep=";", encoding="latin1", parse_dates=True)
divs =  pd.read_csv(METADATA_DIR + "/division.csv", sep=";", encoding="latin1", parse_dates=True)
votes =  pd.read_csv(METADATA_DIR + "/vote.csv", sep=";", encoding="latin1", parse_dates=True)
bill_leg_vote =  pd.read_csv(METADATA_DIR + "/bill_legislator_vote.csv", sep=";", encoding="latin1", parse_dates=True)
vote_rec_type =  pd.read_csv(METADATA_DIR + "/vote_recorded_type.csv", sep=";", encoding="latin1", parse_dates=True)
party =  pd.read_csv(METADATA_DIR + "/party.csv", sep=";", encoding="latin1", parse_dates=True)
persons =  pd.read_csv(METADATA_DIR + "/person.csv", sep=";", encoding="latin1", parse_dates=True)

# metadata wrangling
party = party.rename(columns={"id": "party_id"})


bills = bills[~bills['chamber_id'].isna()]
bills['chamber_id'] = bills['chamber_id'].astype(int)
bills['sc_id'] = bills['session_id'].astype(str) + "-" + bills['chamber_id'].astype(str)


bills.loc[bills['signed'].isna(), 'signed'] = 0
bills.loc[bills['signed'] != 0, 'signed'] = 1
bills['signed'] = bills.signed.astype(int)

bills.loc[bills['passed_lower'].isna(), 'passed_lower'] = 0
bills.loc[bills['passed_lower'] != 0, 'passed_lower'] = 1
bills['passed_lower'] = bills.passed_lower.astype(int)

bills.loc[bills['passed_upper'].isna(), 'passed_upper'] = 0
bills.loc[bills['passed_upper'] != 0, 'passed_upper'] = 1
bills['passed_upper'] = bills.passed_upper.astype(int)


passage_percentage = bills[['signed', 'sc_id']].groupby('sc_id').agg(['mean', 'count']).reset_index()
passage_percentage.columns = ['sc_id', 'mean', 'count']
passage_percentage.to_csv(DERIVED_DIR + 'passage_percentage.csv', index=False)

unique_bills = pd.DataFrame(bv['bill_id'].unique(), columns=['bill_id'])
sessions = sessions.rename(columns={'id':'session_id'})
votes2 = votes.merge(sessions, on = ['session_id', 'division_id'], how='left')
votes2 = votes2[['id', 'division_id', 'session_id', 'bill_id', "chamber_id", 'yes', 'no', 'other']]

party["is_liberal"] = 1
conservatives = set(["Republican", "Republican \n"])
mask = party["name"].isin(conservatives)
party.loc[mask, "is_liberal"] = 0
party.head()
persons= persons.merge(party[["party_id", "is_liberal"]], left_on = "party_id", right_on = "party_id")
persons = persons.rename(columns = {"id": "person_id"})

vote_rec_type = vote_rec_type.rename(columns={"id": "vote_recorded_type_id"})
blv = bill_leg_vote.merge(vote_rec_type, left_on = 'vote_recorded_type_id', right_on = "vote_recorded_type_id")
blv = blv.merge(persons[["person_id", "is_liberal"]], on = "person_id")

lib_vote_count = blv[['bill_id', 'vote_id', 'is_liberal']].groupby(['bill_id', 'vote_id']).agg(['sum', 'count'])
lib_vote_count = lib_vote_count.reset_index()
lib_vote_count.columns = ['bill_id', 'vote_id', 'n_lib_votes', 'total_votes']
lib_vote_count.head()

votes3 = votes2.merge(lib_vote_count, on = ['bill_id'], how='left')
votes3 = votes3.merge(unique_bills, on = 'bill_id')
votes3 = votes3.merge(divs[['id', 'abbr']], left_on='division_id', right_on='id')

total_votes = votes3[['abbr', 'division_id', 'session_id', 'chamber_id', 'yes', 'n_lib_votes', 'total_votes']].groupby(['abbr', 'division_id', 'session_id', 'chamber_id']).agg(['count', 'sum'])
a = total_votes['n_lib_votes']['sum'] / total_votes['total_votes']['sum'] 
a = a.to_frame().reset_index().rename(columns={"sum":"partisan_lean"})
a = a[a.chamber_id != 153]
a = a[a.abbr != 'PR']

a.loc[(a['abbr'] == 'AK') & (a['chamber_id'] == 1), 'partisan_lean'] = 15/40
a.loc[(a['abbr'] == 'AK') & (a['chamber_id'] == 2), 'partisan_lean'] = 7/20
a.loc[(a['abbr'] == 'ME') & (a['chamber_id'] == 1), 'partisan_lean'] = 88/151
a.loc[(a['abbr'] == 'ME') & (a['chamber_id'] == 2), 'partisan_lean'] = 21/35
a.loc[(a['abbr'] == 'MO') & (a['chamber_id'] == 1), 'partisan_lean'] = 48/163
a.loc[(a['abbr'] == 'MO') & (a['chamber_id'] == 2), 'partisan_lean'] = 10/34
a.loc[(a['abbr'] == 'MT') & (a['chamber_id'] == 1), 'partisan_lean'] = 41/100
a.loc[(a['abbr'] == 'MT') & (a['chamber_id'] == 2), 'partisan_lean'] = 20/50
a.loc[(a['abbr'] == 'OK') & (a['chamber_id'] == 1), 'partisan_lean'] = 24/101
a.loc[(a['abbr'] == 'OK') & (a['chamber_id'] == 2), 'partisan_lean'] = 9/39
a.loc[(a['abbr'] == 'SC') & (a['chamber_id'] == 1), 'partisan_lean'] = 45/124
a.loc[(a['abbr'] == 'SC') & (a['chamber_id'] == 2), 'partisan_lean'] = 19/46
a.loc[(a['abbr'] == 'UT') & (a['chamber_id'] == 1), 'partisan_lean'] = 16/75
a.loc[(a['abbr'] == 'UT') & (a['chamber_id'] == 2), 'partisan_lean'] = 6/29
a.loc[(a['abbr'] == 'VA') & (a['chamber_id'] == 1), 'partisan_lean'] = 55/100
a.loc[(a['abbr'] == 'VA') & (a['chamber_id'] == 2), 'partisan_lean'] = 21/40


a.to_csv(DERIVED_DIR + "partisan_lean.csv")