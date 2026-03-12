import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from collections import Counter
import warnings
warnings.simplefilter("ignore")

# ──────────────────────────────────────────────
# Load Data
# ──────────────────────────────────────────────

Train = pd.read_csv("../archive/Train-1542865627584.csv")
Train_Inpatientdata = pd.read_csv("../archive/Train_Inpatientdata-1542865627584.csv")
Train_Outpatientdata = pd.read_csv("../archive/Train_Outpatientdata-1542865627584.csv")
Train_Beneficiarydata = pd.read_csv("../archive/Train_Beneficiarydata-1542865627584.csv")

Test = pd.read_csv("../archive/Test-1542969243754.csv")
Test_Beneficiarydata = pd.read_csv("../archive/Test_Beneficiarydata-1542969243754.csv")
Test_Inpatientdata = pd.read_csv("../archive/Test_Inpatientdata-1542969243754.csv")
Test_Outpatientdata = pd.read_csv("../archive/Test_Outpatientdata-1542969243754.csv")

# ──────────────────────────────────────────────
# Beneficiary Data: Age & Alive/Dead
# ──────────────────────────────────────────────

for df in [Train_Beneficiarydata, Test_Beneficiarydata]:
    df['DOB'] = pd.to_datetime(df['DOB'], format='%Y-%m-%d')
    df['DOD'] = pd.to_datetime(df['DOD'], format='%Y-%m-%d', errors='ignore')
    df['Age'] = round((df['DOD'] - df['DOB']).dt.days / 365)
    df['Age'].fillna(round((pd.to_datetime('2009-12-01') - df['DOB']).dt.days / 365), inplace=True)
    df['AliveorDead'] = df['DOD'].notna().astype(int)

# ──────────────────────────────────────────────
# Inpatient Data
# ──────────────────────────────────────────────

for df in [Train_Inpatientdata, Test_Inpatientdata]:
    df['AdmissionDt'] = pd.to_datetime(df['AdmissionDt'], format='%Y-%m-%d')
    df['DischargeDt'] = pd.to_datetime(df['DischargeDt'], format='%Y-%m-%d')
    df['NumberofDaysAdmitted'] = (df['DischargeDt'] - df['AdmissionDt']).dt.days + 1
    df['ClaimEndDt'] = pd.to_datetime(df['ClaimEndDt'], format='%Y-%m-%d')
    df['ClaimStartDt'] = pd.to_datetime(df['ClaimStartDt'], format='%Y-%m-%d')
    df['DurationofClaim'] = (df['ClaimEndDt'] - df['ClaimStartDt']).dt.days
    df['Admitted'] = 1

# ──────────────────────────────────────────────
# Outpatient Data
# ──────────────────────────────────────────────

for df in [Train_Outpatientdata, Test_Outpatientdata]:
    df['ClaimEndDt'] = pd.to_datetime(df['ClaimEndDt'], format='%Y-%m-%d')
    df['ClaimStartDt'] = pd.to_datetime(df['ClaimStartDt'], format='%Y-%m-%d')
    df['DurationofClaim'] = (df['ClaimEndDt'] - df['ClaimStartDt']).dt.days
    df['Admitted'] = 0

# ──────────────────────────────────────────────
# Merge Inpatient + Outpatient
# ──────────────────────────────────────────────

common_cols = list(set(Train_Inpatientdata.columns).intersection(set(Train_Outpatientdata.columns)))

Train_Allpatientdata = pd.merge(Train_Outpatientdata, Train_Inpatientdata, on=common_cols, how='outer')
Test_Allpatientdata = pd.merge(Test_Outpatientdata, Test_Inpatientdata, on=common_cols, how='outer')

# Merge with Beneficiary data
df_train = Train_Allpatientdata.merge(Train_Beneficiarydata, on='BeneID', how='inner')
df_test = Test_Allpatientdata.merge(Test_Beneficiarydata, on='BeneID', how='inner')

# Merge with Provider fraud labels
df_train1 = pd.merge(Train, df_train, on='Provider')
df_test1 = pd.merge(Test, df_test, on='Provider')

# ──────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────

for df in [df_train1, df_test1]:
    df['RenalDiseaseIndicator'].replace('Y', '1', inplace=True)
    df['RenalDiseaseIndicator'] = df['RenalDiseaseIndicator'].astype(int)

df_train1.drop(columns=['DOB', 'DOD'], axis=1, inplace=True)
df_test1.drop(columns=['DOB', 'DOD'], axis=1, inplace=True)

df_train1['ClmDiagnosisCodeIndex'] = df_train1.filter(regex='ClmDiagnosisCode_').notnull().sum(axis=1)
df_test1['ClmDiagnosisCodeIndex'] = df_test1.filter(regex='ClmDiagnosisCode_').notnull().sum(axis=1)

df_train1['ClmProcedureCodeIndex'] = df_train1.filter(regex='ClmProcedureCode_').notnull().sum(axis=1)
df_test1['ClmProcedureCodeIndex'] = df_test1.filter(regex='ClmProcedureCode_').notnull().sum(axis=1)

columns_to_drop = df_train1.filter(regex='ClmProcedureCode_|ClmDiagnosisCode_').columns
df_train1 = df_train1.drop(columns_to_drop, axis=1)
df_test1 = df_test1.drop(columns_to_drop, axis=1)

df_train1['NumberofDaysAdmitted'] = df_train1['NumberofDaysAdmitted'].fillna(0)
df_test1['NumberofDaysAdmitted'] = df_test1['NumberofDaysAdmitted'].fillna(0)

df_train1 = df_train1.dropna(subset=['AttendingPhysician'])
df_test1 = df_test1.dropna(subset=['AttendingPhysician'])

df_train1['DeductibleAmtPaid'] = df_train1['DeductibleAmtPaid'].fillna(df_train1['DeductibleAmtPaid'].mean())
df_test1['DeductibleAmtPaid'] = df_test1['DeductibleAmtPaid'].fillna(df_test1['DeductibleAmtPaid'].mean())

# Per-Provider averages
provider_avg_cols = [
    "InscClaimAmtReimbursed", "DeductibleAmtPaid", "IPAnnualReimbursementAmt",
    "IPAnnualDeductibleAmt", "OPAnnualReimbursementAmt", "OPAnnualDeductibleAmt",
    "Age", "NoOfMonths_PartACov", "NoOfMonths_PartBCov", "DurationofClaim",
    "NumberofDaysAdmitted"
]
for col in provider_avg_cols:
    df_train1[f"PerProviderAvg_{col}"] = df_train1.groupby('Provider')[col].transform('mean')
    df_test1[f"PerProviderAvg_{col}"] = df_test1.groupby('Provider')[col].transform('mean')

# Per-BeneID and Per-AttendingPhysician averages
bene_physician_avg_cols = [
    "InscClaimAmtReimbursed", "DeductibleAmtPaid", "IPAnnualReimbursementAmt",
    "IPAnnualDeductibleAmt", "OPAnnualReimbursementAmt", "OPAnnualDeductibleAmt",
    "DurationofClaim", "NumberofDaysAdmitted"
]
for col in bene_physician_avg_cols:
    df_train1[f"PerBeneIDAvg_{col}"] = df_train1.groupby('BeneID')[col].transform('mean')
    df_test1[f"PerBeneIDAvg_{col}"] = df_test1.groupby('BeneID')[col].transform('mean')

    df_train1[f"PerAttendingPhysician Avg_{col}"] = df_train1.groupby('AttendingPhysician')[col].transform('mean')
    df_test1[f"PerAttendingPhysician Avg_{col}"] = df_test1.groupby('AttendingPhysician')[col].transform('mean')

# ──────────────────────────────────────────────
# Drop Unnecessary Columns
# ──────────────────────────────────────────────

df_train1.drop(columns=[
    'ClmAdmitDiagnosisCode', 'Provider', 'State', 'Race', 'Gender', 'County',
    'AdmissionDt', 'AttendingPhysician', 'OtherPhysician', 'OperatingPhysician',
    'DischargeDt', 'ClaimID', 'ClaimEndDt', 'DiagnosisGroupCode', 'ClaimStartDt', 'BeneID'
], axis=1, inplace=True)

df_test1.drop(columns=[
    'ClmAdmitDiagnosisCode', 'State', 'Race', 'County', 'Gender', 'AdmissionDt',
    'DiagnosisGroupCode', 'OperatingPhysician', 'DischargeDt', 'AttendingPhysician',
    'OtherPhysician', 'ClaimID', 'ClaimEndDt', 'ClaimStartDt'
], axis=1, inplace=True)

# ──────────────────────────────────────────────
# Train / Val Split
# ──────────────────────────────────────────────

df_train2, df_val = train_test_split(df_train1, test_size=0.10, random_state=42)

y_train = df_train2.pop('PotentialFraud')
X_train = df_train2

y_val = df_val.pop('PotentialFraud')
X_val = df_val

X_test = df_test1

# ──────────────────────────────────────────────
# One-Hot Encode ChronicCond_ Features
# ──────────────────────────────────────────────

categorical_cols = [col for col in X_train.columns if col.startswith('ChronicCond_')]
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

encoded_data_train = encoder.fit_transform(X_train[categorical_cols])
encoded_data_test = encoder.fit_transform(X_test[categorical_cols])
encoded_data_val = encoder.fit_transform(X_val[categorical_cols])

encoded_df_train = pd.DataFrame(encoded_data_train, columns=encoder.get_feature_names_out())
encoded_df_test = pd.DataFrame(encoded_data_test, columns=encoder.get_feature_names_out())
encoded_df_val = pd.DataFrame(encoded_data_val, columns=encoder.get_feature_names_out())

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
X_val = X_val.reset_index(drop=True)

X_train = pd.concat([X_train.drop(categorical_cols, axis=1), encoded_df_train], axis=1)
X_test = pd.concat([X_test.drop(categorical_cols, axis=1), encoded_df_test], axis=1)
X_val = pd.concat([X_val.drop(categorical_cols, axis=1), encoded_df_val], axis=1)

# ──────────────────────────────────────────────
# SMOTE — Handle Class Imbalance
# ──────────────────────────────────────────────

smt = SMOTE()
X_train, y_train = smt.fit_resample(X_train, y_train)

# ──────────────────────────────────────────────
# Save to CSV
# ──────────────────────────────────────────────

X_train.to_csv('xtrain.csv', index=False, header=False)
X_test.to_csv('xtest.csv', index=False, header=False)
X_val.to_csv('xval.csv', index=False, header=False)

y_train.to_csv('ytrain.csv', index=False, header=False)
y_val.to_csv('yval.csv', index=False, header=False)

# Map labels to binary integers
ytrain = pd.read_csv('ytrain.csv', header=None)
yval = pd.read_csv('yval.csv', header=None)

ytrain[0] = ytrain[0].map({'Yes': 1, 'No': 0})
yval[0] = yval[0].map({'Yes': 1, 'No': 0})

ytrain.to_csv('ytrain.csv', index=False, header=False)
yval.to_csv('yval.csv', index=False, header=False)