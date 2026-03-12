import pandas as pd
import torch
import shap

# ==============================
# 1. LOAD DATA
# ==============================

xtrain = pd.read_csv("xtrain.csv", header=None)
xtest = pd.read_csv("xval.csv", header=None)

feature_names = [
'InscClaimAmtReimbursed','DeductibleAmtPaid','Admitted','DurationofClaim',
'NumberofDaysAdmitted','RenalDiseaseIndicator','NoOfMonths_PartACov',
'NoOfMonths_PartBCov','IPAnnualReimbursementAmt','IPAnnualDeductibleAmt',
'OPAnnualReimbursementAmt','OPAnnualDeductibleAmt','Age','AliveorDead',
'ClmDiagnosisCodeIndex','ClmProcedureCodeIndex',
'PerProviderAvg_InscClaimAmtReimbursed','PerProviderAvg_DeductibleAmtPaid',
'PerProviderAvg_IPAnnualReimbursementAmt','PerProviderAvg_IPAnnualDeductibleAmt',
'PerProviderAvg_OPAnnualReimbursementAmt','PerProviderAvg_OPAnnualDeductibleAmt',
'PerProviderAvg_Age','PerProviderAvg_NoOfMonths_PartACov',
'PerProviderAvg_NoOfMonths_PartBCov','PerProviderAvg_DurationofClaim',
'PerProviderAvg_NumberofDaysAdmitted','PerBeneIDAvg_InscClaimAmtReimbursed',
'PerAttendingPhysician_Avg_InscClaimAmtReimbursed','PerBeneIDAvg_DeductibleAmtPaid',
'PerAttendingPhysician_Avg_DeductibleAmtPaid',
'PerBeneIDAvg_IPAnnualReimbursementAmt',
'PerAttendingPhysician_Avg_IPAnnualReimbursementAmt',
'PerBeneIDAvg_IPAnnualDeductibleAmt',
'PerAttendingPhysician_Avg_IPAnnualDeductibleAmt',
'PerBeneIDAvg_OPAnnualReimbursementAmt',
'PerAttendingPhysician_Avg_OPAnnualReimbursementAmt',
'PerBeneIDAvg_OPAnnualDeductibleAmt',
'PerAttendingPhysician_Avg_OPAnnualDeductibleAmt',
'PerBeneIDAvg_DurationofClaim',
'PerAttendingPhysician_Avg_DurationofClaim',
'PerBeneIDAvg_NumberofDaysAdmitted',
'PerAttendingPhysician_Avg_NumberofDaysAdmitted',
'ChronicCond_Alzheimer_1','ChronicCond_Alzheimer_2',
'ChronicCond_Heartfailure_1','ChronicCond_Heartfailure_2',
'ChronicCond_KidneyDisease_1','ChronicCond_KidneyDisease_2',
'ChronicCond_Cancer_1','ChronicCond_Cancer_2',
'ChronicCond_ObstrPulmonary_1','ChronicCond_ObstrPulmonary_2',
'ChronicCond_Depression_1','ChronicCond_Depression_2',
'ChronicCond_Diabetes_1','ChronicCond_Diabetes_2',
'ChronicCond_IschemicHeart_1','ChronicCond_IschemicHeart_2',
'ChronicCond_Osteoporasis_1','ChronicCond_Osteoporasis_2',
'ChronicCond_rheumatoidarthritis_1','ChronicCond_rheumatoidarthritis_2',
'ChronicCond_stroke_1','ChronicCond_stroke_2'
]

xtrain.columns = feature_names
xtest.columns = feature_names


# ==============================
# 2. CONVERT TO TENSORS
# ==============================

X_train_tensor = torch.tensor(xtrain.values, dtype=torch.float32)
X_test_tensor = torch.tensor(xtest.values, dtype=torch.float32)


# ==============================
# 3. LOAD MODEL
# ==============================

model = model.to("cpu")   # assuming model already loaded
model.eval()


# ==============================
# 4. CREATE SHAP EXPLAINER
# ==============================

background = X_train_tensor[:200]

explainer = shap.DeepExplainer(model, background)


# ==============================
# 5. COMPUTE SHAP VALUES
# ==============================

sample_size = 100

shap_values = explainer.shap_values(X_test_tensor[:sample_size])

# If output has extra dimension (binary classifier with single output)
if isinstance(shap_values, list):
    shap_values = shap_values[0]

# Remove last dimension
shap_values = shap_values[:, :, 0]

print("SHAP shape after fix:", shap_values.shape)

# ==============================
# 6. BEESWARM PLOT
# ==============================

shap.summary_plot(
    shap_values,
    xtest.iloc[:sample_size],
    feature_names=feature_names
)


# ==============================
# 7. BAR PLOT (GLOBAL IMPORTANCE)
# ==============================

shap.summary_plot(
    shap_values,
    xtest.iloc[:sample_size],
    feature_names=feature_names,
    plot_type="bar"
)