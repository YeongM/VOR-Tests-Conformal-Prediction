# VOR-Tests-Conformal-Prediction
Conformal Prediction Based Clinical Decision Support Using VOR Tests: Validation of Clinically Meaningful Uncertainty

# modeling.py
- df's columns: ['age', 'sex', 'Gain_0.01', 'Gain_0.02', 'Gain_0.04','Gain_0.08', 'Gain_0.16', 'Gain_0.32', 'Gain_0.64', 'Asymmetry_0.01','Asymmetry_0.02', 'Asymmetry_0.04', 'Asymmetry_0.08', 'Asymmetry_0.16','Asymmetry_0.32', 'Asymmetry_0.64', 'Phase_0.01', 'Phase_0.02','Phase_0.04', 'Phase_0.08', 'Phase_0.16', 'Phase_0.32', 'Phase_0.64','RtCool', 'RtWarm', 'LtWarm', 'LtCool', 'RtSum', 'LtSum', 'TotalSum','CP', 'DP', 'Label']
- Label: 0-Normal, 1-Right peripheral vestibular dysfunction, 2-Left peripheral vestibular dysfunction, 3-Others

# CP.py
- df's columns: ['id', 'Label', 'class1_proba', 'class2_proba', 'class3_proba','class4_proba', 'Q1_expert1', 'Q2_expert1', 'Q3_expert1', 'Q1_expert2','Q2_expert2', 'Q3_expert2', 'Q1_expert3', 'Q2_expert3', 'Q3_expert3','Q1_expert4', 'Q2_expert4', 'Q3_expert4', 'Q1_expert5', 'Q2_expert5','Q3_expert5']
- Label: 0-Normal, 1-Right peripheral vestibular dysfunction, 2-Left peripheral vestibular dysfunction, 3-Others
- Q1: 1,2,3,4 (Normal, Right peripheral vestibular dysfunction, Left peripheral vestibular dysfunction, Others)
- Q2, Q3: 0,1,2,3,4 (No additional likely diagnosis, Normal, Right peripheral vestibular dysfunction, Left peripheral vestibular dysfunction, Others)
