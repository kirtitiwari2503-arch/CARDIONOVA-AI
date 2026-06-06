# CARDIONOVA Dataset Analysis

## Dataset Name
Cardiovascular Disease Dataset

## Shape
(70000, 13)

## Target Variable
cardio (0 = no disease, 1 = disease)

## Features

### Personal Data
- age (in days)
- gender
- height (cm)
- weight (kg)

### Vital Signs
- ap_hi (systolic blood pressure)
- ap_lo (diastolic blood pressure)

### Medical Indicators
- cholesterol
- gluc

### Lifestyle
- smoke
- alco
- active

## Data Quality
- No missing values found
- Dataset is clean

## Notes
- Age is stored in days (convert to years later)
- BMI can be created from height and weight
- We will engineer additional health risk features later