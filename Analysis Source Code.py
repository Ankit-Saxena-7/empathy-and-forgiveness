import pandas as pd
import numpy as np

from scipy.stats import mannwhitneyu

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

dfMasterData = pd.read_excel('Final Analysis.xlsx', sheet_name=0)

print("Data Description:" + "\n")
print(dfMasterData.info())
print("\n")

# Data Exploration
print("Data Exploration:")
print(dfMasterData.groupby('Gender', as_index=False)['Unforgiveness'].mean())
print("\n")
print(dfMasterData.groupby('Race', as_index=False)['Unforgiveness'].mean())
print("\n")
print(dfMasterData.groupby('Education Level', as_index=False)['Unforgiveness'].mean())
print("\n")
print(dfMasterData.groupby('Image Race', as_index=False)['Unforgiveness'].mean())
print("\n")
print(dfMasterData.groupby('Image Gender', as_index=False)['Unforgiveness'].mean())
print("\n")
print(dfMasterData.groupby('Prime', as_index=False)['Unforgiveness'].mean())
print("\n")
print(dfMasterData.groupby('Group Match', as_index=False)['Unforgiveness'].mean())
print("\n")

# Boolean conditions
Prime = dfMasterData['Prime'] == "E-TRIM"
NonPrime = dfMasterData['Prime'] == "TRIM-E"
InGroup = dfMasterData['Group Match'] == "In-group"
OutGroup = dfMasterData['Group Match'] == "Out-group"
Male = dfMasterData['Gender'] == "Male"
Female = dfMasterData['Gender'] == "Female"

# Pairwise treatment groups
dfInPrime = dfMasterData[InGroup & Prime].copy()
dfOutPrime = dfMasterData[OutGroup & Prime].copy()
dfInNonPrime = dfMasterData[InGroup & NonPrime].copy()
dfOutNonPrime = dfMasterData[OutGroup & NonPrime].copy()

# In-group and out-group samples
dfInGroup = dfMasterData[InGroup].copy()
dfOutGroup = dfMasterData[OutGroup].copy()

# In-group and out-group samples
dfInGroup = dfMasterData[InGroup].copy()
dfOutGroup = dfMasterData[OutGroup].copy()

# Males primed and not primed with empathy
dfMalePrime = dfMasterData[Prime & Male].copy()
dfMaleNonPrime = dfMasterData[NonPrime & Male].copy()

# Males primed and not primed with empathy
dfFemalePrime = dfMasterData[Prime & Female].copy()
dfFemaleNonPrime = dfMasterData[NonPrime & Female].copy()

# MAN WHITNEY U TEST (Two-Sided)
print("\n")

vUStat, vPValue = mannwhitneyu(dfInPrime.Unforgiveness, dfOutPrime.Unforgiveness, alternative='two-sided')
print('In-group Prime <> Out-group Prime | U = %.3f, p = %.3f' % (vUStat, vPValue))

vUStat, vPValue = mannwhitneyu(dfInPrime.Unforgiveness, dfInNonPrime.Unforgiveness, alternative='two-sided')
print('In-group Prime <> In-group Non-Prime | U = %.3f, p = %.3f' % (vUStat, vPValue))

vUStat, vPValue = mannwhitneyu(dfInPrime.Unforgiveness, dfOutNonPrime.Unforgiveness, alternative='two-sided')
print('In-group Prime <> Out-group Non-Prime | U = %.3f, p = %.3f' % (vUStat, vPValue))

vUStat, vPValue = mannwhitneyu(dfOutPrime.Unforgiveness, dfInNonPrime.Unforgiveness, alternative='two-sided')
print('Out-group Prime <> In-group Non-Prime | U = %.3f, p = %.3f' % (vUStat, vPValue))

vUStat, vPValue = mannwhitneyu(dfOutPrime.Unforgiveness, dfOutNonPrime.Unforgiveness, alternative='two-sided')
print('Out-group Prime <> Out-group Non-Prime | U = %.3f, p = %.3f' % (vUStat, vPValue))

vUStat, vPValue = mannwhitneyu(dfInNonPrime.Unforgiveness, dfOutNonPrime.Unforgiveness, alternative='two-sided')
print('In-group Non-Prime <> Out-group Non-Prime | U = %.3f, p = %.3f' % (vUStat, vPValue))

print("\n")

vUStat, vPValue = mannwhitneyu(dfInGroup.Unforgiveness, dfOutGroup.Unforgiveness, alternative='two-sided')
print('In-group <> Out-group | U = %.3f, p = %.3f' % (vUStat, vPValue))

print("\n")

vUStat, vPValue = mannwhitneyu(dfMalePrime.Unforgiveness, dfMaleNonPrime.Unforgiveness, alternative='two-sided')
print('Male Prime <> Male NonPrime | U = %.3f, p = %.3f' % (vUStat, vPValue))

print("\n")

vUStat, vPValue = mannwhitneyu(dfFemalePrime.Unforgiveness, dfFemaleNonPrime.Unforgiveness, alternative='two-sided')
print('Female Prime <> Female NonPrime | U = %.3f, p = %.3f' % (vUStat, vPValue))

# REGRESSION
dfMasterDataRegression = dfMasterData.copy().drop(columns=['Image Race', 'Image Gender'], axis='columns')

vCategoricalColumns = ['Gender', 'Race', 'Education Level', 'Prime', 'Group Match']

dfDummyVariables = pd.get_dummies(dfMasterDataRegression, prefix=['Gender', 'Race', 'Education', 'Prime', 'Group'], prefix_sep=[' ', ' ', ' ', ' ', ' '], columns=vCategoricalColumns)

# Dropping one of the categories in each dummy group
dfDummyVariables.drop(columns=['Age', 'Unforgiveness', 'Gender Male', 'Race Other', 'Education Other', 'Prime TRIM-E', 'Group Out-group'], inplace=True, axis='columns')

dfMasterDataRegression = dfMasterDataRegression.drop(vCategoricalColumns, axis='columns')

dfMasterDataRegression = pd.concat([dfMasterDataRegression, dfDummyVariables], axis='columns')

dfLinearRegressionX = dfMasterDataRegression.iloc[:, np.r_[0, 2:len(dfMasterDataRegression.columns)]]

dfLinearRegressionY = dfMasterDataRegression.iloc[:, 1]

print('\n')
# COMPREHENSIVE REGRESSION
print('Stats Model Regression')
dfLinearRegressionXOLS = sm.add_constant(dfLinearRegressionX)
vModel = sm.OLS(dfLinearRegressionY, dfLinearRegressionXOLS).fit()
print(vModel.summary())

# BASIC STATS REGRESSION
print('Sklearn Regression')
vRegressor = LinearRegression()
vRegressor.fit(dfLinearRegressionX, dfLinearRegressionY)

vCoefficients = pd.DataFrame(vRegressor.coef_, dfLinearRegressionX.columns, columns=['Coefficient'])
print('Coefficients:')
print(vCoefficients)

print('\n')

print('Intercept:')
print(vRegressor.intercept_)

print('\n')

print('Coefficient of determination:')
print(vRegressor.score(dfLinearRegressionX, dfLinearRegressionY))
print('\n')

# DATA FOR THE CI CHARTS

# Pairwise tests
print("Pairwise")
dfMeans = pd.DataFrame(data={'InPrime': [dfInPrime['Unforgiveness'].mean()], 'OutPrime': [dfOutPrime['Unforgiveness'].mean()], 'InNonPrime': [dfInNonPrime['Unforgiveness'].mean()], 'OutNonPrime': [dfOutNonPrime['Unforgiveness'].mean()]})
dfSTD = pd.DataFrame(data={'InPrime': [dfInPrime['Unforgiveness'].std()], 'OutPrime': [dfOutPrime['Unforgiveness'].std()], 'InNonPrime': [dfInNonPrime['Unforgiveness'].std()], 'OutNonPrime': [dfOutNonPrime['Unforgiveness'].std()]})

print('Means:')
print(dfMeans)

print('STD:')
print(dfSTD)

print('\n')

# Hypothesis I
print("Hypothesis I")
dfMeans = pd.DataFrame(data={'In-Group': [dfInGroup['Unforgiveness'].mean()], 'Out-Group': [dfOutGroup['Unforgiveness'].mean()]})
dfSTD = pd.DataFrame(data={'In-Group': [dfInGroup['Unforgiveness'].std()], 'Out-Group': [dfOutGroup['Unforgiveness'].std()]})

print('Means:')
print(dfMeans)

print('STD:')
print(dfSTD)

print('\n')

# Hypothesis II
print("Hypothesis II")
dfMeans = pd.DataFrame(data={'Male Prime': [dfMalePrime['Unforgiveness'].mean()], 'Male Non-Prime': [dfMaleNonPrime['Unforgiveness'].mean()]})
dfSTD = pd.DataFrame(data={'Male Prime': [dfMalePrime['Unforgiveness'].std()], 'Male Non-Prime': [dfMaleNonPrime['Unforgiveness'].std()]})

print('Means:')
print(dfMeans)

print('STD:')
print(dfSTD)

# MANN WHITNEY U TEST (Two-Sided)
print("\n")

dfMalePrime = dfMasterData[Prime & Male].copy()
dfMaleNonPrime = dfMasterData[NonPrime & Male].copy()

# Males primed and not primed with empathy
dfFemalePrime = dfMasterData[Prime & Female].copy()
dfFemaleNonPrime = dfMasterData[NonPrime & Female].copy()

vUStat, vPValue = mannwhitneyu(dfMalePrime.Unforgiveness, dfFemalePrime.Unforgiveness, alternative='two-sided')
print('Male Prime <> Female Prime | U = %.3f, p = %.3f' % (vUStat, vPValue))

vUStat, vPValue = mannwhitneyu(dfMalePrime.Unforgiveness, dfFemaleNonPrime.Unforgiveness, alternative='two-sided')
print('Male Prime <> Female Non-Prime | U = %.3f, p = %.3f' % (vUStat, vPValue))

vUStat, vPValue = mannwhitneyu(dfMaleNonPrime.Unforgiveness, dfFemalePrime.Unforgiveness, alternative='two-sided')
print('Male Non-Prime <> Female Prime | U = %.3f, p = %.3f' % (vUStat, vPValue))

vUStat, vPValue = mannwhitneyu(dfMaleNonPrime.Unforgiveness, dfFemaleNonPrime.Unforgiveness, alternative='two-sided')
print('Male Non-Prime <> Female Non-Prime | U = %.3f, p = %.3f' % (vUStat, vPValue))
