#%%[markdown] 
# HINTS Final Project 
#

#
# TITLE: Cancer Diagnosis Classification
# Source: HINTS 5, Cycle 3 (2019)

#%% 
# libraries
import pandas as pd
import numpy as np 
import matplotlib as mp
import seaborn as sns
import matplotlib.pyplot as plt
import rfit 
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency
from statsmodels.formula.api import ols
import statsmodels.api as sm

#%% 
# create subset data file
# 
# import data
hints_dataset = pd.read_csv("HINTS_cleaned_dataset.csv")
cols_of_interest = ['personid','hhid','raceethn','generalhealth','genderc','whendiagnosedcancer','age','everhadcancer','vegetables','fruit','influencecancer_eatingfruitveg',\
    'timesmoderateexercise','howlongmoderateexerciseminutes','timesstrengthtraining','exrec_notheard','averagesleepnight','averagesleepquality','drinkdaysperweek',\
        'smokenow','bmi','spendtimeinsuntanning']

hints_dataset.head()

# look at data type and basic statistics
rfit.dfchk(hints_dataset)

#%%[markdown] 
#####################
# Variable cleaning #
#####################

#%% 
# Correlation matrix of variables of interest
corr_matrix_full = hints_dataset.loc[:,~hints_dataset.columns.isin(['personid','hhid','Unnamed: 0'])].corr()
ax = sns.heatmap(
    corr_matrix_full, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

#%% 
# View unique values of all variables of interest before cleaning
print("\n raceethn Unique Values : ",hints_dataset['raceethn'].unique())
print("\n genderc Unique Values : ",hints_dataset['genderc'].unique())
print("\n everhadcancer Unique Values : ",hints_dataset['everhadcancer'].unique())
print("\n age Unique Values : ",hints_dataset['age'].unique())
print("\n vegetables Unique Values : ",hints_dataset['vegetables'].unique())
print("\n fruit Unique Values : ",hints_dataset['fruit'].unique())
print("\n influencecancer_eatingfruitveg Unique Values : ",hints_dataset['influencecancer_eatingfruitveg'].unique())
print("\n timesmoderateexercise Unique Values : ",hints_dataset['timesmoderateexercise'].unique())
print("\n howlongmoderateexerciseminutes Unique Values : ",hints_dataset['howlongmoderateexerciseminutes'].unique())
print("\n timesstrengthtraining Unique Values : ",hints_dataset['timesstrengthtraining'].unique())
print("\n exrec_notheard Unique Values : ",hints_dataset['exrec_notheard'].unique())
print("\n averagesleepnight Unique Values : ",hints_dataset['averagesleepnight'].unique())
print("\n averagesleepquality Unique Values : ",hints_dataset['averagesleepquality'].unique())
print("\n drinkdaysperweek Unique Values : ",hints_dataset['drinkdaysperweek'].unique())
print("\n smokenow Unique Values : ",hints_dataset['smokenow'].unique())
print("\n bmi Unique Values : ",hints_dataset['bmi'].unique())
print("\n spendtimeinsuntanning Unique Values : ",hints_dataset['spendtimeinsuntanning'].unique())

#%%
# Clean demographic variables
# race/ethnicity
hints_dataset['raceethn'] = hints_dataset['raceethn'].replace(-9, np.nan)
hints_dataset['raceethn'] = hints_dataset['raceethn'].replace(-7, np.nan)
hints_dataset['raceethn'] = hints_dataset['raceethn'].replace(1, 'Hispanic')
hints_dataset['raceethn'] = hints_dataset['raceethn'].replace(2, 'White')
hints_dataset['raceethn'] = hints_dataset['raceethn'].replace(3, 'Black')
hints_dataset['raceethn'] = hints_dataset['raceethn'].replace(4, 'Other')
hints_dataset['raceethn'] = hints_dataset['raceethn'].replace(5, 'Asian')
hints_dataset['raceethn'] = hints_dataset['raceethn'].replace(6, 'Other')
hints_dataset['raceethn'] = hints_dataset['raceethn'].replace(7, 'Other')
hints_dataset['raceethn'] = pd.Categorical(hints_dataset['raceethn'], 
                                           categories=['White', 'Black', 'Hispanic','Asian', 'Other'], 
                                           ordered=True)
print("\n raceethn Unique Values after cleaning : ",hints_dataset['raceethn'].unique())

# gender; 1=male, 2=female
hints_dataset['genderc'] = hints_dataset['genderc'].replace(-9, np.nan)
hints_dataset['genderc'] = hints_dataset['genderc'].replace(-7, np.nan)
print("\n genderc Unique Values after cleaning : ",hints_dataset['genderc'].unique())

# general health
hints_dataset['generalhealth'] = hints_dataset['generalhealth'].replace(-9, np.nan)
hints_dataset['generalhealth'] = hints_dataset['generalhealth'].replace(-7, np.nan)
hints_dataset['generalhealth'] = hints_dataset['generalhealth'].replace(-5, np.nan)
hints_dataset['generalhealth'] = hints_dataset['generalhealth'].replace(1, 'Excellent')
hints_dataset['generalhealth'] = hints_dataset['generalhealth'].replace(2, 'Very_Good')
hints_dataset['generalhealth'] = hints_dataset['generalhealth'].replace(3, 'Good')
hints_dataset['generalhealth'] = hints_dataset['generalhealth'].replace(4, 'Fair')
hints_dataset['generalhealth'] = hints_dataset['generalhealth'].replace(5, 'Poor')
hints_dataset['generalhealth'] = pd.Categorical(hints_dataset['generalhealth'], 
                                           categories=['Excellent', 'Very_Good', 'Good', 'Fair', 'Poor'], 
                                           ordered=True)
print("\n generalhealth Unique Values after cleaning : ",hints_dataset['generalhealth'].unique())

# age - continuous
hints_dataset['age'] = hints_dataset['age'].replace(-9, np.nan)
hints_dataset['age'] = hints_dataset['age'].replace(-4, np.nan)

bins = [18, 24, 64, 98]
labels = ['18-24', '25-64', '65+']
hints_dataset['age_group'] = pd.cut(hints_dataset['age'], bins=bins, labels=labels)

print("\n age Unique Values after cleaning : ",hints_dataset['age'].unique())
print("\n age_group Unique Values after cleaning : ",hints_dataset['age_group'].unique())

#%%
# Clean exercise variables
# timesmoderateexercise - continuous days per week
hints_dataset['timesmoderateexercise'] = hints_dataset['timesmoderateexercise'].replace([-7, -9], np.nan)
print("\n timesmoderateexercise Unique Values after cleaning : ",hints_dataset['timesmoderateexercise'].unique())

# howlongmoderateexerciseminutes - continuous in minutes
hints_dataset['howlongmoderateexerciseminutes'] = hints_dataset['howlongmoderateexerciseminutes'].replace([-1], 0)
print("\n howlongmoderateexerciseminutes Unique Values after cleaning : ",hints_dataset['howlongmoderateexerciseminutes'].unique())

# timesstrengthtraining - continuous days per week
hints_dataset['timesstrengthtraining'] = hints_dataset['timesstrengthtraining'].replace([-5, -7, -9], np.nan)
print("\n timesstrengthtraining Unique Values after cleaning : ",hints_dataset['timesstrengthtraining'].unique())

# exrec_notheard; 1 = not heard, 2 = heard
hints_dataset['exrec_notheard'] = hints_dataset['exrec_notheard'].replace([-7, -9], np.nan)
print("\n exrec_notheard Unique Values after cleaning : ",hints_dataset['exrec_notheard'].unique())

#%%
# Clean sleep variables
# averagesleepnight - continuous in hours
hints_dataset['averagesleepnight'] = hints_dataset['averagesleepnight'].replace([-4,-7,-9], np.nan)
print("\n averagesleepnight Unique Values after cleaning : ",hints_dataset['averagesleepnight'].unique())

# averagesleepquality; 1=very good, 2=fairly good, 3=fairy bad, 4=very bad
hints_dataset['averagesleepquality'] = hints_dataset['averagesleepquality'].replace([-5,-7,-9], np.nan)
print("\n averagesleepquality Unique Values after cleaning : ",hints_dataset['averagesleepquality'].unique())

#%%
# Clean diet & alcohol variables
data = hints_dataset

def clean_columns(data):
    fv = ['fruit','vegetables']
    for col in fv:
        data.loc[(data[col].isin([-9,-7,-5])),[col]]=np.nan
        data.loc[(data[col]==1),[col]]='0-0.5'
        data.loc[(data[col]==2),[col]]='0.5-1'
        data.loc[(data[col]==3),[col]]='1-2'
        data.loc[(data[col]==4),[col]]='2-3'
        data.loc[(data[col]==5),[col]]='3-4'
        data.loc[(data[col]==6),[col]]='4 or more'
    dpi = ['influencecancer_eatingfruitveg','drinkdaysperweek']
    for col in dpi:
        data.loc[(data[col].isin([-9,-7,-4,-5])),[col]]=np.nan
    data.loc[(data['influencecancer_eatingfruitveg']==1),['influencecancer_eatingfruitveg']]='A lot'
    data.loc[(data['influencecancer_eatingfruitveg']==2),['influencecancer_eatingfruitveg']]='A little'
    data.loc[(data['influencecancer_eatingfruitveg']==3),['influencecancer_eatingfruitveg']]='Not at all'
    data.loc[(data['influencecancer_eatingfruitveg']==4),['influencecancer_eatingfruitveg']]='Dont know'

    return data

hints_dataset = clean_columns(data)

hints_dataset['fruit'] = pd.Categorical(hints_dataset['fruit'], 
                                           categories=['0-0.5', '0.5-1', '1-2','2-3', '3-4', '4 or more'], 
                                           ordered=True)
hints_dataset['vegetables'] = pd.Categorical(hints_dataset['vegetables'], 
                                           categories=['0-0.5', '0.5-1', '1-2','2-3', '3-4', '4 or more'], 
                                           ordered=True)

print("\n fruit Unique Values after cleaning : ",hints_dataset['fruit'].unique())
print("\n vegetables Unique Values after cleaning : ",hints_dataset['vegetables'].unique())
print("\n influencecancer_eatingfruitveg Unique Values after cleaning : ",hints_dataset['influencecancer_eatingfruitveg'].unique())
print("\n drinkdaysperweek Unique Values after cleaning : ",hints_dataset['drinkdaysperweek'].unique())

#%%
# Clean other variables
# bmi - continuous (kg/m^2)
hints_dataset['bmi'] = hints_dataset['bmi'].replace([-4, -7, -9], np.nan)
print("\n bmi Unique Values after cleaning : ",hints_dataset['bmi'].unique())

# smokenow; 1=every day, 2=some days, 3=not at all
hints_dataset['smokenow'] = hints_dataset['smokenow'].replace([-1,-2,-6,-7,-9], np.nan)
print("\n smokenow Unique Values after cleaning : ", hints_dataset['smokenow'].unique())

# spendtimeinsuntanning; 1=often, 2=sometimes, 3-rarely, 4=never, 5=don't go out on sunny days
hints_dataset['spendtimeinsuntanning'] = hints_dataset['spendtimeinsuntanning'].replace([-5,-7,-9], np.nan)
print("\n spendtimeinsuntanning Unique Values after cleaning : ",hints_dataset['spendtimeinsuntanning'].unique())

# Clean outcome of interest
# ever had cancer; 1=yes, 0=no
hints_dataset['everhadcancer'] = hints_dataset['everhadcancer'].replace([-7, -9], np.nan)
hints_dataset['everhadcancer'] = hints_dataset['everhadcancer'].replace([2], 0)
print("\n everhadcancer Unique Values after cleaning : ",hints_dataset['everhadcancer'].unique())

#%%[markdown]
##########################
# Behavioral factors EDA #
##########################

#%%
#Demographic factors EDA
#Let's create a pie chart of the proportions of races and ethnicity
counts = hints_dataset['raceethn'].value_counts()
percentages = counts / counts.sum() * 100
plt.pie(percentages, labels=['']*len(percentages), autopct='%1.1f%%', textprops={'color':'white'})
plt.legend(percentages.index, title='Proportion of Race and Ethnicity', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.axis('off')
plt.show()

#Let's create a pie chart of the proportions of gender
counts = hints_dataset['genderc'].value_counts()
percentages = counts / counts.sum() * 100
plt.pie(percentages, labels=percentages.index, autopct='%1.1f%%', textprops={'color':'white'})
plt.title('Gender Proportions')
plt.legend(percentages.index, title = 'Male and Female Proportions', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

#Let's create a pie chart of the proportions of health
colors = ['green', 'blue', 'orange', 'purple', 'red']
counts = hints_dataset['generalhealth'].value_counts()
percentages = counts / counts.sum() * 100
plt.bar(percentages.index, percentages, color=colors)
plt.xlabel('General Health')
plt.ylabel('Percentage')
plt.title('General Health Proportions')
plt.show()

# Age histogram
plt.hist(hints_dataset['age'], bins=20)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Ages')
plt.show()

# Mean of age
# Generate data without NA values
age_dat = hints_dataset.dropna(subset=['age'])
# Calculate mean age
mean_age = sum(age_dat['age']) / len(age_dat['age'])
# Print the result
print("The mean age is:", mean_age)

# Visualize cancer diagnosis by age and smoking status
df = hints_dataset[['everhadcancer', 'age', 'smokenow']]

# Split the data into two groups: those who have had cancer and those who haven't
df_cancer = df[df['everhadcancer'] == 1]
df_no_cancer = df[df['everhadcancer'] == 0]

# Create the scatter plot
plt.scatter(df_cancer['age'], df_cancer['smokenow'], label='Cancer')
plt.scatter(df_no_cancer['age'], df_no_cancer['smokenow'], label='No Cancer')
plt.xlabel('Age')
plt.ylabel('Smoke Now')
plt.title('Relationship between Age and Smokenow by Cancer Status')
plt.legend()
plt.show()

#%%
# Exercise factors EDA
# timesmoderateexercise histogram
plt.hist(hints_dataset['timesmoderateexercise'], bins=10, color='blue', alpha=0.5)
plt.title('Histogram of timesmoderateexercise')
plt.xlabel('Value of timesmoderateexercise')
plt.ylabel('Frequency')
plt.show()
# frequency table
freq_table_timesmoderateexercise = hints_dataset['timesmoderateexercise'].value_counts()
print(freq_table_timesmoderateexercise)
# proportion table
prop_table_timesmoderateexercise = hints_dataset['timesmoderateexercise'].value_counts(normalize=True)
print(prop_table_timesmoderateexercise)

# howlongmoderateexerciseminutes histogram
plt.hist(hints_dataset['howlongmoderateexerciseminutes'], bins=10, color='blue', alpha=0.5)
plt.title('Histogram of howlongmoderateexerciseminutes')
plt.xlabel('Value of howlongmoderateexerciseminutes')
plt.ylabel('Frequency')
plt.show()

# timesstrengthtraining histogram
plt.hist(hints_dataset['timesstrengthtraining'], bins=10, color='blue', alpha=0.5)
plt.title('Histogram of htimesstrengthtraining')
plt.xlabel('Value of timesstrengthtraining')
plt.ylabel('Frequency')
plt.show()
# frequency table
freq_table_timesstrengthtraining = hints_dataset['timesstrengthtraining'].value_counts()
print(freq_table_timesstrengthtraining)
# proportion table
prop_table_timesstrengthtraining = hints_dataset['timesstrengthtraining'].value_counts(normalize=True)
print(prop_table_timesstrengthtraining)

# exrec_notheard histogram
plt.hist(hints_dataset['exrec_notheard'], bins=10, color='blue', alpha=0.5)
plt.title('Histogram of exrec_notheard')
plt.xlabel('Value of exrec_notheard')
plt.ylabel('Frequency')
plt.show()
# frequency table
freq_table_exrec_notheard = hints_dataset['exrec_notheard'].value_counts()
print(freq_table_exrec_notheard)
# proportion table
prop_table_exrec_notheard = hints_dataset['exrec_notheard'].value_counts(normalize=True)
print(prop_table_exrec_notheard)

# create weekly MVPA variable
"""Weekly MVPA = times per week exercised * minutes per time exercised. 
This variable is often generated in this way in the physical acivity literature.
See: https://youthrex.com/wp-content/uploads/2019/10/IPAQ-TM.pdf
"""
hints_dataset['weeklyMVPA'] = hints_dataset['timesmoderateexercise'] * hints_dataset['howlongmoderateexerciseminutes']
# weekly MVPA histogram
plt.hist(hints_dataset['weeklyMVPA'], bins=10, color='blue', alpha=0.5)
plt.title('Histogram of weeklyMVPA')
plt.xlabel('Value of weeklyMVPA')
plt.ylabel('Frequency')
plt.show()

# create Physical Activity Guidlines for American (PAGA) met/not met variable
""" PAGA are met if two requirements are met: 1) weekly MVPA >= 150 minutes AND 
2) strength training at least two days per week.
See: https://health.gov/sites/default/files/2019-09/Physical_Activity_Guidelines_2nd_edition.pdf
"""
hints_dataset['PAGAmet'] = hints_dataset.apply(lambda x: 1 if (x['weeklyMVPA'] >= 150 and x['timesstrengthtraining'] >= 2) else 0, axis=1)
# PAGA met histogram
plt.hist(hints_dataset['PAGAmet'], bins=10, color='blue', alpha=0.5)
plt.title('Histogram of PAGAmet')
plt.xlabel('Value of PAGAmet')
plt.ylabel('Frequency')
plt.show()
# frequency table
freq_table_PAGAmet = hints_dataset['PAGAmet'].value_counts()
print(freq_table_PAGAmet)
# proportion table
prop_table_PAGAmet = hints_dataset['PAGAmet'].value_counts(normalize=True)
print(prop_table_PAGAmet)

#%%
# Sleep variables EDA
# Create histogram of averagesleepnight
bins = 20
fig, ax = plt.subplots()
ax.hist(hints_dataset['averagesleepnight'], bins=bins, color='purple', edgecolor='black', alpha=0.7, label='Sleep Hours')
ax.set_xlabel('Hours of sleep per night')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Average Sleep Hours during the past 7 Nights')
plt.show()

# Remove missing values from averagesleepquality column
data = hints_dataset.dropna(subset=['averagesleepquality'])
# Count the frequency of each unique value in averagesleepquality
sleep_counts = data['averagesleepquality'].value_counts()
# Create a pie chart of the sleep quality counts
labels = ['Fairly good', 'Very good', 'Fairly bad', 'Very bad']
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0']
plt.pie(sleep_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Average Sleep Quality')
plt.axis('equal')
plt.show()

print(sleep_counts)

#%% 
# Diet & alcohol EDA
hints_dataset['fruit'].value_counts().plot(kind='barh')
plt.xlabel('Number of People')
plt.title('Fruits consumed per day')
plt.show()

hints_dataset['vegetables'].value_counts().plot(kind='barh')
plt.xlabel('Number of People')
plt.title('Vegetables consumed per day')
plt.show()

hints_dataset.drinkdaysperweek.hist()
plt.xlabel('Number of drinking days per week')
plt.title('Distribution of number of drinking days per week')
plt.show()

sns.violinplot(x=hints_dataset['drinkdaysperweek'],y=hints_dataset['everhadcancer'])
plt.xlabel('Number of drinking days per week')
plt.title('Cancer status')
plt.show()

sns.violinplot(x=hints_dataset['drinkdaysperweek'],y=hints_dataset['vegetables'])
plt.xlabel('Number of drinking days per week')
plt.ylabel('Cups of vegetables each day')
plt.title('Vegetable eating Habits')
plt.show()

sns.violinplot(x=hints_dataset['drinkdaysperweek'],y=hints_dataset['fruit'])
plt.xlabel('Number of drinkig days per week')
plt.ylabel('Cups of fruits each day')
plt.title('Fruit eating Habits')
plt.show()

#%%
# Other variables EDA
# bmi 
plt.hist(hints_dataset['bmi'], bins=10, color='blue', alpha=0.5)
plt.title('Histogram of bmi')
plt.xlabel('Value of bmi')
plt.ylabel('Frequency')
plt.show()

# smokenow
# Remove missing values from smokenow column
data1 = hints_dataset.dropna(subset=['smokenow'])
# Count the frequency of each category in smokenow
smoke_counts = data1['smokenow'].value_counts()
# Define the labels and colors for the pie chart
labels = ['Everyday', 'Some days', 'Not at all']
colors = ['#ff9999','#66b3ff','#99ff99']
# Create the pie chart
plt.pie(smoke_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Smoking Status')
plt.axis('equal')
plt.show()

print(smoke_counts)

# tanning
# Remove missing values from spendtimeinsuntanning column
data2 = hints_dataset.dropna(subset=['spendtimeinsuntanning'])
# Count the frequency of each value in spendtimeinsuntanning
sun_counts = data2['spendtimeinsuntanning'].value_counts()
# Define the labels and colors for the pie chart
labels = ['Never','Rarely', 'Sometimes','Often','Donot go out on sunny days']
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#b3b3cc']
# Create the pie chart
plt.pie(sun_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.legend(title='Time Spent in Sun Tanning')
plt.title('Time Spent in Sun Tanning')
plt.axis('equal')
plt.show()

print(sun_counts)

# cancer
# Remove missing values 
data3 = hints_dataset.dropna(subset=['everhadcancer'])
# Count the frequency of each value 
cancer_counts = data2['everhadcancer'].value_counts()
# Define the labels and colors for the pie chart
labels = ['Never had cancer','Had cancer']
# Create the pie chart
plt.pie(cancer_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.legend(title='Cancer diagnosis status')
plt.title('Ever had a cnacer diagnosis')
plt.axis('equal')
plt.show()

print(sun_counts)

# ever had cancer & sleep hours
# Remove missing values from everhadcancer and averagesleepnight columns
data3 = hints_dataset.dropna(subset=['everhadcancer', 'averagesleepnight'])
# Create a count plot of everhadcancer vs averagesleepnight
plot = sns.countplot(x='everhadcancer', hue='averagesleepnight', data=data3, palette='Blues')
plot.legend(title='Sleep Quality', loc='center')
plot.set(title='Ever Had Cancer vs. Average Sleep ', xlabel='Ever Had Cancer', ylabel='Count')
plot.set_xticklabels(['No Cancer','Had Cancer'])
plt.show()

#%%[markdown]
###################
# SMART QUESTIONS #
###################

#%%
# 1.	Is cancer diagnosis dtatus associated with fruit consumption? - Not statistically significant
# Remove missing values 
data1 = hints_dataset.dropna(subset=['fruit', 'everhadcancer'])

# Violin plot
sns.violinplot(x="everhadcancer", y="fruit", data=data1, color='#99c2a2')
plt.xlabel("Ever had cancer")
plt.ylabel("Cups of Fruit")
plt.title("Cancer diagnosis by fruit (cups) consumption")
plt.show()

# Chi-squared test
SQ1_ct = pd.crosstab(data1["fruit"], data1["everhadcancer"])
chi2, p, dof, expected = chi2_contingency(SQ1_ct)
print(p)

#%%
# 2.	Is cancer diagnosis status associated vegetable consumption? - Not statistically significant
# Remove missing values 
data2 = hints_dataset.dropna(subset=['vegetables', 'everhadcancer'])

# Violin plot
sns.violinplot(x="everhadcancer", y="vegetables", data=data2, color='#99c2a2')
plt.xlabel("Ever had cancer")
plt.ylabel("Cups of Vegetables")
plt.title("Cancer diagnosis by vegetables (cups) consumption")
plt.show()

# Chi-squared test
SQ2_ct = pd.crosstab(data2["vegetables"], data2["everhadcancer"])
chi2, p, dof, expected = chi2_contingency(SQ2_ct)
print(p)

#%%
# 3.	Is cancer diagnosis status associated with weekly minutes of exercise (of at least moderate intensity)? - Not statistically significant
# Remove missing values 
data3 = hints_dataset.dropna(subset=['weeklyMVPA', 'everhadcancer'])

# Violin plot
sns.violinplot(x="everhadcancer", y="weeklyMVPA", data=data3)
plt.xlabel("Ever had cancer")
plt.ylabel("Weekly MVPA")
plt.title("Cancer diagnosis by weekly MVPA")
plt.show()

#t-test
weeklyMVPA_tt = ttest_ind(data3[data3["everhadcancer"] == 1]["weeklyMVPA"], 
                          data3[data3["everhadcancer"] == 0]["weeklyMVPA"])
print(weeklyMVPA_tt)

#%%
# 4.	Is cancer diagnosis status associated with meeting the Physical Activity Guidelines for Americans? - Statistically, yes
# Remove missing values 
data4 = hints_dataset.dropna(subset=['PAGAmet', 'everhadcancer'])

# Count plot
plot = sns.countplot(x='everhadcancer', hue='PAGAmet', data=data4, palette='Blues')
plot.legend(title='PAGA', loc='upper right', labels=['No','Yes'])
plot.set(title='Cancer diagnosis by meeting PAGA', xlabel='Ever Had Cancer', ylabel='Count')
plot.set_xticklabels(['No Cancer','Had Cancer'])
plt.show()

# view proportions
cancer_diagnosis = hints_dataset[hints_dataset['everhadcancer'] == 1]
paga_met_prop = cancer_diagnosis['PAGAmet'].mean()
print(f"The proportion of individuals diagnosed with cancer who met physical activity guidelines is: {paga_met_prop:.2f}")

no_cancer_diagnosis = hints_dataset[hints_dataset['everhadcancer'] == 0]
paga_met_prop2 = no_cancer_diagnosis['PAGAmet'].mean()
print(f"The proportion of individuals not diagnosed with cancer who met physical activity guidelines is: {paga_met_prop2:.2f}")

# Chi-squared test
PAGAmet_ct = pd.crosstab(data4["PAGAmet"], data4["everhadcancer"])
chi2, p, dof, expected = chi2_contingency(PAGAmet_ct)
print(p)

#%%
# 5.	Is cancer diagnosis status associated with hours per night of sleep? - Not statistically significant
# Remove missing values 
data5 = hints_dataset.dropna(subset=['averagesleepnight', 'everhadcancer'])

# Violin plot
sns.violinplot(x="everhadcancer", y="averagesleepnight", data=data5)
plt.xlabel("Ever had cancer")
plt.ylabel("Hours of sleep per night")
plt.title("Cancer diagnosis by hours of sleep per night")
plt.show()

#t-test
sleephours_tt = ttest_ind(data5[data5["everhadcancer"] == 1]["averagesleepnight"], 
                          data5[data5["everhadcancer"] == 0]["averagesleepnight"])
print(sleephours_tt)

#%%
# 6.	Is cancer diagnosis status associated with sleep quality? - Statistically significant
# Remove missing values 
data6 = hints_dataset.dropna(subset=['everhadcancer', 'averagesleepquality'])

# Count plot of everhadcancer vs averagesleepquality
plot = sns.countplot(x='everhadcancer', hue='averagesleepquality', data=data6, palette='Blues')
plot.legend(title='Sleep Quality', loc='upper right', labels=['Very good','Fairly good','Fairly bad','Very bad'])
plot.set(title='Cancer diagnosis by sleep quality', xlabel='Ever Had Cancer', ylabel='Count')
plot.set_xticklabels(['No Cancer','Had Cancer'])
plt.show()

# Chi-squared test
sleepq_ct = pd.crosstab(data6["averagesleepquality"], data6["everhadcancer"])
chi2, p, dof, expected = chi2_contingency(sleepq_ct)
print(p)

#%%
# 7.	Is cancer diagnosis status associated with number of drinking days per week? - Not statistically significant
# Remove missing values 
data7 = hints_dataset.dropna(subset=['drinkdaysperweek', 'everhadcancer'])

# Violin plot
sns.violinplot(x="everhadcancer", y="drinkdaysperweek", data=data7)
plt.xlabel("Ever had cancer")
plt.ylabel("Drinking days per week")
plt.title("Cancer diagnosis by drinking days per week")
plt.show()

# Subset the data to include only individuals with a cancer diagnosis
data7_subset = data7[data7['everhadcancer'] == 1]
# Calculate the mean and median number of drinking days among individuals with a cancer diagnosis
mean_drinking_days = np.mean(data7_subset['drinkdaysperweek'])
median_drinking_days = np.median(data7_subset['drinkdaysperweek'])
# Print the result
print("Mean drinking days among those with cancer diagnosis:", mean_drinking_days)
print("Median drinking days among those with cancer diagnosis:", median_drinking_days)

# Subset the data to include only individuals without a cancer diagnosis
data7_subset2 = data7[data7['everhadcancer'] == 0]
# Calculate the mean number of drinking days among individuals with a cancer diagnosis
mean_drinking_days2 = np.mean(data7_subset2['drinkdaysperweek'])
median_drinking_days2 = np.median(data7_subset2['drinkdaysperweek'])
# Print the result
print("Mean drinking days among those without cancer diagnosis:", mean_drinking_days2)
print("Median drinking days among those without cancer diagnosis:", median_drinking_days2)

#t-test
drinks_tt = ttest_ind(data7[data7["everhadcancer"] == 1]["drinkdaysperweek"], 
                   data7[data7["everhadcancer"] == 0]["drinkdaysperweek"])
print(drinks_tt)

#%%
# 8.	Is cancer diagnosis status associated with current tobacco use? - Statistically significant
# Remove missing values 
data8 = hints_dataset.dropna(subset=['everhadcancer', 'smokenow'])

# Count plot of everhadcancer vs averagesleepquality
plot = sns.countplot(x='everhadcancer', hue='smokenow', data=data8, palette='Blues')
plot.legend(title='Smoking status', loc='upper right', labels=['Every day','Some days','Never'])
plot.set(title='Cancer diagnosis by smoking status', xlabel='Ever Had Cancer', ylabel='Count')
plot.set_xticklabels(['No Cancer','Had Cancer'])
plt.show()

# view proportions
# Filter the dataset to only include individuals with a cancer diagnosis
cancer_diagnosis = data8[data8['everhadcancer'] == 1]
# Compute the percentage of individuals with each smoking status among those with a cancer diagnosis
smoking_status_counts = cancer_diagnosis['smokenow'].value_counts(normalize=True) * 100
# Print the results
print("Percentage of individuals with each smoking status among those with a cancer diagnosis:")
print(smoking_status_counts)

# Filter the dataset to only include individuals without a cancer diagnosis
nocancer_diagnosis = data8[data8['everhadcancer'] == 0]
# Compute the percentage of individuals with each smoking status among those with a cancer diagnosis
smoking_status_counts2 = nocancer_diagnosis['smokenow'].value_counts(normalize=True) * 100
# Print the results
print("Percentage of individuals with each smoking status among those without a cancer diagnosis:")
print(smoking_status_counts2)

# Chi-Squared test
smoke_ct = pd.crosstab(data8["smokenow"], data8["everhadcancer"])
chi2, p, dof, expected = chi2_contingency(smoke_ct)
print(p)

#%%
# 9.	Is cancer diagnosis status associated with BMI? - Not statistically significant
# Remove missing values 
data9 = hints_dataset.dropna(subset=['bmi', 'everhadcancer'])

# Violin plot
sns.violinplot(x="everhadcancer", y="bmi", data=data9)
plt.xlabel("Ever had cancer")
plt.ylabel("BMI")
plt.title("Cancer diagnosis by BMI")
plt.show()

#t-test
bmi_tt = ttest_ind(data9[data9["everhadcancer"] == 1]["bmi"], 
                   data9[data9["everhadcancer"] == 0]["bmi"])
print(bmi_tt)

#%%
# 10.	Is cancer diagnosis status associated with sun tanning behavior? - Statistically significant
# Remove missing values 
data10 = hints_dataset.dropna(subset=['bmi', 'everhadcancer'])

# Violin plot
sns.violinplot(x="everhadcancer", y="spendtimeinsuntanning", data=data10)
plt.xlabel("Ever had cancer")
plt.ylabel("Time spent tanning")
plt.title("Cancer diagnosis by tanning behavior")
plt.show()

# Count plot
plot = sns.countplot(x='everhadcancer', hue='spendtimeinsuntanning', data=data10, palette='Blues')
plot.legend(title='Tanning', loc='upper right', labels=['Often','Sometimes','Rarely','Never',"Don't go out in sun"])
plot.set(title='Cancer diagnosis by tanning behavior', xlabel='Ever Had Cancer', ylabel='Count')
plot.set_xticklabels(['No Cancer','Had Cancer'])
plt.show()

# view proportions
# Filter the dataset to only include individuals with a cancer diagnosis
cancer_diagnosis = data10[data10['everhadcancer'] == 1]
# Compute the percentage of individuals with each smoking status among those with a cancer diagnosis
tan_status_counts = cancer_diagnosis['spendtimeinsuntanning'].value_counts(normalize=True) * 100
# Print the results
print("Percentage of individuals with each tanning status among those with a cancer diagnosis:")
print(tan_status_counts)

# Filter the dataset to only include individuals with a cancer diagnosis
nocancer_diagnosis = data10[data10['everhadcancer'] == 0]
# Compute the percentage of individuals with each smoking status among those with a cancer diagnosis
tan_status_counts2 = nocancer_diagnosis['spendtimeinsuntanning'].value_counts(normalize=True) * 100
# Print the results
print("Percentage of individuals with each tanning status among those without a cancer diagnosis:")
print(tan_status_counts2)

# Chi-Squared test
tan_ct = pd.crosstab(data10["spendtimeinsuntanning"], data10["everhadcancer"])
chi2, p, dof, expected = chi2_contingency(tan_ct)
print(p)

#%%[markdown]
#########################################
# Cancer diagnosis classification model #
#########################################

#%%
# Create dummy variables for categorical predictors
hints_dummy = pd.get_dummies(hints_dataset, columns=['raceethn', 'generalhealth', 'genderc', 'vegetables','fruit','influencecancer_eatingfruitveg','PAGAmet','averagesleepquality','smokenow','spendtimeinsuntanning'], drop_first=True)

rfit.dfchk(hints_dummy)

#%%
from statsmodels.formula.api import ols

# get dummy varaiables
hints_dummy = pd.get_dummies(hints_dataset, columns=['genderc','generalhealth','raceethn', 'averagesleepquality', 'fruit', 'vegetables', 'smokenow', 'spendtimeinsuntanning'], drop_first=True)

# clean dummy variable names
hints_dummy.rename(columns={'genderc_2.0': 'female'}, inplace=True)
hints_dummy.rename(columns={'averagesleepquality_2.0': 'averagesleepquality2'}, inplace=True)
hints_dummy.rename(columns={'averagesleepquality_3.0': 'averagesleepquality3'}, inplace=True)
hints_dummy.rename(columns={'averagesleepquality_4.0': 'averagesleepquality4'}, inplace=True)
hints_dummy.rename(columns={'fruit_0-0.5': 'fruit1'}, inplace=True)
hints_dummy.rename(columns={'fruit_0.5-1': 'fruit2'}, inplace=True)
hints_dummy.rename(columns={'fruit_1-2': 'fruit3'}, inplace=True)
hints_dummy.rename(columns={'fruit_2-3': 'fruit4'}, inplace=True)
hints_dummy.rename(columns={'fruit_3-4': 'fruit5'}, inplace=True)
hints_dummy.rename(columns={'fruit_4 or more': 'fruit6'}, inplace=True)
hints_dummy.rename(columns={'vegetables_0-0.5': 'veg1'}, inplace=True)
hints_dummy.rename(columns={'vegetables_0.5-1': 'veg2'}, inplace=True)
hints_dummy.rename(columns={'vegetables_1-2': 'veg3'}, inplace=True)
hints_dummy.rename(columns={'vegetables_2-3': 'veg4'}, inplace=True)
hints_dummy.rename(columns={'vegetables_3-4': 'veg5'}, inplace=True)
hints_dummy.rename(columns={'vegetables_4 or more': 'veg6'}, inplace=True)
hints_dummy.rename(columns={'smokenow_2.0': 'smokenow2'}, inplace=True)
hints_dummy.rename(columns={'smokenow_3.0': 'smokenow3'}, inplace=True)
hints_dummy.rename(columns={'spendtimeinsuntanning_2.0': 'spendtimeinsuntanning2'}, inplace=True)
hints_dummy.rename(columns={'spendtimeinsuntanning_3.0': 'spendtimeinsuntanning3'}, inplace=True)
hints_dummy.rename(columns={'aspendtimeinsuntanning_4.0': 'spendtimeinsuntanning4'}, inplace=True)
hints_dummy.rename(columns={'spendtimeinsuntanning_4.0': 'spendtimeinsuntanning4'}, inplace=True)
hints_dummy.rename(columns={'spendtimeinsuntanning_5.0': 'spendtimeinsuntanning5'}, inplace=True)

rfit.dfchk(hints_dummy)

#%%
# Select variables for logistic regression and create new dataframe
hints_logistic = hints_dummy[['everhadcancer', 'female', 'raceethn_Black', 'raceethn_Hispanic', 'raceethn_Asian', \
                              'raceethn_Other', 'generalhealth_Very_Good', 'generalhealth_Good', 'generalhealth_Fair', \
                                'generalhealth_Poor', 'age', 'fruit2', 'fruit3', 'fruit4', 'fruit5', 'fruit6', 'veg2', \
                                'veg3', 'veg4', 'veg5', 'veg6', 'smokenow2', 'smokenow3', 'weeklyMVPA', 'PAGAmet',\
                                'averagesleepnight', 'averagesleepquality2', 'averagesleepquality3', 'averagesleepquality4',\
                                'drinkdaysperweek', 'bmi', 'spendtimeinsuntanning2', 'spendtimeinsuntanning3', \
                                'spendtimeinsuntanning4', 'spendtimeinsuntanning5']]

# Remove rows with missing values
hints_logistic = hints_logistic.dropna()

# define the predictor variables
X = hints_logistic[['female', 'raceethn_Black', 'raceethn_Hispanic', 'raceethn_Asian', \
                              'raceethn_Other', 'generalhealth_Very_Good', 'generalhealth_Good', 'generalhealth_Fair', \
                                'generalhealth_Poor', 'age', 'fruit2', 'fruit3', 'fruit4', 'fruit5', 'fruit6', 'veg2', \
                                'veg3', 'veg4', 'veg5', 'veg6', 'smokenow2', 'smokenow3', 'weeklyMVPA', 'PAGAmet',\
                                'averagesleepnight', 'averagesleepquality2', 'averagesleepquality3', 'averagesleepquality4',\
                                'drinkdaysperweek', 'bmi', 'spendtimeinsuntanning2', 'spendtimeinsuntanning3', \
                                'spendtimeinsuntanning4', 'spendtimeinsuntanning5']]

# define the outcome variable
y = hints_logistic['everhadcancer']

# fit the logistic regression model
# Model 1
model = sm.Logit(y, X)
modelFit = model.fit()

# print the summary of the model
print(modelFit.summary())

#%%
# drop nonsignificant predictors
X2 = hints_logistic[['female', 'raceethn_Black', 'raceethn_Hispanic', 'raceethn_Asian', \
                              'raceethn_Other', 'generalhealth_Very_Good', 'generalhealth_Good', 'generalhealth_Fair', \
                                'generalhealth_Poor', 'age', 'fruit2', 'fruit3', 'fruit4', 'fruit5', 'fruit6', 'veg2', \
                                'veg3', 'veg4', 'veg5', 'veg6', 'smokenow2', 'smokenow3', 'weeklyMVPA', \
                                'averagesleepnight', 'averagesleepquality2', 'averagesleepquality3', 'averagesleepquality4',\
                                'bmi', 'spendtimeinsuntanning2', 'spendtimeinsuntanning3', \
                                'spendtimeinsuntanning4', 'spendtimeinsuntanning5']]
y2 = hints_logistic['everhadcancer']

# Model 2
model2 = sm.Logit(y2, X2)
model2Fit = model2.fit()

# print the summary of the model
print(model2Fit.summary())

#%%
# Get odds ratios and 95% confidence intervals for Model 2 (final model)
odds_ratios = pd.DataFrame({'OR': model2Fit.params.apply(lambda x: round(np.exp(x), 3))})
conf_int = model2Fit.conf_int(alpha=0.05)
conf_int.columns = ['2.5%', '97.5%']
odds_ratios = pd.concat([odds_ratios, conf_int.apply(lambda x: round(np.exp(x), 3))], axis=1)
print(odds_ratios)


# %%
# final model evaluation 
from sklearn.metrics import accuracy_score

# Generate predicted values
y_pred = (model2Fit.predict(X2) > 0.5).astype(int)

# Compute accuracy
accuracy = accuracy_score(y2, y_pred)

# Print accuracy
print("Accuracy:", accuracy)

# %%
# evaluae the final model when it is split and trained
# train the model with 20% of data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.20, random_state=1)

# Create a logistic regression model
logreg = LogisticRegression()
# Fit the model to the training data
logreg.fit(X_train, y_train)
# Make predictions on the test data
y_pred = logreg.predict(X_test)
# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy score:", accuracy)

# %%