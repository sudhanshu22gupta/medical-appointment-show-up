import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class EDA_medical_appointment:

    def __init__(self, df_medical_appointment_features, target_labels) -> None:
        """
         Initialize the class.
         
         @param df_medical_appointment_features - pandas DataFrame with the features
         @param target_labels - np.array of target labels 
        """
        self.df_medical_appointment = df_medical_appointment_features
        self.target_labels = target_labels
        self.numeric_feature_columns = ['Age']
        self.categorical_feature_columns = ['Gender', 'Neighbourhood', 'Scholarship', 'Hypertension', 'Diabetes', 'Alcoholism', 'Handicap', 'SMS_received']

    def describe_general_stats(self):
        """
         Describe general stats and visualize them
        """
        print("-"*50)
        print("Checking feature datatypes and null values")
        print("-"*50, end='\n\n')
        display(self.df_medical_appointment.info())
        print("-"*50)
        print("Describe continuous numerical features")
        print("-"*50)
        display(self.df_medical_appointment[self.numeric_feature_columns].describe())
        self.df_medical_appointment[self.numeric_feature_columns].plot(kind='hist')
        plt.show()
        print("-"*50)
        print("categorical features Probabilities")
        print("-"*50)
        # Plot Independent Probability of Categorical Variable
        fig, axs = plt.subplots(figsize=(len(self.categorical_feature_columns)*2, 5), ncols=len(self.categorical_feature_columns), sharey=True)
        for cat_var, ax in zip(self.categorical_feature_columns, axs):
            # taking Most Frequent 10 Neighbourhoods to make the plot readable
            if cat_var=='Neighbourhood':
                cat_var_prob = self.df_medical_appointment[cat_var].value_counts().iloc[:10] / len(self.df_medical_appointment)
                cat_var = "Most Frequent 10 Neighbourhoods"
            else:
                cat_var_prob = self.df_medical_appointment[cat_var].value_counts() / len(self.df_medical_appointment)
            cat_var_prob.plot(kind="bar", ax=ax)
            ax.set_xlabel(cat_var)
            ax.set_ylabel("Probability")
            ax.set_ylim(0, 1)
            ax.grid(axis='y')
        plt.suptitle(f"Independent Probability of Categorical Variable")
        plt.tight_layout()
        plt.show()

    def stats_per_unique_patient(self):
        """
         Plot stats of visits per unique patient for each patient.
        """
        df_medical_appointment_groupby_patient = self.df_medical_appointment.groupby('PatientID')
        df_medical_appointment_groupby_patient.count().sort_values(by=['AppointmentID'], ascending=False)['AppointmentID'].plot(kind='hist', density=True, bins=range(100))
        plt.xlabel('N Visits per unique patient')
        plt.title('Density Histogram of visits per unique patient')
        plt.show()
        n_patients_less_than_3_visits = len(df_medical_appointment_groupby_patient.count().loc[df_medical_appointment_groupby_patient.count()['AppointmentID']<3])
        df_patient_last_appointment = df_medical_appointment_groupby_patient.last()

        print("-"*50)
        print("-"*50)
        print(f"n_unique_patients = {len(df_patient_last_appointment)}")
        print(f"n_patients_less_than_3_visits = {n_patients_less_than_3_visits}")
        print(f"percent_patients_less_than_3_visits = {round(100*n_patients_less_than_3_visits/len(df_medical_appointment_groupby_patient.count()))}%")
        print("-"*50)
        print("-"*50)

        fig, axs = plt.subplots(figsize=(len(self.categorical_feature_columns)*2, 5), ncols=len(self.categorical_feature_columns), sharey=True)
        for cat_var, ax in zip(self.categorical_feature_columns, axs):
            # taking Most Frequent 10 Neighbourhoods to make the plot readable
            if cat_var=='Neighbourhood':
                cat_var_counts = df_patient_last_appointment[cat_var].value_counts().iloc[:10] / len(df_patient_last_appointment)
                cat_var = "Most Frequent 10 Neighbourhoods"
            else:
                cat_var_counts = df_patient_last_appointment[cat_var].value_counts() / len(df_patient_last_appointment)
            cat_var_counts.plot(kind="bar", ax=ax)
            ax.set_xlabel(cat_var)
            ax.set_ylabel("Probability")
            ax.set_ylim(0, 1)
            ax.grid(axis='y')
        plt.suptitle(f"Independent Probability of Categorical Variable of Unique Patients")
        plt.tight_layout()
        plt.show()


    def visulaize_class_distribution(self):
        """
         Visualize the class distribution of the target data.
        """
        pd.Series(self.target_labels).value_counts().plot(kind="bar")
        plt.title("Class Distribution\n(0=Show, 1=No Show)")
        plt.ylabel("Count")
        plt.show()

    def visulaize_no_show_prob_per_variable(self):
        """
         Visualize the Probability of Feature given No-Show, i.e., P(Feature|no_show)
        """
        vars_categorical = self.categorical_feature_columns[:]
        vars_categorical.remove("Neighbourhood")
        fig, axs = plt.subplots(figsize=(len(vars_categorical)*2, 5), ncols=len(vars_categorical), sharey=True)
        df_no_show = self.df_medical_appointment.loc[self.target_labels==1]
        # Plots the conditional probability of each categorical variable.
        for cat_var, ax in zip(vars_categorical, axs):
            cat_var_prob_no_show = df_no_show[cat_var].value_counts() / len(df_no_show)
            cat_var_prob_no_show.plot(kind="bar", ax=ax)
            ax.set_xlabel(cat_var)
            ax.set_ylabel("probability")
            ax.set_ylim(0, 1)
            ax.grid(axis='y')
        plt.suptitle(f"Conditional Probability of Feature given No-Show\nP(Feature|no_show)")
        plt.tight_layout()
        plt.show()

        