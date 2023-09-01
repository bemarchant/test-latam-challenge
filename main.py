import numpy as np  
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import sklearn as sk

df = pd.read_csv('dataset_SCL.csv')

#Data 
print(df.head(5))
print(df.info())

#check for missing values
print(df.isna().sum())

sns.countplot(x=df["OPERA"])
plt.show()

# Fecha-I: Scheduled date and time of the flight.
# Vlo-I : Scheduled flight number.
# Ori-I : Programmed origin city code.
# Des-I : Programmed destination city code.
# Emp-I : Scheduled flight airline code.
# Fecha-O : Date and time of flight operation.
# Vlo-O : Flight operation number of the flight.Ori-O : Operation origin city code
# Des-O : Operation destination city code.
# Emp-O : Airline code of the operated flight.
# DIA: Day of the month of flight operation.
# MES : Number of the month of operation of the flight.
# AÑO : Year of flight operation.
# DIANOM : Day of the week of flight operation.
# TIPOVUELO : Type of flight, I =International, N =National.
# OPERA : Name of the airline that operates.
# SIGLAORI: Name city of origin.
# SIGLADES: Destination city name.
