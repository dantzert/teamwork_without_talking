import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

# load in the total rainfall data for summer of 2020 located at "C:\teamwork_without_talking\summer_2020_rain.csv"
# parse the column "period" as a datetime index
rainfall_2020 = pd.read_csv("C:\\teamwork_without_talking\\summer_2020_rain.csv", parse_dates=['period'])

# resample rainfall_2020 to be in 15 minute intervals
rainfall_2020 = rainfall_2020.set_index('period').resample('15T').asfreq()
# format the index as "MM/DD/YYYY HH:MM""
rainfall_2020.index = rainfall_2020.index.strftime("%m/%d/%Y %H:%M")

#fill na's with 0
rainfall_2020 = rainfall_2020.fillna(0)

print(rainfall_2020)

# save each column to a new .dat file in the same directory as the original data
rainfall_2020['C'].to_csv("C:\\teamwork_without_talking\\raingage_C_summer_2020.dat", index=True,header=False,sep='\t')
rainfall_2020['J'].to_csv("C:\\teamwork_without_talking\\raingage_J_summer_2020.dat", index=True,header=False,sep='\t')
rainfall_2020['N'].to_csv("C:\\teamwork_without_talking\\raingage_N_summer_2020.dat", index=True,header=False,sep='\t')
rainfall_2020['S'].to_csv("C:\\teamwork_without_talking\\raingage_S_summer_2020.dat", index=True,header=False,sep='\t')

# do the exact same thing as above, but substitute "2020" with "2021"
rainfall_2021 = pd.read_csv("C:\\teamwork_without_talking\\summer_2021_rain.csv", parse_dates=['period'])
rainfall_2021 = rainfall_2021.set_index('period').resample('15T').asfreq()
rainfall_2021.index = rainfall_2021.index.strftime("%m/%d/%Y %H:%M")
rainfall_2021 = rainfall_2021.fillna(0)

rainfall_2021['C'].to_csv("C:\\teamwork_without_talking\\raingage_C_summer_2021.dat", index=True,header=False,sep='\t')
rainfall_2021['J'].to_csv("C:\\teamwork_without_talking\\raingage_J_summer_2021.dat", index=True,header=False,sep='\t')
rainfall_2021['N'].to_csv("C:\\teamwork_without_talking\\raingage_N_summer_2021.dat", index=True,header=False,sep='\t')
rainfall_2021['S'].to_csv("C:\\teamwork_without_talking\\raingage_S_summer_2021.dat", index=True,header=False,sep='\t')







