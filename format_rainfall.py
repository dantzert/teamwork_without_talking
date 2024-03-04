import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import meteostat
import datetime

# load in the total rainfall data for summer of 2020 located at "C:\teamwork_without_talking\summer_2020_rain.csv"
# parse the column "period" as a datetime index
rainfall_2020 = pd.read_csv("C:\\teamwork_without_talking\\summer_2020_rain.csv", parse_dates=['period'])
# resample rainfall_2020 to be in 5 minute intervals
rainfall_2020 = rainfall_2020.set_index('period').resample('5T').asfreq()

original_depths = rainfall_2020.sum()

# convert accumluations to intensities by multiplying by 4. originally in 15 minute accumulations
rainfall_2020 = rainfall_2020 * 4
# now in inches / hour rainfall intensity (make sure swmm model reflects this)


# for any nonzero entry at a time ending with :15 :30 :45 or :00, redistribute that rainfall intensity over the previous 3 intervals 
for moment in rainfall_2020.index:
    if any(rainfall_2020.loc[moment] != 0) and  (moment.minute == 15 or moment.minute == 30 or moment.minute == 45 or moment.minute == 0):
        rainfall_2020.loc[moment - datetime.timedelta(minutes=5)] = rainfall_2020.loc[moment]
        rainfall_2020.loc[moment - datetime.timedelta(minutes=10)] = rainfall_2020.loc[moment]
        rainfall_2020.loc[moment - datetime.timedelta(minutes=15)] = rainfall_2020.loc[moment]
        rainfall_2020.loc[moment] = 0

#fill na's with 0
rainfall_2020 = rainfall_2020.fillna(0)

# convert from inches per hour to inches accumulated in each 5 minute interval
new_depths = rainfall_2020.sum()/12
# compare original and new depths after reformatting (should match)
print(original_depths)
print(new_depths)


# format the index as "MM/DD/YYYY HH:MM""
rainfall_2020.index = rainfall_2020.index.strftime("%m/%d/%Y %H:%M")

print(rainfall_2020)

# save each column to a new .dat file in the same directory as the original data
rainfall_2020['C'].to_csv("C:\\teamwork_without_talking\\raingage_C_summer_2020.dat", index=True,header=False,sep='\t')
rainfall_2020['J'].to_csv("C:\\teamwork_without_talking\\raingage_J_summer_2020.dat", index=True,header=False,sep='\t')
rainfall_2020['N'].to_csv("C:\\teamwork_without_talking\\raingage_N_summer_2020.dat", index=True,header=False,sep='\t')
rainfall_2020['S'].to_csv("C:\\teamwork_without_talking\\raingage_S_summer_2020.dat", index=True,header=False,sep='\t')

# do the same thing as above, but replace "2020" with "2021" 
rainfall_2021 = pd.read_csv("C:\\teamwork_without_talking\\summer_2021_rain.csv", parse_dates=['period'])
rainfall_2021 = rainfall_2021.set_index('period').resample('5T').asfreq()
og_depths = rainfall_2021.sum()
rainfall_2021 = rainfall_2021 * 4
for moment in rainfall_2021.index:
    if any(rainfall_2021.loc[moment] != 0) and  (moment.minute == 15 or moment.minute == 30 or moment.minute == 45 or moment.minute == 0):
        rainfall_2021.loc[moment - datetime.timedelta(minutes=5)] = rainfall_2021.loc[moment]
        rainfall_2021.loc[moment - datetime.timedelta(minutes=10)] = rainfall_2021.loc[moment]
        rainfall_2021.loc[moment - datetime.timedelta(minutes=15)] = rainfall_2021.loc[moment]
        rainfall_2021.loc[moment] = 0

rainfall_2021 = rainfall_2021.fillna(0)
rainfall_2021.index = rainfall_2021.index.strftime("%m/%d/%Y %H:%M")
rainfall_2021['C'].to_csv("C:\\teamwork_without_talking\\raingage_C_summer_2021.dat", index=True,header=False,sep='\t')
rainfall_2021['J'].to_csv("C:\\teamwork_without_talking\\raingage_J_summer_2021.dat", index=True,header=False,sep='\t')
rainfall_2021['N'].to_csv("C:\\teamwork_without_talking\\raingage_N_summer_2021.dat", index=True,header=False,sep='\t')
rainfall_2021['S'].to_csv("C:\\teamwork_without_talking\\raingage_S_summer_2021.dat", index=True,header=False,sep='\t')

new_depths = rainfall_2021.sum()/12

print(og_depths)
print(new_depths)
