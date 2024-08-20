import matplotlib
from matplotlib.lines import lineStyles
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime


year = '2020' # '2020' or '2021'
control_scenarios = ['hi-fi','lo-fi','local']
#control_scenarios = ['hi-fi']
packet_loss_chances = [0.0,0.2,0.5,0.8,0.9,0.95,0.98,0.99,0.999,0.9993,0.9995,0.9997,0.9999]
scores = pd.DataFrame(index=control_scenarios, columns=packet_loss_chances)

for control_scenario in control_scenarios:
    for packet_loss in packet_loss_chances:
        # read in the corresponding cost file
        df = pd.read_csv("C:/teamwork_without_talking/results/"+str(control_scenario) + "_" + str(packet_loss) + "_summer_"+str(year) +"_costs.csv")
        scores.at[control_scenario, packet_loss] = df[' total cost'][0]
        # try a higher penalty on TSS
        #scores.at[control_scenario, packet_loss] = df['flood cost'][0] + df[' flow cost'][0] + 1000000*df[' TSS loading (kg)'][0]
    
print(scores)

# make a list called "expected_report_frequency" which is 5 minutes divided by (1-packet_loss_chance)
expected_report_frequency = [5/(1-p) for p in packet_loss_chances]

# go through expected_report_frequency and express the time elapsed in one time unit (minutes, hours, or days)
# take the value to one decimal place. use "m" for minutes, "h" for hours, and "d" for days
for i in range(len(expected_report_frequency)):
    if expected_report_frequency[i] < 60:
        expected_report_frequency[i] = str(round(expected_report_frequency[i],1)) + " min"
    elif expected_report_frequency[i] < 1440:
        expected_report_frequency[i] = str(round(expected_report_frequency[i]/60,1)) + " hours"
    else:
        expected_report_frequency[i] = str(round(expected_report_frequency[i]/1440,1)) + " days"

# load in the cost for the centralized scenario
centralized_cost = pd.read_csv(str("C:/teamwork_without_talking/results/centralized_0.0_summer_"+str(year)+"_costs.csv"))[' total cost'][0]
uncontrolled_cost = pd.read_csv(str("C:/teamwork_without_talking/results/uncontrolled_0.0_summer_"+str(year)+"_costs.csv"))[' total cost'][0]
# try a higher penalty on TSS 
#centralized_cost = pd.read_csv(str("C:/teamwork_without_talking/results/centralized_0.0_summer_"+str(year)+"_costs.csv"))['flood cost'][0] + pd.read_csv(str("C:/teamwork_without_talking/results/centralized_0.0_summer_"+str(year)+"_costs.csv"))[' flow cost'][0]  + 1000000*pd.read_csv(str("C:/teamwork_without_talking/results/centralized_0.0_summer_"+str(year)+"_costs.csv"))[' TSS loading (kg)'][0]
print(centralized_cost)
print(uncontrolled_cost)

# plot the scores with cost as the y axis and expected report frequency as the x axis
l_width = 7
fig, ax = plt.subplots(figsize=(10,7.5)) # 12x9 for paper figure, smaller for abstract (relatively bigger text)
ax.plot(expected_report_frequency, 1 - scores.loc['hi-fi']/uncontrolled_cost, label='high fidelity', linestyle='dashed',color='blue', marker='o', linewidth=l_width, markersize = 3*l_width)
ax.plot(expected_report_frequency, 1 - scores.loc['lo-fi']/uncontrolled_cost, label='low fidelity', linestyle='dotted',color='red', marker='o', linewidth=l_width, markersize = 3*l_width)
ax.plot(expected_report_frequency, 1 - scores.loc['local']/uncontrolled_cost, label='local', linestyle='dashdot', color='green',marker='o', linewidth=l_width, markersize = 3*l_width)
ax.axhline(y=1 - centralized_cost/uncontrolled_cost, linestyle='solid', label='centralized',color='black', linewidth=l_width)
#ax.axhline(y=uncontrolled_cost, linestyle='dashed', label='uncontrolled',color='black', linewidth=l_width)
ax.set_xlabel('Expected Report Frequency',fontsize='xx-large')
ax.set_ylabel('Cost\nReduction\nfrom\nUncontrolled',fontsize='xx-large',labelpad = -120)
# rotate the y label
ax.yaxis.label.set_rotation(0)
ax.legend(loc='lower left',fontsize='xx-large')
# make the y ticks xxL and the x ticks xL
plt.yticks(fontsize='xx-large')
plt.xticks(fontsize='x-large')
plt.xticks(rotation=45)
# format the y ticks as percents
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
# set ymin to 0
plt.ylim(bottom=0)
#add a vertical line at the 5.0 days label
ax.axvline(x=9, linestyle='dotted', color='black', linewidth=l_width)
plt.tight_layout()
# save the plot
plt.savefig("C:/teamwork_without_talking/results/expected_report_frequency_vs_cost_"+str(year)+".png",dpi=450)
plt.savefig("C:/teamwork_without_talking/results/expected_report_frequency_vs_cost_"+str(year)+".svg",dpi=450)
plt.show()
