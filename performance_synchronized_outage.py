import matplotlib
from matplotlib.lines import lineStyles
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime


year = '2021' # '2020' or '2021'
control_scenarios = ['hi-fi','lo-fi','local']
#control_scenarios = ['hi-fi','lo-fi']
outage_lengths = [datetime.timedelta(minutes=5),datetime.timedelta(minutes=10),datetime.timedelta(minutes=15),
                      datetime.timedelta(minutes=30),datetime.timedelta(hours=1),datetime.timedelta(hours=2),
                      datetime.timedelta(hours=4),datetime.timedelta(hours=8),datetime.timedelta(days=1),
                      datetime.timedelta(days=2),datetime.timedelta(days=3),datetime.timedelta(days=7),
                      datetime.timedelta(days=14),datetime.timedelta(days=30),datetime.timedelta(days=60),
                      datetime.timedelta(days=90),datetime.timedelta(days=120)]
#outage_lengths = [datetime.timedelta(days=90)] # for dev

scores = pd.DataFrame(index=control_scenarios, columns=outage_lengths)

for control_scenario in control_scenarios:
    for outage_length in outage_lengths:
        # format the outage_length so it can go in a filename
        outage_length_string = str(outage_length).replace(':','_')
        outage_length_string = outage_length_string.replace(' ','_')
        outage_length_string = outage_length_string.replace(',','')
        # read in the corresponding cost file
        df = pd.read_csv("C:/teamwork_without_talking/results/"+str(control_scenario) + "_synchronized_" + str(outage_length_string) + "_summer_"+str(year) +"_costs.csv")
        scores.at[control_scenario, outage_length] = df[' total cost'][0]
        # try a higher penalty on TSS
        #scores.at[control_scenario, packet_loss] = df['flood cost'][0] + df[' flow cost'][0] + 1000000*df[' TSS loading (kg)'][0]
    
print(scores)

# load in the cost for the centralized scenario
centralized_cost = pd.read_csv(str("C:/teamwork_without_talking/results/centralized_0.0_summer_"+str(year)+"_costs.csv"))[' total cost'][0]
uncontrolled_cost = pd.read_csv(str("C:/teamwork_without_talking/results/uncontrolled_0.0_summer_"+str(year)+"_costs.csv"))[' total cost'][0]
# try a higher penalty on TSS 
#centralized_cost = pd.read_csv(str("C:/teamwork_without_talking/results/centralized_0.0_summer_"+str(year)+"_costs.csv"))['flood cost'][0] + pd.read_csv(str("C:/teamwork_without_talking/results/centralized_0.0_summer_"+str(year)+"_costs.csv"))[' flow cost'][0]  + 1000000*pd.read_csv(str("C:/teamwork_without_talking/results/centralized_0.0_summer_"+str(year)+"_costs.csv"))[' TSS loading (kg)'][0]
print(centralized_cost)
print(uncontrolled_cost)
# print in scientific notation
print("{:.2e}".format(centralized_cost))
print("{:.2e}".format(uncontrolled_cost))

# plot the scores with cost as the y axis and outage length as the x axis
l_width = 7
fig, ax = plt.subplots(figsize=(10,7.5)) # 12x9 for paper figure, smaller for abstract (relatively bigger text)
#ax.plot(expected_report_frequency, 1 - scores.loc['hi-fi']/uncontrolled_cost, label='high fidelity', linestyle='dashed',color='blue', marker='o', linewidth=l_width, markersize = 3*l_width)
outage_length_labels = [str(ol.days) + " days" if ol.days > 0 else str(ol.seconds//3600) + " hours" for ol in outage_lengths]

ax.plot(outage_length_labels, 1 - scores.loc['hi-fi']/uncontrolled_cost, label='new method', linestyle='dashed',color='blue', marker='o', linewidth=l_width, markersize = 3*l_width)

try:
    ax.plot(outage_length_labels, 1 - scores.loc['lo-fi']/uncontrolled_cost, label='low fidelity', linestyle='dotted',color='red', marker='o', linewidth=l_width, markersize = 3*l_width)
except:
    pass
try:
    ax.plot(outage_length_labels, 1 - scores.loc['local']/uncontrolled_cost, label='local', linestyle='dashdot', color='green',marker='o', linewidth=l_width, markersize = 3*l_width)
except:
    pass
ax.axhline(y=1 - centralized_cost/uncontrolled_cost, linestyle='solid', label='centralized',color='black', linewidth=l_width)
#ax.axhline(y=uncontrolled_cost, linestyle='dashed', label='uncontrolled',color='black', linewidth=l_width)
ax.set_xlabel('Outage Length',fontsize='xx-large')
ax.set_ylabel('Cost\nReduction\nfrom\nUncontrolled',fontsize='xx-large',labelpad = 40)
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
if len(control_scenarios) < 2:
    plt.savefig("C:/teamwork_without_talking/results/outage_length_vs_cost_"+str(year)+"_abstract_version.png",dpi=450)
    plt.savefig("C:/teamwork_without_talking/results/outage_length_vs_cost_"+str(year)+"_abstract_version.svg",dpi=450)
plt.savefig("C:/teamwork_without_talking/results/outage_length_vs_cost_"+str(year)+".png",dpi=450)
plt.savefig("C:/teamwork_without_talking/results/outage_length_vs_cost_"+str(year)+".svg",dpi=450)
plt.show()

