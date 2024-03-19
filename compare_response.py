import matplotlib
from matplotlib.lines import lineStyles
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

from seaborn._core.properties import LineStyle


year = '2021' # or '2021'
duration = 'storm' # 'full' or 'storm'
dropout_rate = '0.9993_' # many options for this one, see tune_control.py
# include the _ after to make sure you don't include file names for which this is a substring (e.g., 0.99 also grabs 0.999 and 0.9999)
control_scenarios = ['centralized','hi-fi','lo-fi','local']
control_scenarios = ['centralized','hi-fi']

path = "C:/teamwork_without_talking/results/"

# Load data
files = []
for file in os.listdir(path):
    if year in file and dropout_rate in file and ".pkl" in file:
        files.append(file)
        
# for each file, load the data
for file in files:
    if 'hi-fi' in file:
        print(path+file)
        hi_fi_data = pd.read_pickle(path+file)
    elif 'lo-fi' in file:
        print(path+file)
        lo_fi_data = pd.read_pickle(path+file)
    elif 'local' in file:
        print(path+file)
        local_data = pd.read_pickle(path+file)
    else:
        print('File not recognized')
        
# also add the centralized data, which won't have the dropout rate in the file name
centralized_data = pd.read_pickle(str("C:/teamwork_without_talking/results/centralized_0.0_summer_" + str(year) + ".pkl"))

# project file is in english units
cfs2cms = 35.315
cfs2lps = 28.3168
ft2meters = 3.281
basin_max_depths =  [10.0, 10.0, 20.0, 10.0, 10.0, 13.72] # feet
flow_threshold_value = 0.5 # cfs
flow_threshold = np.ones(6)*flow_threshold_value # cfs

valves = ["O1","O4","O6","O7","O8","O10"]
storage_nodes = ["1","4","6","7","8","10"]

# plot the data
# make a grid of 3 rows and 4 columns where each plot is twice as wide as it is tall
fig, axs = plt.subplots(4, 3, figsize=(12, 9)) # 12x9 for paper figure, smaller for abstract (relatively bigger text)
line_width = 3

# add the skeleton of the network
#fig.add_artist(matplotlib.lines.Line2D([0.2, 0.2,0.9], [0.9,0.15,0.15], transform=fig.transFigure, color='black', linestyle='solid', linewidth=10,alpha=0.08))
#fig.add_artist(matplotlib.lines.Line2D([0.8, 0.8,0.215], [0.9,0.4,0.4], transform=fig.transFigure, color='black', linestyle='solid', linewidth=10,alpha=0.08))
#fig.add_artist(matplotlib.lines.Line2D([0.5, 0.5,0.785], [0.9,0.6,0.42], transform=fig.transFigure, color='black', linestyle='solid', linewidth=10,alpha=0.08))
skeleton_linewidth = 15
skeleton_alpha = 0.25
fig.add_artist(matplotlib.lines.Line2D([0.18, 0.18], [0.76,0.74], transform=fig.transFigure, color='black', linestyle='solid', linewidth=skeleton_linewidth,alpha=skeleton_alpha))
fig.add_artist(matplotlib.lines.Line2D([0.18, 0.18], [0.52,0.50], transform=fig.transFigure, color='black', linestyle='solid', linewidth=skeleton_linewidth,alpha=skeleton_alpha))
fig.add_artist(matplotlib.lines.Line2D([0.18, 0.18], [0.28,0.26], transform=fig.transFigure, color='black', linestyle='solid', linewidth=skeleton_linewidth,alpha=skeleton_alpha))

fig.add_artist(matplotlib.lines.Line2D([0.48, 0.48], [0.76,0.74], transform=fig.transFigure, color='black', linestyle='solid', linewidth=skeleton_linewidth,alpha=skeleton_alpha))
fig.add_artist(matplotlib.lines.Line2D([0.64, 0.66], [0.52,0.50], transform=fig.transFigure, color='black', linestyle='solid', linewidth=skeleton_linewidth,alpha=skeleton_alpha))

fig.add_artist(matplotlib.lines.Line2D([0.84, 0.84], [0.76,0.74], transform=fig.transFigure, color='black', linestyle='solid', linewidth=skeleton_linewidth,alpha=skeleton_alpha))
fig.add_artist(matplotlib.lines.Line2D([0.84, 0.84], [0.52,0.50], transform=fig.transFigure, color='black', linestyle='solid', linewidth=skeleton_linewidth,alpha=skeleton_alpha))

fig.add_artist(matplotlib.lines.Line2D([0.64, 0.66], [0.40,0.40], transform=fig.transFigure, color='black', linestyle='solid', linewidth=skeleton_linewidth,alpha=skeleton_alpha))
fig.add_artist(matplotlib.lines.Line2D([0.31, 0.33], [0.40,0.40], transform=fig.transFigure, color='black', linestyle='solid', linewidth=skeleton_linewidth,alpha=skeleton_alpha))

fig.add_artist(matplotlib.lines.Line2D([0.64, 0.66], [0.15,0.15], transform=fig.transFigure, color='black', linestyle='solid', linewidth=skeleton_linewidth,alpha=skeleton_alpha))
fig.add_artist(matplotlib.lines.Line2D([0.31, 0.33], [0.15,0.15], transform=fig.transFigure, color='black', linestyle='solid', linewidth=skeleton_linewidth,alpha=skeleton_alpha))


if year == '2020':
    if duration == 'storm':
        start_date = datetime.datetime(2020, 5, 14, 0, 0)
        end_date = datetime.datetime(2020, 5, 20, 0, 0)
    else:
        start_date = datetime.datetime(2020, 3, 1, 0, 0)
        end_date = datetime.datetime(2020, 10, 1, 0, 0)

if year == '2021':
    if duration == 'storm': # dates TBD for 2021
        start_date = datetime.datetime(2021, 6, 1, 0, 0)
        end_date = datetime.datetime(2021, 6, 30, 0, 0)
    else:
        start_date = datetime.datetime(2021, 3, 1, 0, 0)
        end_date = datetime.datetime(2021, 10, 1, 0, 0)

# plot all the responses
for control_scenario in control_scenarios:
    if control_scenario == 'centralized':
        data = centralized_data
        color = 'black'
        line_style = 'solid'
    elif control_scenario == 'hi-fi':
        data = hi_fi_data
        color = 'blue'
        line_style = 'dashed'
    elif control_scenario == 'lo-fi':
        continue # only do hi-fi and centralized for the conference abstract (simplify things)
        data = lo_fi_data
        color = 'orange'
        line_style = 'dotted'
    elif control_scenario == 'local':
        continue
        data = local_data
        color = 'green'
        line_style = 'dashdot'
    else:
        print('Control scenario not recognized')

    # plot 10 depth
    axs[0,0].plot(data['simulation_time'],np.array(data['depthN']['10']) / ft2meters, label=control_scenario,color = color,linestyle = line_style,linewidth = line_width,alpha=0.6)
    axs[0,0].hlines(basin_max_depths[5] / ft2meters, data['simulation_time'][0], data['simulation_time'][-1], label='threshold',color='red', linestyle='dotted',linewidth = line_width,alpha=0.6)
    axs[0,0].annotate("10 depth",xy=(0.1,0.75),xycoords = 'axes fraction', fontsize = 'xx-large')
    axs[0,0].set_xticks([])
    axs[0,0].set_xlim(start_date, end_date)
    #axs[0,0].set_yticks([])
    #axs[0,0].set_ylabel('Depth (m)')
    # draw an arrow thorugh the x axis pointing downward
    #axs[0,0].annotate('', xy=(0.5, -0.1), xycoords='axes fraction', xytext=(0.5, 0.1), textcoords='axes fraction', arrowprops=dict(facecolor='gray',headwidth=10,headlength=10))
    # plot 10 flow
    axs[1,0].plot(data['simulation_time'],np.array(data['flow']['O10']) * cfs2lps, label=control_scenario,color = color,linestyle = line_style,linewidth = line_width,alpha=0.6)
    axs[1,0].hlines(flow_threshold_value * cfs2lps, data['simulation_time'][0], data['simulation_time'][-1], label='threshold',color='red', linestyle ='dotted',linewidth = line_width,alpha=0.6)
    axs[1,0].annotate("10 flow",xy=(0.1,0.75),xycoords = 'axes fraction', fontsize = 'xx-large')
    axs[1,0].set_xticks([])
    axs[1,0].set_xlim(start_date, end_date)
    #axs[1,0].set_yticks([])
    #axs[1,0].set_ylabel('Flow (lps)')
    #axs[1,0].annotate('', xy=(0.5, -0.1), xycoords='axes fraction', xytext=(0.5, 0.1), textcoords='axes fraction', arrowprops=dict(facecolor='gray',headwidth=10,headlength=10))
    # plot 7 depth
    axs[0,1].plot(data['simulation_time'],np.array(data['depthN']['7']) / ft2meters, label=control_scenario,color = color,linestyle = line_style,linewidth = line_width,alpha=0.6)
    axs[0,1].hlines(basin_max_depths[3] / ft2meters, data['simulation_time'][0], data['simulation_time'][-1], label='threshold',color='red', linestyle='dotted',linewidth = line_width,alpha=0.6)
    axs[0,1].annotate("7 depth",xy=(0.1,0.75),xycoords = 'axes fraction', fontsize = 'xx-large')
    axs[0,1].set_xticks([])
    axs[0,1].set_xlim(start_date, end_date)
    #axs[0,1].set_yticks([])
    #axs[0,1].annotate('', xy=(0.5, -0.1), xycoords='axes fraction', xytext=(0.5, 0.1), textcoords='axes fraction', arrowprops=dict(facecolor='gray',headwidth=10,headlength=10))
    # plot 7 flow
    axs[1,1].plot(data['simulation_time'],np.array(data['flow']['O7']) * cfs2lps, label=control_scenario,color = color,linestyle = line_style,linewidth = line_width,alpha=0.6)
    axs[1,1].hlines(flow_threshold_value * cfs2lps, data['simulation_time'][0], data['simulation_time'][-1], label='threshold',color='red', linestyle ='dotted',linewidth = line_width,alpha=0.6)
    axs[1,1].annotate("7 flow",xy=(0.1,0.75),xycoords = 'axes fraction', fontsize = 'xx-large')
    axs[1,1].set_xticks([])
    axs[1,1].set_xlim(start_date, end_date)
    #axs[1,1].set_yticks([])
    #axs[1,1].annotate('', xy=(1.25, -0.25), xycoords='axes fraction', xytext=(1.05, -0.05), textcoords='axes fraction', arrowprops=dict(facecolor='black', shrink=0.05))
    # plot 8 depth
    axs[0,2].plot(data['simulation_time'],np.array(data['depthN']['8']) / ft2meters, label=control_scenario,color = color,linestyle = line_style,linewidth = line_width,alpha=0.6)
    axs[0,2].hlines(basin_max_depths[4] / ft2meters, data['simulation_time'][0], data['simulation_time'][-1], label='threshold',color='red', linestyle='dotted',linewidth = line_width,alpha=0.6)
    axs[0,2].annotate("8 depth",xy=(0.1,0.75),xycoords = 'axes fraction', fontsize = 'xx-large')
    axs[0,2].set_xticks([])
    axs[0,2].set_xlim(start_date, end_date)
    #axs[0,2].set_yticks([])
    #axs[0,2].annotate('', xy=(0.5, -0.1), xycoords='axes fraction', xytext=(0.5, 0.1), textcoords='axes fraction', arrowprops=dict(facecolor='gray',headwidth=10,headlength=10))
    # plot 8 flow
    axs[1,2].plot(data['simulation_time'],np.array(data['flow']['O8']) * cfs2lps, label=control_scenario,color = color,linestyle = line_style,linewidth = line_width,alpha=0.6)
    axs[1,2].hlines(flow_threshold_value * cfs2lps, data['simulation_time'][0], data['simulation_time'][-1], label='threshold',color='red', linestyle ='dotted',linewidth = line_width,alpha=0.6)
    axs[1,2].annotate("8 flow",xy=(0.1,0.75),xycoords = 'axes fraction', fontsize = 'xx-large')
    axs[1,2].set_xticks([])
    axs[1,2].set_xlim(start_date, end_date)
    #axs[1,2].set_yticks([])
    #axs[1,2].annotate('', xy=(0.5, -0.1), xycoords='axes fraction', xytext=(0.5, 0.1), textcoords='axes fraction', arrowprops=dict(facecolor='gray',headwidth=10,headlength=10))
    # plot 6 depth
    axs[2,2].plot(data['simulation_time'],np.array(data['depthN']['6']) / ft2meters, label=control_scenario,color = color,linestyle = line_style,linewidth = line_width,alpha=0.6)
    axs[2,2].hlines(basin_max_depths[2] / ft2meters, data['simulation_time'][0], data['simulation_time'][-1], label='threshold',color='red', linestyle='dotted',linewidth = line_width,alpha=0.6)
    axs[2,2].annotate("6 depth",xy=(0.1,0.75),xycoords = 'axes fraction', fontsize = 'xx-large')
    axs[2,2].set_xticks([])
    axs[2,2].set_xlim(start_date, end_date)
    #axs[2,2].set_yticks([])
    #axs[2,2].annotate('', xy=(-0.3, 0.5), xycoords='axes fraction', xytext=(-0.05, 0.5), textcoords='axes fraction', arrowprops=dict(facecolor='black', shrink=0.05))
    # plot 6 flow
    axs[2,1].plot(data['simulation_time'],np.array(data['flow']['O6']) * cfs2lps, label=control_scenario,color = color,linestyle = line_style,linewidth = line_width,alpha=0.6)
    axs[2,1].hlines(flow_threshold_value * cfs2lps, data['simulation_time'][0], data['simulation_time'][-1], label='threshold',color='red', linestyle ='dotted',linewidth = line_width,alpha=0.6)
    axs[2,1].annotate("6 flow",xy=(0.1,0.75),xycoords = 'axes fraction', fontsize = 'xx-large')
    axs[2,1].set_xticks([])
    #axs[2,1].set_yticks([])
    #axs[2,1].annotate('', xy=(-0.3, 0.5), xycoords='axes fraction', xytext=(-0.05, 0.5), textcoords='axes fraction', arrowprops=dict(facecolor='black', shrink=0.05))
    # plot 4 depth
    axs[2,0].plot(data['simulation_time'],np.array(data['depthN']['4']) / ft2meters, label=control_scenario,color = color,linestyle = line_style,linewidth = line_width,alpha=0.6)
    axs[2,0].hlines(basin_max_depths[1] / ft2meters, data['simulation_time'][0], data['simulation_time'][-1], label='threshold',color='red', linestyle='dotted',linewidth = line_width,alpha=0.6)
    axs[2,0].annotate("4 depth",xy=(0.1,0.75),xycoords = 'axes fraction', fontsize = 'xx-large')
    axs[2,0].set_xticks([])
    axs[2,0].set_xlim(start_date, end_date)
    #axs[2,0].set_ylabel('Depth (m)')
    #axs[2,0].set_yticks([])
    #axs[2,0].annotate('', xy=(0.5, -0.1), xycoords='axes fraction', xytext=(0.5, 0.1), textcoords='axes fraction', arrowprops=dict(facecolor='gray',headwidth=10,headlength=10))
    # plot 4 flow
    axs[3,0].plot(data['simulation_time'],np.array(data['flow']['O4']) * cfs2lps, label=control_scenario,color = color,linestyle = line_style,linewidth = line_width,alpha=0.6)
    axs[3,0].hlines(flow_threshold_value * cfs2lps, data['simulation_time'][0], data['simulation_time'][-1], label='threshold',color='red', linestyle ='dotted',linewidth = line_width,alpha=0.6)
    axs[3,0].annotate("4 flow",xy=(0.1,0.75),xycoords = 'axes fraction', fontsize = 'xx-large')
    axs[3,0].set_xlim(start_date, end_date)
    #axs[3,0].annotate('', xy=(1.3, 0.5), xycoords='axes fraction', xytext=(1.05, 0.5), textcoords='axes fraction', arrowprops=dict(facecolor='black', shrink=0.05))
    axs[3,0].set_xticks([start_date, start_date + (end_date - start_date)/2, end_date])
    
    #axs[3,0].set_ylabel('Flow (lps)')
    #axs[3,0].set_yticks([])
    # plot 1 depth
    axs[3,1].plot(data['simulation_time'],np.array(data['depthN']['1']) / ft2meters, label=control_scenario,color = color,linestyle = line_style,linewidth = line_width,alpha=0.6)
    axs[3,1].hlines(basin_max_depths[0] / ft2meters, data['simulation_time'][0], data['simulation_time'][-1], label='threshold',color='red', linestyle='dotted',linewidth = line_width,alpha=0.6)
    axs[3,1].annotate("1 depth",xy=(0.1,0.75),xycoords = 'axes fraction', fontsize = 'xx-large')
    #axs[3,1].set_yticks([])
    axs[3,1].set_xlim(start_date, end_date)
    axs[3,1].set_xticks([start_date, start_date + (end_date - start_date)/2, end_date])
    #axs[3,1].annotate('', xy=(1.3, 0.5), xycoords='axes fraction', xytext=(1.05, 0.5), textcoords='axes fraction', arrowprops=dict(facecolor='black', shrink=0.05))
    # plot 1 flow
    axs[3,2].plot(data['simulation_time'],np.array(data['flow']['O1']) * cfs2lps, label=control_scenario,color = color,linestyle = line_style,linewidth = line_width,alpha=0.6)
    axs[3,2].hlines(flow_threshold_value * cfs2lps, data['simulation_time'][0], data['simulation_time'][-1], label='threshold',color='red', linestyle ='dotted',linewidth = line_width,alpha=0.6)
    axs[3,2].annotate("1 flow",xy=(0.1,0.75),xycoords = 'axes fraction', fontsize = 'xx-large')
    #axs[3,2].set_yticks([])
    # do beginning, middle, and end dates
    axs[3,2].set_xticks([start_date, start_date + (end_date - start_date)/2, end_date])
    axs[3,2].set_xlim(start_date, end_date)


# remove any duplicate entries from the legend handles
handles, labels = axs[2,1].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
axs[2,1].legend(by_label.values(), by_label.keys(),fontsize='x-large', loc='upper right')



plt.tight_layout()
# save the figure, including the dropout rate and 'full' or 'storm' in the file name
plt.savefig("C:/teamwork_without_talking/results/compare_response_"+year+"_"+duration+"_"+dropout_rate+".png",dpi=300)
plt.savefig("C:/teamwork_without_talking/results/compare_response_"+year+"_"+duration+"_"+dropout_rate+".svg",dpi=300)
plt.show()


