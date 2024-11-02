import matplotlib
from matplotlib.lines import lineStyles
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime


ft2meters = 0.3048

year = "2021" # or "2021"
# options are: 'centralized', 'hi-fi', 'lo-fi', and 'local'
control_scenario = 'hi-fi' 
packet_loss_chance = 0.9995

# for figure 5, just need to load in the timeseries of estiamtes of one node by several other nodes
perspectives = ['truth','server','1','4','6','7','8','10']
records = {}
for perspective in perspectives: # was for 2020, now 2021
    records[perspective] = pd.read_csv("C:/teamwork_without_talking/results/"+str(control_scenario) + "_" + str(packet_loss_chance) + "_summer_"+str(year) +"_teammate_inference_tracking_"+str(perspective)+".csv",index_col=0,parse_dates=True)
    '''
    if year == "2020":
        records[perspective] = pd.read_csv("C:/teamwork_without_talking/results/hi-fi_0.999_summer_2020_teammate_inference_tracking_" + str(perspective) + ".csv",index_col=0,parse_dates=True)
    else:
        records[perspective] = pd.read_csv("C:/teamwork_without_talking/results/hi-fi_0.9993_summer_2021_teammate_inference_tracking_" + str(perspective) + ".csv",index_col=0,parse_dates=True)
    '''
    
# plot the estimated state of node X from the perspective of: truth, a, b, and c
fig, ax = plt.subplots(figsize=(12,9))
for perspective in ['truth','4','6','7']:
    if perspective == 'truth':
        color = 'black'
        style = 'solid'
    elif perspective == '4':
        color = 'blue'
        style = 'dashed'
    elif perspective == '6':
        color = 'green'
        style = 'dotted'
    elif perspective == '7':
        color = 'red'
        style = 'dashdot'
        
    if perspective != "truth":
        ax.plot(records[perspective]['8']*ft2meters, label=str("estimate maintained by " + perspective),alpha=0.6,linewidth=7,color=color,linestyle = style)
    else:
        ax.plot(records[perspective]['8']*ft2meters, label="measurement",alpha=0.6,linewidth=7,color=color,linestyle = style)

ax.legend(fontsize='xx-large',loc='upper left')

# annotate the RMSE between 1,4, and 8 and the truth
idx = 0
for perspective in ['4','6','7']:
    if perspective == '4':
        color = 'blue'
    elif perspective == '6':
        color = 'green'
    elif perspective == '7':
        color = 'red'

    rmse = np.sqrt(np.mean((records[perspective]['8']-records['truth']['8'])**2))
    ax.text(0.7,0.9-idx*0.05,perspective + " RMSE: " + str(round(rmse,2)) + " m",transform=ax.transAxes,fontsize='xx-large', color=color)
    # add the NSE as well
    #nse = 1 - np.sum((records[perspective]['8']-records['truth']['10'])**2)/np.sum((records['truth']['10']-np.mean(records['truth']['10']))**2)
    #ax.text(0.2,0.2-idx*0.05,perspective + " RMSE: " + str(round(rmse,2)) + "m, NSE: " + str(round(nse,2)),transform=ax.transAxes,fontsize='x-large')
    idx += 1
    
# draw a little sketch in text that shows the network topology: 10 and 8 both connect to 4, 4 connects to 1
ax.text(0.51,0.95,"7",transform=ax.transAxes,fontsize='xx-large', color = 'red')
ax.text(0.6,0.95,"8",transform=ax.transAxes,fontsize='xx-large', color = 'black')
ax.text(0.55,0.85,"6",transform=ax.transAxes,fontsize='xx-large', color = 'green')
ax.text(0.55,0.75,"4",transform=ax.transAxes,fontsize='xx-large', color=  'blue')
skeleton_linewidth = 2
skeleton_alpha = 0.5
fig.add_artist(matplotlib.lines.Line2D([0.53, 0.55], [0.94,0.87], transform=ax.transAxes, color='black', linestyle='solid', linewidth=skeleton_linewidth,alpha=skeleton_alpha)) # 10 to 4
fig.add_artist(matplotlib.lines.Line2D([0.60, 0.57], [0.95,0.87], transform=ax.transAxes, color='black', linestyle='solid', linewidth=skeleton_linewidth,alpha=skeleton_alpha)) # 7 to 4
fig.add_artist(matplotlib.lines.Line2D([0.56, 0.56], [0.84,0.79], transform=ax.transAxes, color='black', linestyle='solid', linewidth=skeleton_linewidth,alpha=skeleton_alpha)) # 4 to 1

# if it's hard to see, you might want to bound the x-axis to a smaller range. just use set_xlim
#ax.set_xlim([datetime.datetime(2020, 5, 1, 0, 0), datetime.datetime(2020, 7, 1, 0, 10)])
# for 2021
ax.set_xlim([datetime.datetime(2021,8,29,0,0),datetime.datetime(2021,9,20,0,10)])
ax.set_ylim([-0.25,1.5])

# make the y label horizontal
ax.set_ylabel('Node 8\nDepth (m)',fontsize='xx-large',rotation=0,labelpad=30)
# make the x and y ticks xlarge
ax.tick_params(axis='both', which='major', labelsize='xx-large')
# put half as many ticks on the x-axis
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
#ax.set_title("Dynamically coupled nodes have better estimates of each other's state",fontsize='xx-large')
plt.tight_layout()
# save the figure
plt.savefig("C:/teamwork_without_talking/results/teammate_inference_single_example"+str(control_scenario) + "_" + str(packet_loss_chance) + "_summer_"+str(year) +".png",dpi=450)
plt.savefig("C:/teamwork_without_talking/results/teammate_inference_single_example"+str(control_scenario) + "_" + str(packet_loss_chance) + "_summer_"+str(year) +".svg",dpi=450)
plt.show()
#plt.close('all')
    



# for figure 6, will be aggregating RMSE between every pair of nodes based on their relative position in the network
# first, calculate the RMSE between every pair of nodes
rmse = pd.DataFrame(index = ['1','4','6','7','8','10'],columns = ['1','4','6','7','8','10'],dtype=float)
# index is the target, column is the predictor
for target in rmse.index:
    for predictor in rmse.columns:
        rmse.loc[target,predictor] = np.sqrt(np.mean((records[predictor][target]-records['truth'][target])**2))
        # try NSE instead
        #rmse.loc[target,predictor] = 1 - np.sum((records[predictor][target]-records['truth'][target])**2)/np.sum((records['truth'][target]-np.mean(records['truth'][target]))**2)
        # try MAE
        #rmse.loc[target,predictor] = np.mean(np.abs(records[predictor][target]-records['truth'][target]))
        # try kling-gupta efficiency
        #rmse.loc[target,predictor] = 1 - np.sum((records[predictor][target]-records['truth'][target])**2)/np.sum((np.abs(records[predictor][target]-np.mean(records['truth'][target])) + np.abs(records['truth'][target]-np.mean(records['truth'][target]))**2))
        # try normalized RMSE
        #rmse.loc[target,predictor] = np.sqrt(np.mean((records[predictor][target]-records['truth'][target])**2))/np.mean(np.abs(records['truth'][target]-np.mean(records['truth'][target])))

print(rmse)
# plot the RMSE between every pair of nodes as a heatmap (not the final figure, but helpful)
fig, ax = plt.subplots(figsize=(12,9))
im = ax.imshow(rmse.values)
ax.set_xticks(np.arange(len(rmse.columns)))
ax.set_yticks(np.arange(len(rmse.index)))
ax.set_xticklabels(rmse.columns)
ax.set_yticklabels(rmse.index)
ax.set_ylabel("target node")
ax.set_xlabel("predictor node")
for i in range(len(rmse.index)):
    for j in range(len(rmse.columns)):
        text = ax.text(j, i, round(rmse.values[i, j],2), ha="center", va="center", color="w")
ax.set_title("RMSE between every pair of nodes")

plt.tight_layout()
#plt.show()
plt.close('all')
# TODO - change this box and whisker into a scatter, there's not enough data points to justify a box and whisker
# now each square in this heatmap except the diagonals (6x6=36 - 6 = 30) will be assigned to one of 7 bins: 
# 3 upstream, 2 upstream, 1 upstream, 1 downstream, 2 downstream, 3 downstream, different branch
# for the .loc indexing, the first index is the target, the second index is the predictor
# so 3_up means the predictor is 3 nodes upstream of the target
up3 = [rmse.loc['1','7'] , rmse.loc['1','8']]
up2 = [rmse.loc['1','10'] , rmse.loc['1','6'], rmse.loc['4','8'], rmse.loc['4','7']]
up1 = [rmse.loc['1','4'], rmse.loc['4','6'], rmse.loc['4','10'] , rmse.loc['6','7'], rmse.loc['6','8']]
# down1 is the same as up1, but with the indices reversed
down1 = [rmse.loc['4','1'], rmse.loc['6','4'], rmse.loc['10','4'] , rmse.loc['7','6'], rmse.loc['8','6']]
down2 = [rmse.loc['8','4'], rmse.loc['7','4'], rmse.loc['10','1'], rmse.loc['6','1']]
down3 = [rmse.loc['8','1'], rmse.loc['7','1']]
diff_branch = [rmse.loc['7','8'], rmse.loc['8','7'], 
               rmse.loc['6','10'], rmse.loc['10','6'], 
               rmse.loc['8','10'], rmse.loc['10','8'], 
               rmse.loc['7','10'], rmse.loc['10','7']]

# check that the length of those 7 arrays together adds up to 30
print(len(up3) + len(up2) + len(up1) + len(down1) + len(down2) + len(down3) + len(diff_branch))
'''
difference_over_distance = pd.DataFrame(index = ['up3','up2','up1','down1','down2','down3','diff_branch'])# don't fair well... (flatland cavalry)
difference_over_distance.loc['up3','mean'] = np.mean(up3)
difference_over_distance.loc['up2','mean'] = np.mean(up2)
difference_over_distance.loc['up1','mean'] = np.mean(up1)
difference_over_distance.loc['down1','mean'] = np.mean(down1)
difference_over_distance.loc['down2','mean'] = np.mean(down2)
difference_over_distance.loc['down3','mean'] = np.mean(down3)
difference_over_distance.loc['diff_branch','mean'] = np.mean(diff_branch)
'''

# now plot a line chart where each xtick is one of the 7 bins, and the y value is the average RMSE in that bin
fig, ax = plt.subplots(figsize=(12,9))
# plot box plots of each bin, labeled by their names. in the order up3, up2, up1, down1, down2, down3, diff_branch
#ax.boxplot([up3,up2,up1,down1,down2,down3,diff_branch],labels=['3 upstream','2 upstream','1 upstream','1 downstream','2 downstream','3 downstream','different branch'])
# exclude "different branch" for now
#ax.boxplot([up3,up2,up1,down1,down2,down3],labels=['3 upstream','2 upstream','1 upstream','1 downstream','2 downstream','3 downstream'])
# do the same box plot command as above, but make every line thicker
l_width = 5
#ax.boxplot([up3,up2,up1,down1,down2,down3],labels=['3 upstream','2 upstream','1 upstream','1 downstream','2 downstream','3 downstream'],
#           boxprops=dict(linewidth=l_width),whiskerprops=dict(linewidth=l_width),capprops=dict(linewidth=l_width),medianprops=dict(linewidth=l_width))
# make this a scatter plot instead
mark_size = 200
ax.scatter([1]*len(up3),up3,alpha=0.5, s = mark_size, color = 'b')
ax.scatter([2]*len(up2),up2,alpha=0.5, s = mark_size, color = 'g')
ax.scatter([3]*len(up1),up1,alpha=0.5, s = mark_size, color = 'r')
ax.scatter([4]*len(down1),down1,alpha=0.5, s = mark_size, color = 'r')
ax.scatter([5]*len(down2),down2,alpha=0.5, s = mark_size, color = 'g')
ax.scatter([6]*len(down3),down3,alpha=0.5, s = mark_size, color = 'b')
ax.scatter([7]*len(diff_branch),diff_branch,alpha=0.5, s = mark_size, color = 'k') # once results come in evaluate whether to keep this one or not
# add the labels
# if lo-fi, set the y axis to be log scale
if control_scenario == 'lo-fi':
    ax.set_yscale('log')

ax.set_xticks([1,2,3,4,5,6,7])
ax.set_xticklabels(['3 upstream','2 upstream','1 upstream','1 downstream','2 downstream','3 downstream','different branch'],fontsize='xx-large')
#ax.legend(fontsize='x-large')
ax.set_ylabel('RMSE \n(m)',fontsize='xx-large',rotation=0,labelpad=25)
ax.set_xlabel('Position of predictor relative to target',fontsize='xx-large')
ax.tick_params(axis='y', which='major', labelsize='xx-large')
ax.tick_params(axis='x', which='major', labelsize='x-large')
#ax.set_title('Relative network position influences value of local measurements in teammate state inference')
plt.tight_layout()
# save
plt.savefig(str("C:/teamwork_without_talking/results/teammate_inference_relative_position_"+str(control_scenario) + "_" + str(packet_loss_chance) + "_summer_"+str(year) +".png"),dpi=450)
plt.savefig(str("C:/teamwork_without_talking/results/teammate_inference_relative_position_"+str(control_scenario) + "_" + str(packet_loss_chance) + "_summer_"+str(year) +".svg"),dpi=450)

plt.show()
