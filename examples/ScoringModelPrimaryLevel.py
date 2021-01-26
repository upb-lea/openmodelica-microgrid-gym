########################################################################
# Scoring Model of the primary level to quantify the controller performance
# Reading the metrics dataframes that were saved in a pkl file
# Comparison of the metrics
# Each Metric is equally weighted --> could be edited by the user
import pandas as pd
import numpy as np

##########################################################################
df_vd_controller1 = pd.read_pickle("./df_metrics_vd_controller1_droop.pkl")
df_vq_controller1 = pd.read_pickle("./df_metrics_vq_controller1_droop.pkl")
df_slave_frequency_controller1 = pd.read_pickle("./df_metrics_slave_f_controller1_droop.pkl")
df_slave_frequency_controller1['Value'].round(decimals=4)

##########################################################################

df_vd_controller2 = pd.read_pickle("./df_metrics_vd_controller2_droop.pkl")
df_vq_controller2 = pd.read_pickle("./df_metrics_vq_controller2_droop.pkl")
df_slave_frequency_controller2 = pd.read_pickle("./df_metrics_slave_f_controller2_droop.pkl")

###########################################################################

########Vd Two Inverter Droop##############################################

comparison_vd = pd.concat([df_vd_controller1, df_vd_controller2], axis=1)
comparison_vd.columns = ['Controller_1', 'Controller_2']
comparison_vd = comparison_vd.drop(comparison_vd[(comparison_vd == 'No overshoot').any(
    1)].index)  # If no overshoot, the overshoot will not be considered
# Weight_Order = Overshoot | Rise Time | Settling Time | RMSE | Steady State Error
comparison_vd = comparison_vd.assign(Weight=[1, 1, 1, 1, 1])  # Metrics are given weights
comparison_vd['Controller_1'] = comparison_vd['Controller_1'].astype(float).round(4)
comparison_vd['Controller_2'] = comparison_vd['Controller_2'].astype(float).round(4)
comparison_vd['Better Performer'] = np.where(comparison_vd['Controller_1'] > comparison_vd['Controller_2'],
                                             'Controller_2',
                                             (np.where(comparison_vd['Controller_1'] == comparison_vd['Controller_2'],
                                                       'equal', 'Controller_1')))  # Comparison of controllers
Controller_1_better_performer = comparison_vd.loc[comparison_vd['Better Performer'] == 'Controller_1']
Controller_2_better_performer = comparison_vd.loc[comparison_vd['Better Performer'] == 'Controller_2']
Controller_1_points_vd = Controller_1_better_performer['Weight'].sum()  # Scores of Controller 1 are summed up
Controller_2_points_vd = Controller_2_better_performer['Weight'].sum()  # Scores of Controller 2 are summed up

print()
print("###############################")
print("Results for Vd - Primary level")
print("###############################")
print(comparison_vd)
print()
print("Controller1_Vd_Points: ", Controller_1_points_vd)
print("Controller2_Vd_Points: ", Controller_2_points_vd)

########Vq Two Inverter Droop##############################################

comparison_vq = pd.concat([df_vq_controller1, df_vq_controller2], axis=1)
comparison_vq.columns = ['Controller_1', 'Controller_2']
# Weight_Order = RMSE | Steady-State-Error | Absolute Peak Value
comparison_vq = comparison_vq.assign(Weight=[1, 1, 1])  # Metrics are given weights
comparison_vq['Controller_1'] = comparison_vq['Controller_1'].astype(float).round(4)
comparison_vq['Controller_2'] = comparison_vq['Controller_2'].astype(float).round(4)
comparison_vq['Better Performer'] = np.where(comparison_vq['Controller_1'] > comparison_vq['Controller_2'],
                                             'Controller_2',
                                             (np.where(comparison_vq['Controller_1'] == comparison_vq['Controller_2'],
                                                       'equal', 'Controller_1')))  # Comparison of controllers
Controller_1_better_performer = comparison_vq.loc[comparison_vq['Better Performer'] == 'Controller_1']
Controller_2_better_performer = comparison_vq.loc[comparison_vq['Better Performer'] == 'Controller_2']
Controller_1_points_vq = Controller_1_better_performer['Weight'].sum()  # Scores of Controller 1 are summed up
Controller_2_points_vq = Controller_2_better_performer['Weight'].sum()  # Scores of Controller 2 are summed up

print()
print("###############################")
print("Results for Vq - Primary level")
print("###############################")
print(comparison_vq)
print()
print("Controller1_Vq_Points: ", Controller_1_points_vq)
print("Controller2_Vq_Points: ", Controller_2_points_vq)

#######Slave Frequency Two Inverter Droop########

comparison_slave_frequency = pd.concat([df_slave_frequency_controller1, df_slave_frequency_controller2], axis=1)
comparison_slave_frequency.columns = ['Controller_1', 'Controller_2']
# Weight_Order = Overshoot | Rise Time | Settling Time | RMSE | Steady State Error
comparison_slave_frequency = comparison_slave_frequency.assign(Weight=[1, 1, 1, 1, 1])  # Metrics are given weights
comparison_slave_frequency = comparison_slave_frequency.drop(comparison_slave_frequency[
                                                                 (comparison_slave_frequency == 'No overshoot').any(
                                                                     1)].index)  # If no os, row will be dropped
comparison_slave_frequency['Controller_1'] = comparison_slave_frequency['Controller_1'].astype(float).round(4)
comparison_slave_frequency['Controller_2'] = comparison_slave_frequency['Controller_2'].astype(float).round(4)
comparison_slave_frequency['Better Performer'] = np.where(
    comparison_slave_frequency['Controller_1'] > comparison_slave_frequency['Controller_2'], 'Controller_2',
    (np.where(comparison_slave_frequency['Controller_1'] == comparison_slave_frequency['Controller_2'],
              'equal', 'Controller_1')))  # Comparison of controllers
Controller_1_better_performer = comparison_slave_frequency.loc[
    comparison_slave_frequency['Better Performer'] == 'Controller_1']
Controller_2_better_performer = comparison_slave_frequency.loc[
    comparison_slave_frequency['Better Performer'] == 'Controller_2']
Controller_1_points_slave_frequency = Controller_1_better_performer[
    'Weight'].sum()  # Scores of Controller 1 are summed up
Controller_2_points_slave_frequency = Controller_2_better_performer[
    'Weight'].sum()  # Scores of Controller 2 are summed up

print()
print("###########################")
print("Results for Slave Frequency")
print("###########################")
print(comparison_slave_frequency)
print()
print("Controller1_SlaveFrequency_Points: ", Controller_1_points_slave_frequency)
print("Controller2_SlaveFrequency_Points: ", Controller_2_points_slave_frequency)

######Overall Results######

overall_Controller_1_points = Controller_1_points_vd + Controller_1_points_vq + Controller_1_points_slave_frequency
overall_Controller_2_points = Controller_2_points_vd + Controller_2_points_vq + Controller_2_points_slave_frequency
print()
print()
print()
print("Summary")
print("#################################")
print("Overall Results for Primary level")
print("#################################")
print()

print()
print("Controller_1_Points_overall: ", overall_Controller_1_points)
print("Controller_2_Points_overall: ", overall_Controller_2_points)
print()
if overall_Controller_2_points < overall_Controller_1_points:
    print("Controller 1 is the better performer.The results still need to be treated with caution,")
    print("because depending on the purpose of the controller in the microgrid, ")
    print("individual metrics may be more important than others. ")
elif overall_Controller_2_points == overall_Controller_1_points:
    print("No controller is better than the other.The results still need to be treated with caution,")
    print("because depending on the purpose of the controller in the microgrid,")
    print("individual metrics may be more important than others.")
else:
    print("Controller 2 is the better performer.The results still need to be treated with caution,")
    print("because depending on the purpose of the controller in the microgrid,")
    print("individual metrics may be more important than others.")


