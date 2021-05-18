########################################################################
# Scoring Model of the inner level to quantify the controller performance
# Reading the metrics dataframes that were saved in a pkl file
# Comparison of the metrics
# Each Metric is equally weighted --> could be edited by the user


import pandas as pd
import numpy as np

#############Important###########################################################
# Save the data frames of the metrics as pkl.....
# .....at the corresponding positions in tf_innerlevel.....py
# The positions are indicated there.

df_id_controller1 = pd.read_pickle("./df_metrics_id_controller1.pkl")
df_iq_controller1 = pd.read_pickle("./df_metrics_iq_controller1.pkl")
df_vd_controller1 = pd.read_pickle("./df_metrics_vd_controller1.pkl")
df_vq_controller1 = pd.read_pickle("./df_metrics_vq_controller1.pkl")

###################################################################
df_id_controller2 = pd.read_pickle("./df_metrics_id_controller2.pkl")
df_vd_controller2 = pd.read_pickle("./df_metrics_vd_controller2.pkl")
df_iq_controller2 = pd.read_pickle("./df_metrics_iq_controller2.pkl")
df_vq_controller2 = pd.read_pickle("./df_metrics_vq_controller2.pkl")

###################id Single Inverter Current Control##########################################

comparison_id = pd.concat([df_id_controller1, df_id_controller2], axis=1)
comparison_id.columns = ['Controller_1', 'Controller_2']
# Weight_Order = Overshoot | Rise Time | Settling Time | RMSE | Steady State Error
comparison_id = comparison_id.assign(Weight=[1, 1, 1, 1, 1])  # Metrics are given weights
comparison_id = comparison_id.drop(comparison_id[(comparison_id == 'No overshoot').any(1)].index)  # drops if no os
comparison_id['Controller_1'] = comparison_id['Controller_1'].astype(float).round(4)
comparison_id['Controller_2'] = comparison_id['Controller_2'].astype(float).round(4)
comparison_id['Better Performer'] = np.where(comparison_id['Controller_1'] > comparison_id['Controller_2'],
                                             'Controller_2',
                                             (np.where(comparison_id['Controller_1'] == comparison_id['Controller_2'],
                                                       'equal',
                                                       'Controller_1')))  # Comparison of controllers

Controller_1_better_performer = comparison_id.loc[comparison_id['Better Performer'] == 'Controller_1']
Controller_2_better_performer = comparison_id.loc[comparison_id['Better Performer'] == 'Controller_2']
Controller_1_points_id = Controller_1_better_performer['Weight'].sum()  # Scores of Controller 1 are summed up
Controller_2_points_id = Controller_2_better_performer['Weight'].sum()  # Scores of Controller 2 are summed up

###################iq Single Inverter Current Control##########################################

comparison_iq = pd.concat([df_iq_controller1, df_iq_controller2], axis=1)
comparison_iq.columns = ['Controller_1', 'Controller_2']
# Weight_Order = RMSE | Steady-State-Error | Absolute Peak Value
comparison_iq = comparison_iq.assign(Weight=[1, 1, 1])  # Metrics are given weights
comparison_iq['Controller_1'] = comparison_iq['Controller_1'].astype(float).round(4)
comparison_iq['Controller_2'] = comparison_iq['Controller_2'].astype(float).round(4)
comparison_iq['Better Performer'] = np.where(comparison_iq['Controller_1'] > comparison_iq['Controller_2'],
                                             'Controller_2',
                                             (np.where(comparison_iq['Controller_1'] == comparison_iq['Controller_2'],
                                                       'equal',
                                                       'Controller_1')))  # Comparison of controllers
Controller_1_better_performer = comparison_iq.loc[comparison_iq['Better Performer'] == 'Controller_1']
Controller_2_better_performer = comparison_iq.loc[comparison_iq['Better Performer'] == 'Controller_2']
Controller_1_points_iq = Controller_1_better_performer['Weight'].sum()  # Scores of Controller 1 are summed up
Controller_2_points_iq = Controller_2_better_performer['Weight'].sum()  # Scores of Controller 2 are summed up

###################Vd Cascaded Structure (Single Inverter Voltage Current Control)#######################
comparison_vd = pd.concat([df_vd_controller1, df_vd_controller2], axis=1)  # Df of C1 and C2 are merged
comparison_vd.columns = ['Controller_1', 'Controller_2']
# Weight_Order = Overshoot | Rise Time | Settling Time | RMSE | Steady State Error
comparison_vd = comparison_vd.assign(Weight=[1, 1, 1, 1, 1])  # Metrics are given weights
comparison_vd = comparison_vd.drop(
    comparison_vd[
        (comparison_vd == 'No overshoot').any(1)].index)  # If no overshoot, the overshoot will not be considered
comparison_vd['Controller_1'] = comparison_vd['Controller_1'].astype(float).round(4)
comparison_vd['Controller_2'] = comparison_vd['Controller_2'].astype(float).round(4)
comparison_vd['Better Performer'] = np.where(comparison_vd['Controller_1'] > comparison_vd['Controller_2'],
                                             'Controller_2',
                                             (np.where(comparison_vd['Controller_1'] == comparison_vd['Controller_2'],
                                                       'equal',
                                                       'Controller_1')))  # Comparison of controllers
Controller_1_better_performer = comparison_vd.loc[comparison_vd['Better Performer'] == 'Controller_1']
Controller_2_better_performer = comparison_vd.loc[comparison_vd['Better Performer'] == 'Controller_2']
Controller_1_points_vd = Controller_1_better_performer['Weight'].sum()  # Scores of Controller 1 are summed up
Controller_2_points_vd = Controller_2_better_performer['Weight'].sum()  # Scores of Controller 2 are summed up

###################Vq Cascaded Structure (Single Inverter Voltage Current Control)#######################
comparison_vq = pd.concat([df_vq_controller1, df_vq_controller2], axis=1)
comparison_vq.columns = ['Controller_1', 'Controller_2']
# Weight_Order = RMSE | Steady-State-Error | absolute Peak Value
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
print("###########################################################################################")
print("Results for Vd - cascaded control structure (outer voltage control, inner current control)")
print("###########################################################################################")
print(comparison_vd)
print()
print("Controller1_Vd_Points: ", Controller_1_points_vd)
print("Controller2_Vd_Points: ", Controller_2_points_vd)

print()
print("#############################################################################################")
print("Results for Vq - cascaded control structure (outer control: voltage, inner control: current)")
print("#############################################################################################")
print(comparison_vq)
print()
print("Controller1_Vq_Points: ", Controller_1_points_vq)
print("Controller2_Vq_Points: ", Controller_2_points_vq)

print()
print("#################################################")
print("Results for id - Single Inverter Current Control")
print("#################################################")
print(comparison_id)
print()
print("Controller1_id_Points: ", Controller_1_points_id)
print("Controller2_id_Points: ", Controller_2_points_id)

print()
print("#################################################")
print("Results for iq - Single Inverter Current Control")
print("#################################################")
print(comparison_iq)
print()
print("Controller1_iq_Points: ", Controller_1_points_iq)
print("Controller2_iq_Points: ", Controller_2_points_iq)

###############Overall Results - Output ###########

########Current Control (Idq)############

overall_Controller_1_points_current_control = Controller_1_points_id + Controller_1_points_iq
overall_Controller_2_points_current_control = Controller_2_points_id + Controller_2_points_iq
print()
print()
print()
print()
print()
print("Summary")
print("##############################")
print("Overall Results Current Control")
print("##############################")
print()
print("Controller_1_Points_overall: ", overall_Controller_1_points_current_control)
print("Controller_2_Points_overall: ", overall_Controller_2_points_current_control)
print()
if overall_Controller_2_points_current_control < overall_Controller_1_points_current_control:
    print("Controller 1 is the better performer for current control.The results still need to be")
    print("treated with caution, because depending on the purpose of the controller in the microgrid, ")
    print("individual metrics may be more important than others. ")
elif overall_Controller_2_points_current_control == overall_Controller_1_points_current_control:
    print("No controller is the better performer for current control.The results still need to be")
    print("treated with caution, because depending on the purpose of the controller in the microgrid,")
    print("individual metrics may be more important than others.")
else:
    print("Controller 2 is the better performer for current control.The results still need to be")
    print("treated with caution, because depending on the purpose of the controller in the microgrid,")
    print("individual metrics may be more important than others.")

################Cascaded Control Structure (Vdq)###########

overall_Controller_1_points_vd = Controller_1_points_vd + Controller_1_points_vq
overall_Controller_2_points_vq = Controller_2_points_vd + Controller_2_points_vq
print()
print("Summary")
print("####################################################")
print("Overall Results Voltage Control (Cascaded Structure)")
print("####################################################")
print()
print("Controller_1_Points_overall: ", overall_Controller_1_points_vd)
print("Controller_2_Points_overall: ", overall_Controller_2_points_vq)
print()
if overall_Controller_2_points_vq < overall_Controller_1_points_vd:
    print("Controller 1 is the better performer for voltage control.The results still need to be")
    print("treated with caution,because depending on the purpose of the controller in the microgrid, ")
    print("individual metrics may be more important than others. ")
elif overall_Controller_2_points_vq == overall_Controller_1_points_vd:
    print("No controller is the better performer for the voltage control.The results still need to be treated")
    print("with caution, because depending on the purpose of the controller in the microgrid,")
    print("individual metrics may be more important than others.")
else:
    print("Controller 2 is the better performer for voltage control.The results still need to be")
    print("treated with caution, because depending on the purpose of the controller in the microgrid,")
    print("individual metrics may be more important than others.")
