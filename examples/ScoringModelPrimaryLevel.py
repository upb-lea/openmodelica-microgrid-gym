import pandas as pd
import numpy as np
#exec(open("Controller1_two_inverter_droop_safe_opt.py").read())
#exec(open("Controller2_two_inverter_droop_safe_opt.py").read())

df_vd_controller1= pd.read_pickle("./df_metrics_vd_controller1_droop.pkl")
df_vd_controller1['Value'].round(decimals=4)
df_vq_controller1=pd.read_pickle("./df_metrics_vq_controller1_droop.pkl")
df_vq_controller1['Value'].round(decimals=4)
df_slave_frequency_controller1=pd.read_pickle("./df_metrics_slave_f_controller1_droop.pkl")
df_slave_frequency_controller1['Value'].round(decimals=4)


###########################################################

df_vd_controller2= pd.read_pickle("./df_metrics_vd_controller2_droop.pkl")
df_vq_controller2=pd.read_pickle("./df_metrics_vq_controller2_droop.pkl")
df_slave_frequency_controller2=pd.read_pickle("./df_metrics_slave_f_controller2_droop.pkl")

#########################################################

########Vd0########

comparison_vd=pd.concat([df_vd_controller1, df_vd_controller2], axis=1)
comparison_vd.columns=['Controller_1', 'Controller_2']
comparison_vd= comparison_vd.drop(comparison_vd[(comparison_vd == 'No overshoot').any(1)].index)
comparison_vd['Controller_1'] = comparison_vd['Controller_1'].astype(float).round(4)
comparison_vd['Controller_2'] = comparison_vd['Controller_2'].astype(float).round(4)
comparison_vd['Better Performer'] = np.where(comparison_vd['Controller_1'] > comparison_vd['Controller_2'], 'Controller_2',
                                             (np.where(comparison_vd['Controller_1'] == comparison_vd['Controller_2'],
                                                        'equal', 'Controller_1')))
Controller_1_points_Vd=comparison_vd.loc[comparison_vd['Better Performer'] == 'Controller_1'].shape[0]
Controller_2_points_Vd=comparison_vd.loc[comparison_vd['Better Performer'] == 'Controller_2'].shape[0]

print()
print("###############################")
print("Results for Vd - Primary level")
print("###############################")
print(comparison_vd)
print()
print("Controller1_Vd_Points: ", Controller_1_points_Vd)
print("Controller2_Vd_Points: ", Controller_2_points_Vd)


########Vq0#########

comparison_vq=pd.concat([df_vq_controller1, df_vq_controller2], axis=1)
comparison_vq.columns=['Controller_1', 'Controller_2']
comparison_vq= comparison_vq.drop(comparison_vq[(comparison_vq == 'No overshoot').any(1)].index)
comparison_vq['Controller_1'] = comparison_vq['Controller_1'].astype(float).round(4)
comparison_vq['Controller_2'] = comparison_vq['Controller_2'].astype(float).round(4)
comparison_vq['Better Performer'] = np.where(comparison_vq['Controller_1'] > comparison_vq['Controller_2'], 'Controller_2',
                                             (np.where(comparison_vq['Controller_1'] == comparison_vq['Controller_2'],
                                                        'equal', 'Controller_1')))
Controller_1_points_Vq=comparison_vq.loc[comparison_vq['Better Performer'] == 'Controller_1'].shape[0]
Controller_2_points_Vq=comparison_vq.loc[comparison_vq['Better Performer'] == 'Controller_2'].shape[0]

print()
print("###############################")
print("Results for Vq - Primary level")
print("###############################")
print(comparison_vq)
print()
print("Controller1_Vq_Points: ", Controller_1_points_Vq)
print("Controller2_Vq_Points: ", Controller_2_points_Vq)



#######Slave_Frequency#######

comparison_slave_frequency=pd.concat([df_slave_frequency_controller1, df_slave_frequency_controller2], axis=1)
comparison_slave_frequency.columns=['Controller_1', 'Controller_2']
comparison_slave_frequency= comparison_slave_frequency.drop(comparison_slave_frequency[(comparison_slave_frequency=='No overshoot').any(1)].index)
comparison_slave_frequency['Controller_1'] = comparison_slave_frequency['Controller_1'].astype(float).round(4)
comparison_slave_frequency['Controller_2'] = comparison_slave_frequency['Controller_2'].astype(float).round(4)
comparison_slave_frequency['Better Performer'] = np.where(comparison_slave_frequency['Controller_1'] > comparison_slave_frequency['Controller_2'], 'Controller_2',
                                              (np.where(comparison_slave_frequency['Controller_1']==comparison_slave_frequency['Controller_2'],
                                                        'equal', 'Controller_1')))
Controller_1_points_slave_frequency=comparison_slave_frequency.loc[comparison_slave_frequency['Better Performer'] == 'Controller_1'].shape[0]
Controller_2_points_slave_frequency=comparison_slave_frequency.loc[comparison_slave_frequency['Better Performer'] == 'Controller_2'].shape[0]

print()
print("###########################")
print("Results for Slave Frequency")
print("###########################")
print(comparison_slave_frequency)
print()
print("Controller1_SlaveFrequency_Points: ",Controller_1_points_slave_frequency)
print("Controller2_SlaveFrequency_Points: ",Controller_2_points_slave_frequency)

######Overall Results######

overall_Controller_1_points= Controller_1_points_Vd + Controller_1_points_Vq + Controller_1_points_slave_frequency
overall_Controller_2_points= Controller_2_points_Vd + Controller_2_points_Vq + Controller_2_points_slave_frequency
print()
print()
print()
print("Summary")
print("#################################")
print("Overall Results for Primary level")
print("#################################")
print()

print()
print("Controller_1_Points_overall: ",overall_Controller_1_points)
print("Controller_2_Points_overall: ", overall_Controller_2_points)
print()
if overall_Controller_2_points<overall_Controller_1_points:
    print("Controller 1 is the better performer.The results still need to be treated with caution,")
    print("because depending on the purpose of the controller in the microgrid, ")
    print("individual metrics may be more important than others. ")
elif overall_Controller_2_points==overall_Controller_1_points:
    print("No controller is better than the other.The results still need to be treated with caution,")
    print("because depending on the purpose of the controller in the microgrid,")
    print("individual metrics may be more important than others.")
else:
    print("Controller 2 is the better performer.The results still need to be treated with caution,")
    print("because depending on the purpose of the controller in the microgrid,")
    print("individual metrics may be more important than others.")




#####TO-Do

#----Sys Fehlermeldung einfügen (auch vielleicht schreiben sonst grenzen anpassen)
#---- befehl  benutzen
#---- auf inner level gehen