import pandas as pd
import numpy as np
exec(open("Controller1_two_inverter_droop_safe_opt.py").read())
exec(open("Controller2_two_inverter_droop_safe_opt.py").read())

df_vd0_controller1= pd.read_pickle("./df_metrics_vd0_controller1_droop.pkl")
df_vd0_controller1['Value'].round(decimals=4)
df_vq0_controller1=pd.read_pickle("./df_metrics_vq0_controller1_droop.pkl")
df_vq0_controller1['Value'].round(decimals=4)
df_slave_frequency_controller1=pd.read_pickle("./df_metrics_slave_f_controller1_droop.pkl")
df_slave_frequency_controller1['Value'].round(decimals=4)


###########################################################

df_vd0_controller2= pd.read_pickle("./df_metrics_vd0_controller2_droop.pkl")
df_vq0_controller2=pd.read_pickle("./df_metrics_vq0_controller2_droop.pkl")
df_slave_frequency_controller2=pd.read_pickle("./df_metrics_slave_f_controller2_droop.pkl")

#########################################################

########Vd0########

comparison_vd0=pd.concat([df_vd0_controller1, df_vd0_controller2], axis=1)
comparison_vd0.columns=['Controller_1', 'Controller_2']
comparison_vd0= comparison_vd0.drop(comparison_vd0[(comparison_vd0=='No overshoot').any(1)].index)
comparison_vd0['Controller_1'] = comparison_vd0['Controller_1'].astype(float).round(4)
comparison_vd0['Controller_2'] = comparison_vd0['Controller_2'].astype(float).round(4)
comparison_vd0['Better Performer'] = np.where(comparison_vd0['Controller_1'] > comparison_vd0['Controller_2'], 'Controller_2',
                                              (np.where(comparison_vd0['Controller_1']==comparison_vd0['Controller_2'],
                                                        'equal', 'Controller_1')))
Controller_1_points_Vd0=comparison_vd0.loc[comparison_vd0['Better Performer'] == 'Controller_1'].shape[0]
Controller_2_points_Vd0=comparison_vd0.loc[comparison_vd0['Better Performer'] == 'Controller_2'].shape[0]

print()
print("###############")
print("Results for Vd0")
print("###############")
print(comparison_vd0)
print()
print("Controller1_Vd0_Points: ",Controller_1_points_Vd0)
print("Controller2_Vd0_Points: ",Controller_2_points_Vd0)


########Vq0#########

comparison_vq0=pd.concat([df_vq0_controller1, df_vq0_controller2], axis=1)
comparison_vq0.columns=['Controller_1', 'Controller_2']
comparison_vq0= comparison_vq0.drop(comparison_vq0[(comparison_vq0=='No overshoot').any(1)].index)
comparison_vq0['Controller_1'] = comparison_vq0['Controller_1'].astype(float).round(4)
comparison_vq0['Controller_2'] = comparison_vq0['Controller_2'].astype(float).round(4)
comparison_vq0['Better Performer'] = np.where(comparison_vq0['Controller_1'] > comparison_vq0['Controller_2'], 'Controller_2',
                                              (np.where(comparison_vq0['Controller_1']==comparison_vq0['Controller_2'],
                                                        'equal', 'Controller_1')))
Controller_1_points_Vq0=comparison_vq0.loc[comparison_vq0['Better Performer'] == 'Controller_1'].shape[0]
Controller_2_points_Vq0=comparison_vq0.loc[comparison_vq0['Better Performer'] == 'Controller_2'].shape[0]

print()
print("###############")
print("Results for Vq0")
print("###############")
print(comparison_vq0)
print()
print("Controller1_Vq0_Points: ",Controller_1_points_Vq0)
print("Controller2_Vq0_Points: ",Controller_2_points_Vq0)



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
print(comparison_vq0)
print()
print("Controller1_SlaveFrequency_Points: ",Controller_1_points_Vq0)
print("Controller2_SlaveFrequency_Points: ",Controller_2_points_Vq0)

######Overall Results######

overall_Controller_1_points=Controller_1_points_Vd0+Controller_1_points_Vq0
overall_Controller_2_points=Controller_2_points_Vd0+Controller_2_points_Vq0
print()
print("###########################")
print("Overall Results")
print("###########################")
print()
print("Note: If the scores of Controller 1 and 2 are equal, the scores of the slave frequency comparison are added to the scores,")
print("although the frequency does not physically exist and is therefore a poor indicator for performance. Further details can be")
print("found in the related paper 'Development of a Testing Framework for Intelligent Microgrid Control'.")
print()
print("Controller_1_Points_overall: ",overall_Controller_1_points)
print("Controller_2_Points_overall: ", overall_Controller_2_points)
print()
if overall_Controller_2_points<overall_Controller_1_points:
    print("Controller 1 is the better performer.The results still need to be treated with caution,")
    print("because depending on the purpose of the controller in the microgrid, ")
    print("individual metrics may be more important than others. ")
elif overall_Controller_2_points==overall_Controller_1_points:
    overall_Controller_1_points=overall_Controller_1_points+Controller_1_points_slave_frequency
    overall_Controller_2_points=overall_Controller_2_points+Controller_2_points_slave_frequency
    print("The scores of the slave frequency comparison are added to the scores.")
    if overall_Controller_2_points<overall_Controller_1_points:
        print("Controller_1_Points (Points for Slave Frequency included): ", overall_Controller_1_points)
        print("Controller_2_Points (Points for Slave Frequency included): ", overall_Controller_2_points)
        print("Controller 1 is the better performer.The results still need to be treated with caution,")
        print("because depending on the purpose of the controller in the microgrid,")
        print("individual metrics may be more important than others." )
    elif overall_Controller_2_points==overall_Controller_1_points:
        print("No controller is better than the other")
    else:
        print("Controller 2 is the better performer.The results still need to be treated with caution,")
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