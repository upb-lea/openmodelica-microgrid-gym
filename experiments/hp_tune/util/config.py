cfg = dict(lea_vpn_nodes=['lea-skynet', 'lea-picard', 'lea-barclay',
                          'lea-cyberdyne', 'webbah-ThinkPad-L380', 'LEA_WORK35'],

           # STUDY_NAME='PC2_TD3_Vctrl_single_inv_2',
           # STUDY_NAME='PC2_DDPG_Vctrl_single_inv_23_added_Past_vals',
           STUDY_NAME='PC2_DDPG_Vctrl_single_inv_HPO_noI_term_study_25',
           # STUDY_NAME='PC2_DDPG_Vctrl_single_inv_PastValHPO_without_phaseFeature',
           meas_data_folder='Json_to_MonogDB_study_24/',
           MONGODB_PORT=12001,
           loglevel='test',
           is_dq0=True,
           train_episode_length=2881,  # defines when in training the env is reset e.g. for exploring starts,
           # nothing -> Standard FeatureWrapper; past -> FeatureWrapper_pastVals; future -> FeatureWrapper_futureVals
           env_wrapper='no-I-term'
           )
