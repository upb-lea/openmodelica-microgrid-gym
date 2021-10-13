cfg = dict(lea_vpn_nodes=['lea-skynet', 'lea-picard', 'lea-barclay',
                          'lea-cyberdyne', 'webbah-ThinkPad-L380', 'LEA_WORK35'],
           STUDY_NAME='GEM_no_I_term_3',
           meas_data_folder='Json_to_MonogDB_GEM_no_I_term_3/',
           MONGODB_PORT=12001,
           loglevel='train',
           is_dq0=False,
           train_episode_length=2881,  # defines when in training the env is reset e.g. for exploring starts,
           # nothing -> Standard FeatureWrapper; past -> FeatureWrapper_pastVals; future -> FeatureWrapper_futureVals
           # I-controller -> DDPG as P-term + standard I-controller; no-I-term -> Pure DDPG without integrator
           env_wrapper='no-I-term'
           )
