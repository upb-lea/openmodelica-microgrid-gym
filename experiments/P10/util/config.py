cfg = dict(lea_vpn_nodes=['lea-skynet', 'lea-picard', 'lea-barclay',
                          'lea-cyberdyne', 'webbah-ThinkPad-L380', 'LEA_WORK35'],
           STUDY_NAME='P10_I_term_2',
           meas_data_folder='Json_to_MonogDB_P10_I_term_2/',
           MONGODB_PORT=12001,
           loglevel='setting',  # setting ~ config + return/learning curve (most is stored anyway, only effects in
           #           test saving stuff
           # test ~ setting + test-results (measurements)
           # train ~ test + training measurements
           is_dq0=True,

           # train_episode_length=2881,  # defines when in training the env is reset e.g. for exploring starts,

           # nothing -> Standard FeatureWrapper; past -> FeatureWrapper_pastVals; future -> FeatureWrapper_futureVals
           # I-controller -> DDPG as P-term + standard I-controller; no-I-term -> Pure DDPG without integrator
           env_wrapper='past'
           )
