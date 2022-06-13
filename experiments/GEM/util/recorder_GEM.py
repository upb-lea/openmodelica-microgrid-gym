import json
from os import makedirs

import sshtunnel
from pymongo import MongoClient

from experiments.GEM.util.config import cfg

MONGODB_PORT = cfg['MONGODB_PORT']  # 12001


class Recorder:

    def __init__(self, node, database_name):
        """
        Class to record measured data to mongo database using pymongo
        Depending on the node we are operating at it connects via ssh to
         - in lea_vpn: to cyberdyne port 12001
         - else: assume pc2 node -> connect to frontend
         and stores data to mongoDB at port MONGODB_PORT ( =12001).
         HINT: From pc2 frontend permanent tunnel from cyberdyne port 12001 to frontend 12001
         is needed (assuming Mongod-Process running on cyberdyne
         :params node: platform.uname().node
         :params database_name: string for the database name to store data in
        """
        self.node = node
        self.save_count = 0

        if self.node in cfg['lea_vpn_nodes']:
            self.server_name = 'lea38'
            self.tun_cfg = {'remote_bind_address': ('127.0.0.1',
                                                    MONGODB_PORT)}
            self.save_folder = cfg['meas_data_folder']
        else:
            # assume we are on a node of pc2 -> connect to frontend and put data on prt 12001
            # from there they can be grep via permanent tunnel from cyberdyne
            self.server_name = 'fe.pc2.uni-paderborn.de'
            self.tun_cfg = {'remote_bind_address': ('127.0.0.1',
                                                    MONGODB_PORT),
                            'ssh_username': 'webbah'}

            self.save_folder = '/scratch/hpc-prf-reinfl/weber/OMG/' + cfg['meas_data_folder']

        self.database_name = database_name
        makedirs(self.save_folder, exist_ok=True)
        # pathlib.Path(self.save_folder.mkdir(exist_ok=True))

    def save_to_mongodb(self, col: str = ' trails', data=None):
        """
        Stores data to database in document col
        """
        with sshtunnel.open_tunnel(self.server_name, **self.tun_cfg) as tun:
            with MongoClient(f'mongodb://localhost:{tun.local_bind_port}/') as client:
                db = client[self.database_name]
                trial_coll = db[col]  # get collection named col
                trial_coll.insert_one(data)

    def save_to_json(self, col: str = ' trails', data=None):
        """
        Stores data to json file in specified directory. From there the data can be grept by another process
        and can be stored to a DB via ssh
        To distinguish the files of one trail a save_count is incremented and added to the filename
        """

        with open(self.save_folder + self.database_name + '_' + col + '_' + str(self.save_count) + '.json',
                  'w') as outfile:
            json.dump(data, outfile)

        self.save_count += 1
