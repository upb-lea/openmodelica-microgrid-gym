import json
import os
import platform
import re
import time

import numpy as np

import sshtunnel
from pymongo import MongoClient
# from experiments.hp_tune.util.config import cfg
from experiments.GEM.util.config import cfg

print('Log Config: GEM!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')


class Reporter:

    def __init__(self):
        """
        Greps json data which is stored in the cfg[meas_data_folder] and sends it to mongoDB
        on cyberdyne (lea38) via sshtunnel on port MONGODB_PORT
        """

        MONGODB_PORT = cfg['MONGODB_PORT']

        node = platform.uname().node

        if node in cfg['lea_vpn_nodes']:
            self.server_name = 'lea38'
            self.tun_cfg = {'remote_bind_address': ('127.0.0.1',
                                                    MONGODB_PORT)}
            self.save_folder = './' + cfg['meas_data_folder']
        else:
            # assume we are on a node of pc2 -> connect to frontend and put data on prt 12001
            # from there they can be grep via permanent tunnel from cyberdyne
            self.server_name = 'fe.pc2.uni-paderborn.de'
            self.tun_cfg = {'remote_bind_address': ('127.0.0.1',
                                                    MONGODB_PORT),
                            'ssh_username': 'webbah'}

            self.save_folder = '/scratch/hpc-prf-reinfl/weber/OMG/' + cfg['meas_data_folder']

    def save_to_mongodb(self, database_name: str, col: str = ' trails', data=None):
        """
        Stores data to database in document col
        """
        with sshtunnel.open_tunnel(self.server_name, **self.tun_cfg) as tun:
            with MongoClient(f'mongodb://localhost:{tun.local_bind_port}/') as client:
                db = client[database_name]
                trial_coll = db[col]  # get collection named col
                trial_coll.insert_one(data)

    def oldest_file_in_tree(self, extension=".json"):
        """
        Returns the oldest file-path string
        """
        print(os.getcwd())
        return min(
            (os.path.join(dirname, filename)
             for dirname, dirnames, filenames in os.walk(self.save_folder)
             for filename in filenames
             if filename.endswith(extension)),
            key=lambda fn: os.stat(fn).st_mtime)

    def oldest_file_with_name_in_tree(self, count_number_to_find, extension=".json"):
        """
        Returns the oldest file-path string

        :param count_number_to_find: List of count_numbers to find and store instead of storing all
        """
        print(os.getcwd())
        return min(
            (os.path.join(dirname, filename)
             for dirname, dirnames, filenames in os.walk(self.save_folder)
             for filename in filenames
             if filename.endswith(str(count_number_to_find) + extension)),
            key=lambda fn: os.stat(fn).st_mtime)

    def json_to_mongo_via_sshtunnel(self, file_name_to_store=None):

        if not len(os.listdir(self.save_folder)) == 0:

            if file_name_to_store is None:
                try:
                    oldest_file_path = self.oldest_file_in_tree()
                except(ValueError) as e:
                    print('Folder seems empty or no matching data found!')
                    print(f'ValueError{e}')
                    print('Empty directory! Go to sleep for 5 minutes!')
                    time.sleep(5 * 60)
                    return
            else:
                oldest_file_path = file_name_to_store

            with open(oldest_file_path) as json_file:
                data = json.load(json_file)

            successfull = False
            retry_counter = 0

            while not successfull:
                try:
                    now = time.time()
                    if os.stat(oldest_file_path).st_mtime < now - 60:
                        self.save_to_mongodb(database_name=data['Database name'],
                                             col='Trial_number_' + data['Trial number'], data=data)
                        print('Reporter: Data stored successfully to MongoDB and will be removed locally!')
                        os.remove(oldest_file_path)
                        successfull = True
                except (sshtunnel.BaseSSHTunnelForwarderError) as e:
                    wait_time = np.random.randint(1, 60)
                    retry_counter += 1
                    if retry_counter > 10:
                        print('Stopped after 10 connection attempts!')
                        raise e
                    print(f'Reporter: Could not connect via ssh to frontend, retry in {wait_time} s')
                    time.sleep(wait_time)

        else:
            print('Empty directory! Go to sleep for 5 minutes!')
            time.sleep(5 * 60)


if __name__ == "__main__":

    reporter = Reporter()
    print("Starting Reporter for logging from local savefolder to mongoDB")

    file_ending_number = [178, 179]

    print(f"Searching for files in directory with number ending on {file_ending_number}")

    # print(reporter.oldest_file_in_tree())
    while True:
        # reporter.json_to_mongo_via_sshtunnel()

        for number in file_ending_number:
            try:
                oldest_named_file_path = reporter.oldest_file_with_name_in_tree(number)
                print(oldest_named_file_path)

                reporter.json_to_mongo_via_sshtunnel(oldest_named_file_path)

            except(ValueError) as e:
                print(f'No file with number {number} ending')
                print(f'ValueError{e}')
                print('Go to sleep for 5 seconds and go on with next number!')
                time.sleep(5)
