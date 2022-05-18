"""Allocate jobs executing a certain cmd endlessly. Make sure not to surpass
the allowed cpu core limit"""

import os
import pathlib
import uuid
import time

import optuna
from optuna.samplers import TPESampler

from experiments.hp_tune.util import pc2
from experiments.P10.util.config import cfg

# config
USER = os.getenv('USER')
ALLOWED_MAX_CPU_CORES = 300  # 512
STUDY_NAME = cfg['STUDY_NAME']
DB_NAME = 'optuna'
# resources request
job_resource_plan = {
    'duration': 24,  # in hours
    'ncpus': 2,
    'memory': 12,
    'vmemory': 16,
}

MAX_WORKERS = ALLOWED_MAX_CPU_CORES // job_resource_plan['ncpus']

PC2_LOCAL_PORT2MYSQL = 11998
SERVER_LOCAL_PORT2MYSQL = 3306


def main():
    started_workers = 0
    print('Start slavedriving loop..')
    old_ccsinfo_counts = None
    while True:

        creds_path = f'{os.getenv("HOME")}/creds/optuna_mysql'

        with open(creds_path, 'r') as f:
            optuna_creds = ':'.join([s.strip(' \n') for s in f.readlines()])

        study = optuna.create_study(
            storage=f'mysql+pymysql://{optuna_creds}@localhost:{PC2_LOCAL_PORT2MYSQL}/{DB_NAME}',
            # storage=f'postgresql://{optuna_creds}@localhost:{port}/{DB_NAME}',
            sampler=TPESampler(n_startup_trials=2500), study_name=STUDY_NAME,
            load_if_exists=True,
            direction='maximize')

        complete_trials = len([t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE])
        print(f'Completed trials in study: {complete_trials}')
        if complete_trials > 12000:
            print('Maximal completed trials reached - STOPPING')
            break

        job_files_path = pathlib.Path(
            f"/scratch/hpc-prf-reinfl/weber/P10/ccs_job_files/{STUDY_NAME}")  # SCRATCH = $PC2PFS/hpc_....re/OMG_prjecet
        job_files_path.mkdir(parents=False, exist_ok=True)

        # read ccsinfo
        ccsinfo = pc2.get_ccsinfo(USER)
        ccsinfo_state_counts = ccsinfo.state.value_counts()
        ccs_running = ccsinfo_state_counts.get('ALLOCATED', 0)
        ccs_planned = ccsinfo_state_counts.get('PLANNED', 0)
        total_busy = ccs_running + ccs_planned
        if not ccsinfo_state_counts.equals(old_ccsinfo_counts):
            print("\n## ccs summary ##")
            print(f"Running: {ccs_running}")
            print(f"Planned : {ccs_planned}")
            print(f"Total busy workers (ccs): {total_busy}")

        if total_busy < MAX_WORKERS:
            #  call workers to work
            n_workers = MAX_WORKERS - total_busy
            print(f'Start {n_workers} workers:')
            for w in range(n_workers):
                started_workers += 1
                jobid = str(uuid.uuid4()).split('-')[0]
                cluster = "oculus"
                job_name = job_files_path / f"pc2_job_{jobid}.sh"
                res_plan = pc2.calculate_resources(**job_resource_plan)

                execution_line = "PYTHONPATH=$HOME/openmodelica-microgrid-gym/ " \
                                 "python $HOME/openmodelica-microgrid-gym/experiments/P10/hp_tune_ddpg_objective_P10.py -n 1"

                print(f'Start job {jobid} ..')
                pc2.create_n_run_script(
                    job_name,
                    pc2.build_shell_script_lines(job_files_path, cluster,
                                                 job_name, res_plan,
                                                 execution_line),
                    dry=False)
                print('sleep 10s for better DB interaction', end='\r')
                time.sleep(10)

        old_ccsinfo_counts = ccsinfo_state_counts

        print('sleep..', end='\r')
        time.sleep(300)


if __name__ == '__main__':
    main()
