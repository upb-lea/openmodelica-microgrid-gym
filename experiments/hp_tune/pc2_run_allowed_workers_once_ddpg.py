"""Allocate jobs executing a certain cmd endlessly. Make sure not to surpass
the allowed cpu core limit"""

import os
import pathlib
import time
import uuid

from experiments.hp_tune.util import pc2
# config
from experiments.hp_tune.util.config import cfg

USER = os.getenv('USER')
ALLOWED_MAX_CPU_CORES = 512

# resources request
job_resource_plan = {
    'duration': 12,  # in hours
    'ncpus': 2,
    'memory': 12,
    'vmemory': 16,
}

MAX_WORKERS = ALLOWED_MAX_CPU_CORES // job_resource_plan['ncpus']
STUDY_NAME = cfg['STUDY_NAME']
NUMBER_INTERATIONS = 1

def main():
    print('Start slavedriving loop..')
    print('Will start MAX_WORKERS and terminate.')
    old_ccsinfo_counts = None
    for _ in range(MAX_WORKERS):
        job_files_path = pathlib.Path(
            f"/scratch/hpc-prf-reinfl/weber/OMG/ccs_job_files/{STUDY_NAME}")  # SCRATCH = $PC2PFS/hpc_....re/OMG_prjecet
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
            # n_workers = MAX_WORKERS - total_busy
            # print(f'Start {n_workers} workers:')
            # for w in range(n_workers):
            jobid = str(uuid.uuid4()).split('-')[0]
            cluster = "oculus"
            job_name = job_files_path / f"pc2_job_{jobid}.sh"
            res_plan = pc2.calculate_resources(**job_resource_plan)

            execution_line = "PYTHONPATH=$HOME/openmodelica-microgrid-gym/ " \
                             "python $HOME/openmodelica-microgrid-gym/experiments/hp_tune/hp_tune_ddpg_objective.py -n 3"

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

        # print('sleep..', end='\r')
        # time.sleep(120)
    print('Finished, need resatart to schedule again!..', end='\r')


if __name__ == '__main__':
    main()
