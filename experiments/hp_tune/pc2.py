"""Helper functions for the communication with the high-performance computing
cluster 'Paderborn Center for Parallel Computing' (PC²)"""

import subprocess as sub
import pathlib
import time
import pandas as pd


def build_shell_script_lines(path, cluster, job_name, res_plan, execution_lines):
    cfg_id_d = {'oculus': '#CCS', 'noctua': '#SBATCH'}
    assert cluster in cfg_id_d, f'cluster "{cluster}" not supported'
    assert isinstance(path, pathlib.Path)
    log_path = path / 'logs'
    log_path.mkdir(parents=True, exist_ok=True)
    cfg_id = cfg_id_d[cluster]
    lines = ['#! /usr/bin/env zsh', '#! /bin/zsh', '',
             f'{cfg_id} -t {res_plan["duration"]}',
             f'{cfg_id} -o {log_path / "%reqid.log"}',
             f'{cfg_id} -N {job_name}',
             f'{cfg_id} --res=rset={res_plan["rset"]}'
             f':ncpus={res_plan["ncpus"]}'
             f':mem={res_plan["mem"]}'
             f':vmem={res_plan["vmem"]}',
             f'{cfg_id} -j', '']

    lines.extend(execution_lines if isinstance(execution_lines, list)
                 else [execution_lines])
    return [line + '\n' for line in lines]


def calculate_resources(duration=1, ncpus=6, memory=4, vmemory=8):
    # todo: Think of a more intelligent (adaptive) resource plan
    plan = {'duration': str(duration) + 'h',
            'rset': '1',
            'ncpus': str(ncpus),
            'mem': str(memory) + 'g',
            'vmem': str(vmemory) + 'g'
            }
    return plan


def create_n_run_script(name, content, dry=False):
    with open(name, 'w+') as f:
        f.writelines(content)
    sub.run(["chmod", "+x", name])  # Make script executable

    if not dry:
        # allocate and run, zB name = pc2_job_412643.sh
        sub.run(['ccsalloc', name])
        time.sleep(1)


def get_ccsinfo(user):
    """Returns the current ccs schedule as DataFrame"""
    ccsinfo = sub.run(['ccsinfo', '-s', f'--user={user}', '--raw'],
                      stdout=sub.PIPE).stdout.decode().strip('\n').split('\n')

    # def run(*popenargs,
    #        input=None, capture_output=False, timeout=None, check=False, **kwargs):

    info_lines = [l.strip().split() for l in ccsinfo]
    base_columns = ['jobid', 'jobname', 'user',
                    'state', 'time', 'allocated_time_days',
                    'allocated_time_hm', ]
    if any(len(l) > 9 for l in info_lines):
        columns = base_columns + ['efficiency_1', 'efficiency_2', 'resources']
    else:
        columns = base_columns + ['efficiency', 'resources']
    ccsinfo = pd.DataFrame(info_lines
                           if len(info_lines) > 0 and len(info_lines[0]) > 0
                           else None, columns=columns)

    return ccsinfo
