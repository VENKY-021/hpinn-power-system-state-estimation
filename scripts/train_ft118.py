"""IEEE 118-bus physics-informed networks: composite-loss training started from the supervised network of the same seed
(pre-training followed by physics-informed fine-tuning), because the from-scratch run did not converge within the epoch budget.
Runs the jobs in parallel single-thread, below-normal-priority processes."""
import os
import sys
import time
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
NPROC = int(os.environ.get('NPROC', '4'))
JOBS = [('pinn', 1), ('pinn', 2), ('pinn_fixed', 0), ('pinn_noconf', 0), ('pinn_nodesc', 0), ('pinn_nomeas', 0), ('pinn_nophys', 0)]
EPOCHS, LR = '30', '3e-4'


def done(v, s):
    return os.path.exists(os.path.join(HERE, '..', 'results', 'models', 'ieee118', f'{v}_s{s}.pt'))


if __name__ == '__main__':
    queue = [j for j in JOBS if not done(*j)]
    running, t0 = [], time.time()
    log = open(os.path.join(HERE, '..', 'results', 'train_ft118.log'), 'a')
    while queue or running:
        running = [(p, j) for p, j in running if p.poll() is None]
        while queue and len(running) < NPROC:
            v, s = queue.pop(0)
            env = dict(os.environ, NN_THREADS='1', NN_INIT=f'dnn_s{s}', NN_LR=LR, NN_MAX_EPOCHS=EPOCHS)
            p = subprocess.Popen([sys.executable, '-u', os.path.join(HERE, 'nn.py'), 'ieee118', v, str(s), '-', ''], stdout=log, stderr=subprocess.STDOUT, env=env)
            running.append((p, (v, s)))
        time.sleep(2)
    log.write(f'ALL DONE {time.time() - t0:.0f}s\n')
    log.close()
