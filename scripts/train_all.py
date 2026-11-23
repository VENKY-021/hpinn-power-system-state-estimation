"""Train every model of one benchmark system in parallel (3 single-thread, below-normal-priority processes)."""
import os, sys, time, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
LAB = {'ieee14': (200, 1000, 4000), 'ieee30': (300, 1500, 6000), 'ieee118': (400, 2000, 8000)}


def jobs(system, prefix=''):
    J = []
    for s in range(5):
        for v in ('pinn', 'dnn'):
            J.append((v, s, '-'))
    if prefix:
        return [(v, s, n) for v, s, n in J if s < 3]
    for v in ('pinn_fixed', 'pinn_noconf', 'pinn_nodesc', 'pinn_nomeas', 'pinn_nophys'):
        for s in range(2):
            J.append((v, s, '-'))
    for n in LAB[system]:
        for v in ('dnn', 'pinn'):
            for s in range(2):
                J.append((v, s, str(n)))
    return J


if __name__ == '__main__':
    system = sys.argv[1]
    prefix = 'u_' if len(sys.argv) > 2 and sys.argv[2] == 'uni' else ''
    nproc = int(os.environ.get('NPROC', '3'))
    env = dict(os.environ, NN_THREADS='1')
    log = open(os.path.join(HERE, '..', 'results', f'train_{prefix}{system}.log'), 'a')
    queue, running, t0 = jobs(system, prefix), [], time.time()
    # skip models that exist
    def done(v, s, n):
        nm = f'{prefix}{v}' + (f'_L{n}' if n != '-' else '') + f'_s{s}'
        return os.path.exists(os.path.join(HERE, '..', 'results', 'models', system, nm + '.pt'))
    queue = [j for j in queue if not done(*j)]
    while queue or running:
        running = [(p, j) for p, j in running if p.poll() is None]
        while queue and len(running) < nproc:
            j = queue.pop(0)
            p = subprocess.Popen([sys.executable, '-u', os.path.join(HERE, 'nn.py'), system, j[0], str(j[1]), j[2], prefix],
                                 stdout=log, stderr=subprocess.STDOUT, env=env)
            running.append((p, j))
        time.sleep(2)
    log.write(f'ALL DONE {system} {time.time() - t0:.0f}s\n'); log.close()
