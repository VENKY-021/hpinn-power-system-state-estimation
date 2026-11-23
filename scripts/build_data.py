"""Build all datasets of one benchmark system: training/validation/calibration mixture and the test scenarios."""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import get_grid, get_states, make_set, scenarios, MIX, spec

SIZES = {'ieee14': dict(train=20000, val=3000, cal=1500, test=600, mix=1000),
         'ieee30': dict(train=30000, val=3000, cal=1500, test=600, mix=1000),
         'ieee118': dict(train=40000, val=3000, cal=1500, test=400, mix=800)}


def build(name, uni=False):
    K = SIZES[name]
    pre = 'u_' if uni else ''
    op = 'uni' if uni else 'id'
    mix = dict(MIX, op=op)
    t0 = time.time()
    make_set(name, K['train'], mix, 100, pre + 'train')
    make_set(name, K['val'], mix, 200, pre + 'val')
    make_set(name, K['cal'], mix, 300, pre + 'cal', with_obs=True)
    print(name, 'train/val/cal done', f'{time.time() - t0:.0f}s', flush=True)
    base = get_states(name, K['test'], op, 0.0, 400)
    shift = get_states(name, K['test'], 'shift', 0.0, 500)
    seed = 1000
    for tag, sp in scenarios().items():
        seed += 10
        sp = dict(sp)
        if sp['op'] == 'id':
            sp['op'] = op
        if tag == 'mix':
            make_set(name, K['mix'], mix, seed, pre + 'test_mix', with_obs=True)
        elif tag.startswith('param'):
            make_set(name, K['test'], sp, seed, pre + 'test_' + tag, with_obs=True)
        elif tag.startswith('shift'):
            if uni:
                continue
            make_set(name, K['test'], sp, seed, pre + 'test_' + tag, states=shift, with_obs=True)
        else:
            make_set(name, K['test'], sp, seed, pre + 'test_' + tag, states=base, with_obs=True)
        print(' ', tag, f'{time.time() - t0:.0f}s', flush=True)
    print(name, 'all datasets done', f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    build(sys.argv[1], uni=len(sys.argv) > 2 and sys.argv[2] == 'uni')
