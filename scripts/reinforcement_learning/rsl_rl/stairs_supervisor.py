"""Host-side persistent controller: finish the active run before new stages."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
CONTAINER_ROOT = '/workspace/isaaclab_extension_template'
RUN_ROOT = ROOT / 'logs/rsl_rl/atdog_dog2_stairs_him'
parser = argparse.ArgumentParser()
parser.add_argument('--wait-run', required=True)
parser.add_argument('--target', type=int, default=3300)
parser.add_argument('--iterations', type=int, default=800)
parser.add_argument('--rounds', type=int, default=12)
args = parser.parse_args()
output = ROOT / 'logs/stairs_11cm_campaign'
output.mkdir(parents=True, exist_ok=True)
state = dict(status='waiting', wait_run=args.wait_run, target=args.target, rounds=[])


def save():
    temporary = output / 'state.tmp'
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(output / 'state.json')


def running():
    result = subprocess.run(['docker', 'exec', 'robot-lab', 'pgrep', '-af', '^/workspace/isaaclab/_isaac_sim/kit/python/bin/python3 scripts/reinforcement_learning/rsl_rl/train.py'], capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    return bool(result.stdout.strip())


def container_path(path):
    return CONTAINER_ROOT + '/' + str(Path(path).relative_to(ROOT))


def worker(mode, checkpoint, folder, **options):
    if running():
        raise RuntimeError('Another train.py process is active; refusing concurrent GPU training')
    command = ['docker', 'exec', 'robot-lab', '/workspace/isaaclab/isaaclab.sh', '-p',
               CONTAINER_ROOT + '/scripts/reinforcement_learning/rsl_rl/stairs_stage.py',
               '--mode', mode, '--checkpoint', container_path(checkpoint), '--output', container_path(folder)]
    for key, value in options.items():
        command += ['--' + key, str(value)]
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'console.log').open('w') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def evaluate(checkpoint, label):
    results = []
    for seed in (42, 137, 2026):
        folder = output / label / f'eval_{seed}'
        state.update(status='evaluating', checkpoint=str(checkpoint), evaluation=str(folder))
        save()
        worker('eval', checkpoint, folder, num_envs=64, seed=seed)
        result = json.loads((folder / 'evaluation.json').read_text())
        results.append(result)
    return results


def passed(results):
    return (sum(r['episodes'] for r in results) >= 300 and
            all(r['success_rate'] >= .9 and r['fall_rate'] <= .05 for r in results))


def latest_checkpoint(folder):
    return max(folder.glob('model_*.pt'), key=lambda path: int(path.stem.split('_')[-1]))


def train_metrics(folder):
    # Read scalars in the installed TensorBoard environment after training.
    code = '''import json, statistics, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
e=EventAccumulator(sys.argv[1]); e.Reload()
print(json.dumps({k: statistics.mean(x.value for x in e.Scalars(k)[-100:]) for k in e.Tags()['scalars'] if k.startswith(('Episode_Reward/', 'Metrics/', 'Episode_Termination/'))}))'''
    result = subprocess.run(['docker', 'exec', 'robot-lab', '/workspace/isaaclab/_isaac_sim/python.sh', '-c', code, container_path(folder)], capture_output=True, text=True, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def main():
    save()
    initial = RUN_ROOT / args.wait_run
    checkpoint = initial / f'model_{args.target}.pt'
    while running():
        # Do not touch the current process, even after the target file appears.
        state['target_checkpoint_present'] = checkpoint.exists()
        save()
        time.sleep(30)
    if not checkpoint.exists():
        raise RuntimeError(f'Training stopped before the required checkpoint: {checkpoint}')
    baseline = evaluate(checkpoint, 'baseline')
    state['baseline'] = baseline
    save()
    if passed(baseline):
        state.update(status='complete', best_checkpoint=str(checkpoint))
        save()
        return
    best_score = sum(r['success_rate'] for r in baseline) / 3
    best_checkpoint = checkpoint
    plateau = 0
    metrics = train_metrics(initial)
    for index in range(args.rounds):
        # A short easier bridge, then increasing exposure to the 11 cm target.
        low, high = ((.06, .09) if index == 0 else (.07, .11) if index == 1 else (.09, .11))
        profile = 'balanced'
        if metrics.get('Episode_Termination/illegal_contact', 0) > .12:
            profile = 'support'
        elif abs(metrics.get('Episode_Reward/stair_dynamic_swing', 0)) > .6:
            profile = 'clearance'
        folder = RUN_ROOT / f'campaign11_{time.strftime("%Y-%m-%d_%H-%M-%S")}_{index:02d}'
        state.update(status='training', checkpoint=str(checkpoint), active_run=str(folder))
        save()
        worker('train', checkpoint, folder, iterations=args.iterations, num_envs=5500,
               low=low, high=high, profile=profile, seed=42 + index)
        candidate = latest_checkpoint(folder)
        metrics = train_metrics(folder)
        results = evaluate(candidate, f'round_{index:02d}')
        score = sum(r['success_rate'] for r in results) / 3
        entry = dict(run=str(folder), checkpoint=str(candidate), height_range=[low, high],
                     profile=profile, training_metrics=metrics, evaluations=results)
        state['rounds'].append(entry)
        if score > best_score:
            best_score, best_checkpoint = score, candidate
            plateau = 0
        else:
            plateau += 1
        state.update(best_checkpoint=str(best_checkpoint), best_success=best_score)
        if passed(results):
            state['status'] = 'complete'
            save()
            return
        # Preserve the best measured model when a candidate regresses markedly.
        checkpoint = best_checkpoint if score < best_score - .05 else candidate
        save()
        if plateau >= 3 and index >= 3:
            state.update(status='needs_review', reason='Three evaluations without improved 11 cm success; best model preserved')
            save()
            return
    state.update(status='needs_review', reason='Campaign round limit reached without meeting all acceptance criteria')
    save()


if __name__ == '__main__':
    lock = (output / 'campaign.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        main()
    except BaseException as error:
        state.update(status='error', error=f'{type(error).__name__}: {error}')
        save()
        raise
