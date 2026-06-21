import argparse
from copy import deepcopy
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Average multiple checkpoints into one evaluation checkpoint.')
    parser.add_argument(
        '--checkpoints',
        nargs='+',
        help='Explicit checkpoint paths to average.')
    parser.add_argument(
        '--work-dir',
        help='Optional work dir containing iter_*.pth checkpoints.')
    parser.add_argument(
        '--iters',
        nargs='+',
        type=int,
        help='Iteration numbers to average when --work-dir is used.')
    parser.add_argument(
        '--output',
        required=True,
        help='Output checkpoint path.')
    return parser.parse_args()


def resolve_checkpoints(args):
    if args.checkpoints:
        paths = [Path(p).expanduser().resolve() for p in args.checkpoints]
    elif args.work_dir and args.iters:
        work_dir = Path(args.work_dir).expanduser().resolve()
        paths = [work_dir / f'iter_{iteration}.pth' for iteration in args.iters]
    else:
        raise ValueError(
            'Provide either --checkpoints or both --work-dir and --iters.')

    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            'Missing checkpoints:\n' + '\n'.join(missing))
    return paths


def average_state_dicts(state_dicts):
    averaged = {}
    keys = list(state_dicts[0].keys())

    for key in keys:
        tensors = [state_dict[key] for state_dict in state_dicts]
        first = tensors[0]
        if torch.is_floating_point(first):
            acc = first.detach().to(torch.float64).clone()
            for tensor in tensors[1:]:
                acc.add_(tensor.detach().to(torch.float64))
            averaged[key] = (acc / len(tensors)).to(first.dtype)
        else:
            averaged[key] = first.clone()
    return averaged


def main():
    args = parse_args()
    checkpoint_paths = resolve_checkpoints(args)

    checkpoints = [
        torch.load(path, map_location='cpu') for path in checkpoint_paths
    ]

    state_dicts = [checkpoint['state_dict'] for checkpoint in checkpoints]
    reference_keys = list(state_dicts[0].keys())
    for index, state_dict in enumerate(state_dicts[1:], start=1):
        if list(state_dict.keys()) != reference_keys:
            raise KeyError(
                f'State dict keys differ for checkpoint {checkpoint_paths[index]}')

    averaged = {
        'meta': deepcopy(checkpoints[0].get('meta', {})),
        'state_dict': average_state_dicts(state_dicts),
    }
    averaged['meta']['averaged_from'] = [str(path) for path in checkpoint_paths]

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(averaged, output_path)

    print('Averaged checkpoints:')
    for path in checkpoint_paths:
        print(f'  {path}')
    print(f'Wrote: {output_path}')


if __name__ == '__main__':
    main()
