"""Command-line interface for cfr.

Entry point declared in pyproject.toml as ``cfr = "cfr._cli:main"``.
The legacy script ``bin/cfr`` delegates here for backwards compatibility.
"""

from __future__ import annotations

import argparse

import cfr


def main() -> None:
    parser = argparse.ArgumentParser(
        description='''
========================================================================================
 cfr: a scripting system for CFR
----------------------------------------------------------------------------------------
 Usage example for DA:
    cfr da -c config.yml -vb -s 1 2 -r
    # -c config.yml: run the reconstruction job according to config.yml
    # -vb: output the verbose runtime information
    # -s 1 2: set seeds as integers from 1 to 2
    # -r: run the Monte-Carlo iterations for PDA

 Usage example for GraphEM:
    cfr graphem -c config.yml -vb
    # -c config.yml: run the reconstruction job according to config.yml
    # -vb: output the verbose runtime information
========================================================================================
        ''',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        '-v', '--version',
        action='version',
        version='%(prog)s version: {}'.format(cfr.__version__),
    )

    subparsers = parser.add_subparsers(help='running mode')
    subparsers.dest = 'mode'

    # DA
    parser_da = subparsers.add_parser('da', help='run a DA-based reconstruction')
    parser_da.add_argument('-c', '--config', required=True,
                           help='path of the config YAML file')
    parser_da.add_argument('-vb', '--verbose',
                           action=argparse.BooleanOptionalAction,
                           help='output the verbose runtime information')
    parser_da.add_argument('-s', '--seeds', nargs='*', default=None,
                           help='the start and end of the random seeds for reconstruction')
    parser_da.add_argument('-r', '--run',
                           action=argparse.BooleanOptionalAction,
                           help='prepare the job without running Monte-Carlo')

    # GraphEM
    parser_graphem = subparsers.add_parser('graphem',
                                           help='run a GraphEM-based reconstruction')
    parser_graphem.add_argument('-c', '--config', required=True,
                                help='path of the config YAML file')
    parser_graphem.add_argument('-vb', '--verbose',
                                action=argparse.BooleanOptionalAction,
                                help='output the verbose runtime information')

    args = parser.parse_args()

    job = cfr.ReconJob()
    if args.mode == 'da':
        if args.seeds is None:
            seeds = None
        elif len(args.seeds) == 1:
            seeds = int(args.seeds[0])
        elif len(args.seeds) == 2:
            seeds = list(range(int(args.seeds[0]), int(args.seeds[-1]) + 1))
        else:
            raise ValueError('Wrong number of seeds')
        job.run_da_cfg(args.config, seeds=seeds, run_mc=args.run, verbose=args.verbose)

    elif args.mode == 'graphem':
        job.run_graphem_cfg(args.config, verbose=args.verbose)


if __name__ == '__main__':
    main()
