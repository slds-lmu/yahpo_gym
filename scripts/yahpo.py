from yahpo_gym import benchmark_set
import yahpo_gym.benchmarks.lcbench
import argparse
import typing


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--benchmark', type=str, required=False, default="lcbench")
parser.add_argument('-i', '--instance', type=str, required=False, default="3945")
parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs, default=None)
args = parser.parse_args()

def sample_random(bench):
    return bench.get_opt_space().sample_configuration(1).get_dictionary()
    
def eval(args):
    bench = benchmark_set.BenchmarkSet(args.benchmark)
    bench.set_instance(args.instance)
    if args.kwargs is None:
        args.kwargs = sample_random(bench)
    ys = bench.objective_function(args.kwargs)
    return ys
    
if __name__ == '__main__':
    print(eval(args))
