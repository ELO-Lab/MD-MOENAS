from util.load_cfg import load_cfg

import click

from optimizer.EA.ea_agent import EvoAgent
from optimizer.EA.util.callback import IGDMonitor, NonDominatedProgress, TimeLogger, CheckpointSaver

from tensorboardX import SummaryWriter

from copy import deepcopy

@click.command()
@click.option('--summary_writer', '-sw', is_flag=True, help='Use summary writer to log graphs.')
@click.option('--console_log', is_flag=True, help='Log output to the console.')
@click.option("--loops_if_rand", type=int, default=10, help="Total runs for evaluation.")
@click.option('--seed', '-s', default=-1, type=int, help='Random seed.')
@click.option('--pop_size', '-p', default=50, type=int, help='Population size')
@click.option('--n_evals', default=3000, type=int, help='Number of evaluations.')
@click.option('--use_archive', is_flag=True, help='Use elitist archive to evaluate for IGD instead of rank 0 in the population.')
@click.option('--eval_igd', is_flag=True, help='Calculate IGD each generation during the search.')
@click.option('--search_space', '-ss', required=True, type=click.Choice(['tss', 'sss'], case_sensitive=False), help='Choose search space to perform NAS. Valid arguments: TSS/SSS')
@click.option('--datasets', '-dts', required=True, multiple=True, type=click.Choice(['cifar10', 'cifar100'], case_sensitive=True))
@click.option('--efficiency', '-f0', default='flops', type=click.Choice(['flops', 'params', 'latency'], case_sensitive=True), help='Choose which objective to optimize for model efficiency')
@click.option('--epoch', '-ep', default=24, type=int, help='Number of training epochs to perform for each candidate architecture')
@click.option('--eval_dts', type=click.Choice(['cifar10', 'ImageNet16-120'], case_sensitive=True), default='ImageNet16-120')
def cli(console_log,
        loops_if_rand,
        seed,
        **kwargs):

    CONFIG = 'config/moenas.yml'
    if seed < 0 and loops_if_rand > 0:
        try:
            for i in range(loops_if_rand):
                cfg = load_cfg(CONFIG, seed=i, console_log=console_log, callback=update_cfg, **kwargs)
                solver = setup_agent(config=cfg, seed=i, **kwargs)
                solver.run()
        except KeyboardInterrupt:
            print('Interrupted. You have entered CTRL+C...')
        except Exception as e:
            import traceback
            traceback.print_exc()
    else:
        seed = seed if seed >= 0 else 0
        cfg = load_cfg(CONFIG, seed=seed, console_log=console_log, callback=update_cfg, **kwargs)
        solver = setup_agent(config=cfg, seed=seed, **kwargs)
        solver.solve()

def setup_agent(config, 
                seed, 
                summary_writer, 
                use_archive, 
                eval_igd, 
                **kwargs):
    cfg = deepcopy(config)
    summary_writer = SummaryWriter(cfg.summary_dir) if summary_writer else None

    callbacks = [
        NonDominatedProgress(plot_pf=False, labels=['Floating-point operations (M)', 'Error rate (%)']),
        CheckpointSaver(),
        TimeLogger()
    ]

    if eval_igd:
        igd_monitor = IGDMonitor(
            normalize=True, 
            from_archive=use_archive, 
            convert_to_pf_space=True, 
            topk=5
        )
        callbacks = [igd_monitor] + callbacks

    agent = EvoAgent(
        cfg, 
        seed,
        callbacks=callbacks,
        summary_writer=summary_writer
    )

    return agent

def update_cfg(cfg, 
               pop_size, 
               n_evals, 
               search_space, 
               datasets, 
               efficiency, 
               epoch, 
               eval_dts, 
               **kwargs):
    HP = {'tss': 200, 'sss': 90}

    assert epoch < HP[search_space]

    if search_space == 'sss':
        cfg.eliminate_duplicates.name = 'DefaultDuplicateElimination'
        cfg.eliminate_duplicates.kwargs = {}
    else:
        cfg.eliminate_duplicates.name = 'duplicate.TSSDuplicateElimination'
        cfg.eliminate_duplicates.kwargs = {'isomorphic': True}

    cfg.algorithm.kwargs.pop_size = pop_size
    cfg.algorithm.kwargs.n_offsprings = pop_size
    cfg.termination.kwargs.n_max_evals = n_evals
    cfg.problem.kwargs.search_space = search_space

    if len(datasets) > 1:
        cfg.problem.name = cfg.problem.name.format('MD')
    else:
        cfg.problem.name = cfg.problem.name.format('')
        datasets = datasets[0]

    exp_name = '{search_space}-{datasets}-{efficiency}_error'.format(
        search_space=search_space,
        datasets=datasets,
        efficiency=efficiency
    )
    cfg.exp_name = cfg.exp_name.format(exp_name)
    
    cfg.problem.kwargs.dataset = datasets
    cfg.problem.kwargs.epoch = epoch
    cfg.problem.kwargs.efficiency = efficiency
    cfg.problem.kwargs.pf_path = cfg.problem.kwargs.pf_path.format(
        dataset=eval_dts,
        search_space=search_space,
        efficiency=efficiency,
        hp=HP[search_space]
    )
    cfg.problem.kwargs.pf_dict.dataset = eval_dts

    return cfg

if __name__ == '__main__':
    cli()

    