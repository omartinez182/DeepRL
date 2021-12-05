import d3rlpy

# Scorers
from d3rlpy.metrics.scorer import initial_state_value_estimation_scorer
from d3rlpy.metrics.scorer import true_q_value_scorer # Custom True Q value scorer
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import soft_opc_scorer

# Import Data sets
from d3rlpy.datasets import get_pybullet

# Import Algo
from d3rlpy.algos import CQL
from d3rlpy.gpu import Device
from d3rlpy.ope import FQE
from sklearn.model_selection import train_test_split
import argparse


def main(args):
    # Get dataset & environment
    dataset, env = get_pybullet(args.dataset) 

    d3rlpy.seed(args.seed)
    env.seed(args.seed)

    train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

    device = None if args.gpu is None else Device(args.gpu)

    # Show Logs
    print("=========================")
    print("Q FUNQ :  ", args.q_func)
    print("USE GPU : ", device)
    print("DATASET : ", args.dataset)
    print("EPOCHS (CQL) : ", args.epochs_cql)
    print("EPOCHS (FQE) : ", args.epochs_fqe)
    print("=========================")

    # Train CQL
    cql = CQL(q_func_factory=args.q_func,
              batch_size=256,
              n_action_samples=10,
              use_gpu=device)

    cql.fit(train_episodes,
            eval_episodes=test_episodes,
            n_epochs=args.epochs_cql,
            save_interval=10,
            scorers={
                'environment': evaluate_on_environment(env, epsilon=0.05),
                'init_value': initial_state_value_estimation_scorer,
                "true_q_value": true_q_value_scorer
            },
            with_timestamp=False,
            verbose=True,
            experiment_name=f"CQL{args.dataset}_{args.seed}")

    # Train OPE (FQE) for trained policy evaluation
    fqe = FQE(algo=cql,
              n_epochs=args.epochs_fqe,
              q_func_factory='qr',
              use_gpu=device)

    fqe.fit(dataset.episodes,
            n_epochs=args.epochs_fqe,
            eval_episodes=dataset.episodes,
            scorers={
                'init_value': initial_state_value_estimation_scorer,
                'soft_opc': soft_opc_scorer(600),
                "true_q_value": true_q_value_scorer
            },
            with_timestamp=False,
            verbose=True,
            experiment_name=f"FQE_{args.dataset}_{args.seed}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        default='hopper-bullet-mixed-v0')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs_cql', type=int, default=1)
    parser.add_argument('--epochs_fqe', type=int, default=1)
    parser.add_argument('--q-func',
                        type=str,
                        default='mean',
                        choices=['mean', 'qr', 'iqn', 'fqf'])
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    main(args)