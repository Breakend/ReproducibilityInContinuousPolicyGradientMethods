from ddpg_tensorflow.ddpg import DDPG
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.exploration_strategies.ou_strategy import OUStrategy
from sandbox.rocky.tf.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from sandbox.rocky.tf.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction

from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.gym_env import GymEnv
from rllab.misc import ext
import pickle
import tensorflow as tf

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("type", help="Type of DDPG to run: unified, unified-gated, regular")
parser.add_argument("env", help="The environment name from OpenAIGym environments")
parser.add_argument("--num_epochs", default=100, type=int)
parser.add_argument("--plot", action="store_true")
# parser.add_argument("--data_dir", default="./data/")
args = parser.parse_args()

stub(globals())
ext.set_seed(1)

supported_gym_envs = ["MountainCarContinuous-v0", "Hopper-v1", "Walker2d-v1", "Humanoid-v1", "Reacher-v1", "HalfCheetah-v1", "Swimmer-v1", "HumanoidStandup-v1"]

other_env_class_map  = { "Cartpole" :  CartpoleEnv}

if args.env in supported_gym_envs:
    gymenv = GymEnv(args.env, force_reset=True, record_video=False, record_log=False)
    # gymenv.env.seed(1)
else:
    gymenv = other_env_class_map[args.env]()

#TODO: assert continuous space


ddpg_type_map = {"regular" : DDPG}


ddpg_class = ddpg_type_map[args.type]



env = TfEnv(normalize(gymenv))




## loops:
num_experiments = 5

layer_1 = [400, 100, 100]
layer_2 = [300, 100, 50]

layer_size = 3


# n_itr = int(np.ceil(float(n_episodes*max_path_length)/flags['batch_size']))


for l in range(layer_size): 

        policy = DeterministicMLPPolicy(
            env_spec=env.spec,
            name="policy",
            # The neural network policy should have two hidden layers, each with 32 hidden units.
            hidden_sizes=(layer_1[l], layer_2[l]),
            hidden_nonlinearity=tf.nn.relu,
        )

        es = OUStrategy(env_spec=env.spec)

        qf = ContinuousMLPQFunction(env_spec=env.spec,
                                    hidden_sizes=(layer_1[l], layer_2[l]),
                                    hidden_nonlinearity=tf.nn.relu,)


        for e in range(num_experiments):

            algo = ddpg_class(
                env=env,
                policy=policy,
                es=es,
                qf=qf,
                batch_size=32,
                max_path_length=env.horizon,
                epoch_length=1000,
                min_pool_size=10000,
                n_epochs=args.num_epochs,
                discount=0.99,
                scale_reward=0.1,
                qf_learning_rate=1e-3,
                policy_learning_rate=1e-4,
                # Uncomment both lines (this and the plot parameter below) to enable plotting
                plot=args.plot,
            )


            run_experiment_lite(
                algo.train(),
                # log_dir=args.data_dir,
                # Number of parallel workers for sampling
                n_parallel=1,
                # Only keep the snapshot parameters for the last iteration
                snapshot_mode="last",
                # Specifies the seed for the experiment. If this is not provided, a random seed
                # will be used
                exp_name="reproducibility_ML/" + "DDPG/" + "HalfCheetah/" + "Network_Structure_Tune/" + "Layer_Size_" + str(l) + "_Experiment_" + str(e),
                seed=1,
                plot=args.plot,
            )
