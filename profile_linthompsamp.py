from six.moves import range
from cProfile import Profile
import numpy as np
import matplotlib.pyplot as plt

from striatum.storage import (
    MemoryHistoryStorage,
    MemoryModelStorage,
    MemoryActionStorage,
    Action,
)
from striatum.bandit import LinThompSamp
from striatum import simulation


def main():
    n_rounds = 100
    n_actions = 100
    context_dimension = 500
    action_storage = MemoryActionStorage()
    action_storage.add([Action(i) for i in range(n_actions)])
    random_state = np.random.RandomState(0)

    policy = LinThompSamp(MemoryHistoryStorage(), MemoryModelStorage(),
                            action_storage,
                            context_dimension=context_dimension,
                            delta=0.1, R=0.01, epsilon=0.5,
                            random_state=random_state,
                            use_gpu=True)
    context1, desired_actions1 = simulation.simulate_data(
        n_rounds, context_dimension, action_storage, random_state=0, sparse=True)
    profiler = Profile()
    profiler.enable()
    simulation.evaluate_policy(policy, context1,
                               desired_actions1)
    profiler.disable()
    profiler.print_stats()

if __name__ == '__main__':
    main()
