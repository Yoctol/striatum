"""Thompson Sampling with Linear Payoff Solumilken Version
"""

""" Thompson Sampling with Linear Payoff
In This module contains a class that implements Thompson Sampling with Linear
Payoff. Thompson Sampling with linear payoff is a contexutal multi-armed bandit
algorithm which assume the underlying relationship between rewards and contexts
is linear. The sampling method is used to balance the exploration and
exploitation. Please check the reference for more details.
"""
import logging
import six
from six.moves import zip
import pdb

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spslg
import cudamat as cm

import pycuda.gpuarray as gpuarray
import pycuda.autoinit

from .bandit import BaseBandit
from ..utils import get_random_state

LOGGER = logging.getLogger(__name__)


class LinThompSamp(BaseBandit):
    r"""Thompson sampling with linear payoff.

    Parameters
    ----------
    history_storage : HistoryStorage object
        The HistoryStorage object to store history context, actions and rewards.

    model_storage : ModelStorage object
        The ModelStorage object to store model parameters.

    action_storage : ActionStorage object
        The ActionStorage object to store actions.

    recommendation_cls : class (default: None)
        The class used to initiate the recommendations. If None, then use
        default Recommendation class.

    delta: float, 0 < delta < 1
        With probability 1 - delta, LinThompSamp satisfies the theoretical
        regret bound.

    R: float, R >= 0
        Assume that the residual  :math:`ri(t) - bi(t)^T \hat{\mu}`
        is R-sub-gaussian. In this case, R^2 represents the variance for
        residuals of the linear model :math:`bi(t)^T`.

    epsilon: float, 0 < epsilon < 1
        A  parameter  used  by  the  Thompson Sampling algorithm.
        If the total trials T is known, we can choose epsilon = 1/ln(T).

    random_state: {int, np.random.RandomState} (default: None)
        If int, np.random.RandomState will used it as seed. If None, a random
        seed will be used.

    References
    ----------
    .. [1]  Shipra Agrawal, and Navin Goyal. "Thompson Sampling for Contextual
            Bandits with Linear Payoffs." Advances in Neural Information
            Processing Systems 24. 2011.
    """

    def __init__(self, history_storage, model_storage, action_storage,
                 recommendation_cls=None, context_dimension=128, delta=0.5,
                 R=0.01, epsilon=0.5, random_state=None,
                 use_sparse_svd=False, sparse_svd_k=6):
        super(LinThompSamp, self).__init__(history_storage, model_storage,
                                           action_storage, recommendation_cls)
        self.random_state = get_random_state(random_state)
        self.use_sparse_svd = use_sparse_svd
        self.sparse_svd_k = sparse_svd_k
        self.context_dimension = context_dimension

        # 0 < delta < 1
        if not isinstance(delta, float):
            raise ValueError("delta should be float")
        elif (delta < 0) or (delta >= 1):
            raise ValueError("delta should be in (0, 1]")
        else:
            self.delta = delta

        # R > 0
        if not isinstance(R, float):
            raise ValueError("R should be float")
        elif R <= 0:
            raise ValueError("R should be positive")
        else:
            self.R = R  # pylint: disable=invalid-name

        # 0 < epsilon < 1
        if not isinstance(epsilon, float):
            raise ValueError("epsilon should be float")
        elif (epsilon < 0) or (epsilon > 1):
            raise ValueError("epsilon should be in (0, 1)")
        else:
            self.epsilon = epsilon

        # model initialization
        invB = np.identity(self.context_dimension)  # pylint: disable=invalid-name
        mu_hat = np.zeros(shape=(self.context_dimension, 1))
        f = np.zeros(shape=(self.context_dimension, 1))
        self._model_storage.save_model({'invB': invB, 'mu_hat': mu_hat,
                                        'f': f, 'U': None, 'D': None})

        cm.cublas_init()

    def __del__(self):
        cm.shutdown()

    def _linthompsamp_score(self, context):
        """Thompson Sampling"""
        action_ids = list(six.viewkeys(context))
        context_array = cm.CUDAMatrix(
            np.asarray([context[action_id] for action_id in action_ids])
        )
        # context_array = np.asarray(list(six.viewvalues(context)))

        model = self._model_storage.get_model()
        invB = model['invB']  # pylint: disable=invalid-name
        mu_hat = cm.CUDAMatrix(model['mu_hat'])
        U = model['U']
        D = model['D']
        v = self.R * np.sqrt(24 / self.epsilon
                             * self.context_dimension
                             * np.log(1 / self.delta))                     ###############  1

        #mu_tilde = self.random_state.multivariate_normal(
        #    mu_hat.flat, v**2 * invB)[..., np.newaxis]
        if U is None or D is None:
            if self.use_sparse_svd:
                B_sps = sps.csr_matrix(invB)
                U, D, V = spslg.svds(B_sps, k=self.sparse_svd_k)
            else:
                U, D, V = np.linalg.svd(invB, full_matrices=False)
            model['U'] = U
            model['D'] = D

        x = np.random.normal(0.0, 1.0, size=len(D))
        cm_U = cm.CUDAMatrix(U)
        cm_x = cm.CUDAMatrix((v * np.sqrt(D) * x).reshape((len(D), 1)))
        mu_tilde = cm_U.dot(cm_x).add(mu_hat)

        estimated_reward_array = context_array.dot(mu_hat).asarray()
        score_array = context_array.dot(mu_tilde).asarray()

        estimated_reward_dict = {}
        uncertainty_dict = {}
        score_dict = {}
        for action_id, estimated_reward, score in zip(
                action_ids, estimated_reward_array, score_array):
            estimated_reward_dict[action_id] = float(estimated_reward)
            score_dict[action_id] = float(score)
            uncertainty_dict[action_id] = float(score - estimated_reward)
        return estimated_reward_dict, uncertainty_dict, score_dict

    def add_history(self, context):
        history_id = self._history_storage.add_history(context, [])
        return history_id

    def get_action(self, context, n_actions=None):
        """Return the action to perform

        Parameters
        ----------
        context : dictionary
            Contexts {action_id: context} of different actions.

        n_actions: int (default: None)
            Number of actions wanted to recommend users. If None, only return
            one action. If -1, get all actions.

        Returns
        -------
        history_id : int
            The history id of the action.

        recommendations : list of dict
            Each dict contains
            {Action object, estimated_reward, uncertainty}.
        """
        if self._action_storage.count() == 0:
            return self._get_action_with_empty_action_storage(context,
                                                              n_actions)

        if not isinstance(context, dict):
            raise ValueError(
                "LinThompSamp requires context dict for all actions!")
        if n_actions == -1:
            n_actions = self._action_storage.count()

        estimated_reward, uncertainty, score = self._linthompsamp_score(context)

        if n_actions is None:
            recommendation_id = max(score, key=score.get)
            recommendations = self._recommendation_cls(
                action=self._action_storage.get(recommendation_id,
                                                return_copy=False),
                estimated_reward=estimated_reward[recommendation_id],
                uncertainty=uncertainty[recommendation_id],
                score=score[recommendation_id],
            )
        else:
            recommendation_ids = sorted(score, key=score.get,
                                        reverse=True)[:n_actions]
            recommendations = []  # pylint: disable=redefined-variable-type
            for action_id in recommendation_ids:
                recommendations.append(self._recommendation_cls(
                    action=self._action_storage.get(action_id,
                                                    return_copy=False),
                    estimated_reward=estimated_reward[action_id],
                    uncertainty=uncertainty[action_id],
                    score=score[action_id],
                ))

        history_id = self._history_storage.add_history(context, recommendations)
        return history_id, recommendations

    def reward(self, history_id, rewards):
        """Reward the previous action with reward.

        Parameters
        ----------
        history_id : int
            The history id of the action to reward.

        rewards : dictionary
            The dictionary {action_id, reward}, where reward is a float.
        """
        context = (self._history_storage
                   .get_unrewarded_history(history_id)
                   .context)
        # Update the model
        model = self._model_storage.get_model()
        invB = cm.CUDAMatrix(model['invB'])
        f = cm.CUDAMatrix(model['f'])
        invB_context_t = cm.empty((self.context_dimension, 1))
        sub_inv_array = cm.empty((
            self.context_dimension, self.context_dimension
        ))
        checkpoint = cm.empty((1, 1))
        context_t = cm.empty((self.context_dimension, 1))

        for action_id, reward in six.viewitems(rewards):
            context_t.assign(
                cm.CUDAMatrix(np.reshape(context[action_id], (-1, 1)))
            )
            cm.dot(invB, context_t, invB_context_t)
            cm.dot(
                context_t.T, invB_context_t, checkpoint
            )
            checkpoint.copy_to_host()
            invertible_checkpoint = 1.0 + checkpoint.numpy_array[0][0]
            if abs(invertible_checkpoint) < 1e-5:
                invertible_checkpoint = \
                    np.sign(invertible_checkpoint) \
                    * (abs(invertible_checkpoint) + 1e+5)
            cm.dot(
                invB_context_t,
                invB_context_t.T,
                sub_inv_array
            )
            sub_inv_array.mult(-1.0 / invertible_checkpoint)
            invB.add(sub_inv_array)
            f.add(
                context_t.mult(int(reward))
            )
        mu_hat = cm.dot(invB, f)
        self._model_storage.save_model({
            'invB': invB.asarray(),
            'mu_hat': mu_hat.asarray(),
            'f': f.asarray(),
            'U': None,
            'D': None
        })
        # Update the history
        self._history_storage.add_reward(history_id, rewards)

    def add_action(self, actions):
        """ Add new actions (if needed).

        Parameters
        ----------
        actions : iterable
            A list of Action oBjects for recommendation
        """
        self._action_storage.add(actions)

    def remove_action(self, action_id):
        """Remove action by id.

        Parameters
        ----------
        action_id : int
            The id of the action to remove.
        """
        self._action_storage.remove(action_id)
