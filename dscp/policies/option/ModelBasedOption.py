# Core imports
import ipdb
import random
import itertools
import numpy as np
from copy import deepcopy
from sklearn import svm

# Local imports
from dscp.rl.MDPClass import MDP
from dscp.rl.PolicyClass import Policy
from dscp.policies.sac.SACClass import SAC
from dscp.policies.option.GoalExtractorClass import GoalExtractor

# Typing imports
from typing import Union

class ModelBasedOption(object):
    def __init__(self, , 
        mdp: MDP,
        goal_extractor : GoalExtractor,
        target: Union[ModelBasedOption, np.ndarray],
        global_solver: Policy,
        gestation_period: int,
        timeout: int, 
        max_steps: int,
        ):

        assert mdp.get_state_dim() == goal_extractor.get_state_dim()
        if isinstance(target, np.ndarray):
            assert target.shape == (goal_extractor.get_goal_dim(),)

        self._policy : Policy = SAC(
            mpd.get_state_dim() + goal_extractor.get_goal_dim(),
            mdp.get_action_dim(),
            mdp.get_action_bounds(),
            mpd.get_gamma(),

            #TODO: These are default assumptions, I should add in the code to tune them
            int(1e6),
            256,
            0.001,
            0.001
            )

        self._mdp = mdp
        self._goal_extractor = goal_extractor
        self._target = target
        self._global_solver = global_solver
        self._gestation_period = gestation_period
        self._timeout = timeout
        self._max_steps = max_steps

        # TODO set up seeding and option identity
        #self.seed = seed
        #self.option_idx = option_idx

        self._num_goal_hits = 0
        self._positive_examples = []
        self._negative_examples = []
        self._optimistic_classifier = None
        self._pessimistic_classifier = None
        
        # TODO: Ask Akhil what this does
        self._in_out_pairs = []

    # ------------------------------------------------------------
    # Learning Phase Methods
    # ------------------------------------------------------------

    def is_init_true(self, state : np.ndarray) -> bool:
        return (self._num_goal_hits < self._gestation_period)

        # TODO:

        #features = self.mdp.get_position(state)
        #return self.optimistic_classifier.predict([features])[0] == 1

    def is_term_true(self, state):
        if self.parent is None:
            return self.target_salient_event(state) # Change this

        return self.parent.pessimistic_is_init_true(state) # and this

    def pessimistic_is_init_true(self, state):
        # TODO: HERE
        if self.global_init or self.get_training_phase() == "gestation":
            return True

        features = self.mdp.get_position(state)
        return self.pessimistic_classifier.predict([features])[0] == 1

    def is_at_local_goal(self, state, goal):
        """ Goal-conditioned termination condition. """

        reached_goal = self.mdp.sparse_gc_reward_function(state, goal, {})[1]
        reached_term = self.is_term_true(state) or state.is_terminal()
        return reached_goal and reached_term

    # ------------------------------------------------------------
    # Control Loop Methods
    # ------------------------------------------------------------

    def act(self, state, goal):
        """ Epsilon-greedy action selection. """

        if random.random() < 0.2:
            return self.mdp.sample_random_action()
        return self.solver.act(state, goal)

    def update(self, state, action, reward, next_state):
        """ Learning update for option model/actor/critic. """

        solver = self.solver if self.global_init else self.global_solver
        solver.step(state.features(), action, reward, next_state.features(), next_state.is_terminal())

    def get_goal_for_rollout(self):
        """ Sample goal to pursue for option rollout. """

        if self.parent is None and self.target_salient_event is not None:
            return self.target_salient_event.get_target_position()

        sampled_goal = self.parent.sample_from_initiation_region()
        assert sampled_goal is not None
        return sampled_goal.squeeze()

    def rollout(self, step_number):
        """ Main option control loop. """

        start_state = deepcopy(self.mdp.cur_state)
        assert self.is_init_true(start_state)

        num_steps = 0
        total_reward = 0
        visited_states = []
        option_transitions = []

        state = deepcopy(self.mdp.cur_state)
        goal = self.get_goal_for_rollout()

        print(f"[Step: {step_number}] Rolling out {self.name}, from {state.position} targeting {goal}")

        while not self.is_at_local_goal(state, goal) and step_number < self.max_steps and num_steps < self.timeout:

            # Control
            action = self.act(state, goal)
            # TODO: Don't update on-line anymore
            reward, next_state = self.mdp.execute_agent_action(action)
            #self.update(state, action, reward, next_state)

            # Logging
            num_steps += 1
            step_number += 1
            total_reward += reward
            visited_states.append(state)
            option_transitions.append((state, action, reward, next_state))
            state = deepcopy(self.mdp.cur_state)

        visited_states.append(state)

        self.in_out_pairs.append((start_state.features(), state.features()))

        if self.is_term_true(state):
            self.num_goal_hits += 1

        # Train the policy using HER
        self.gc_train_solver(option_transitions, goal)
        self.gc_train_solver(option_transitions, state)

        # Train your initiation classifier
        self.derive_positive_and_negative_examples(visited_states)

        # Always be refining your initiation classifier
        if not self.global_init:
            self.fit_initiation_classifier()

        return option_transitions, total_reward


    # ------------------------------------------------------------
    # HER implementation
    # ------------------------------------------------------------

    def gc_train_solver(self, option_transitions, goal):
        goal_state = self.mdp.get_position(goal)
        gc_reward_function = self.mdp.get_gc_reward_function(self.dense_reward)

        # Goal conditioned experience replay
        for state, action, _, next_state in option_transitions:
            gc_state = np.concatenate([state.features(), goal_state], axis = 0)
            gc_next_state = np.concatenate([next_state.features(), goal_state], axis = 0)
            gc_reward = gc_reward_function(state, goal, {})
            
            self.solver.step(gc_state, action, gc_reward, gc_next_state, next_state.is_terminal())


    # ------------------------------------------------------------
    # Learning Initiation Classifiers
    # ------------------------------------------------------------

    def sample_from_initiation_region(self):
        """ Sample from the pessimistic initiation classifier. """

        positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)
        training_predictions = self.pessimistic_classifier.predict(positive_feature_matrix)
        positive_training_examples = positive_feature_matrix[training_predictions == 1]
        if positive_training_examples.shape[0] > 0:
            # TODO: Replace this with numpy sample, for consistent seeding
            idx = random.sample(range(positive_training_examples.shape[0]), k=1)
            return positive_training_examples[idx]

    def derive_positive_and_negative_examples(self, visited_states):
        start_state = visited_states[0]
        final_state = visited_states[-1]

        if self.is_term_true(final_state):
            positive_states = [start_state] + visited_states[-self.buffer_length:]
            positive_examples = [self.mdp.get_position(s) for s in positive_states]
            self.positive_examples.append(positive_examples)
        else:
            negative_examples = [self.mdp.get_position(start_state)]
            self.negative_examples.append(negative_examples)

    def fit_initiation_classifier(self):
        if len(self.negative_examples) > 0 and len(self.positive_examples) > 0:
            self.train_two_class_classifier()
        elif len(self.positive_examples) > 0:
            self.train_one_class_svm()

    @staticmethod
    def construct_feature_matrix(examples):
        states = list(itertools.chain.from_iterable(examples))
        return np.array(states)

    def train_one_class_svm(self, nu=0.1):
        positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)
        self.pessimistic_classifier = svm.OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
        self.pessimistic_classifier.fit(positive_feature_matrix)

        self.optimistic_classifier = svm.OneClassSVM(kernel="rbf", nu=nu/10., gamma="scale")
        self.optimistic_classifier.fit(positive_feature_matrix)

    def train_two_class_classifier(self, nu=0.1):
        positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)
        negative_feature_matrix = self.construct_feature_matrix(self.negative_examples)
        positive_labels = [1] * positive_feature_matrix.shape[0]
        negative_labels = [0] * negative_feature_matrix.shape[0]

        X = np.concatenate((positive_feature_matrix, negative_feature_matrix))
        Y = np.concatenate((positive_labels, negative_labels))

        if negative_feature_matrix.shape[0] >= 10:
            kwargs = {"kernel": "rbf", "gamma": "scale", "class_weight": "balanced"}
        else:
            kwargs = {"kernel": "rbf", "gamma": "scale"}

        self.optimistic_classifier = svm.SVC(**kwargs)
        self.optimistic_classifier.fit(X, Y)

        training_predictions = self.optimistic_classifier.predict(X)
        positive_training_examples = X[training_predictions == 1]

        if positive_training_examples.shape[0] > 0:
            self.pessimistic_classifier = svm.OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
            self.pessimistic_classifier.fit(positive_training_examples)


    # ------------------------------------------------------------
    # Convenience functions # TODO Fix all of these, I don't think options should have names
    # ------------------------------------------------------------

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if isinstance(other, ModelBasedOption):
            return self.name == other.name
        return False