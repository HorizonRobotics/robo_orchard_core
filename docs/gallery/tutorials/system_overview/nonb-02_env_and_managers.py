# ruff: noqa: E501 D415 D205

# Project RoboOrchard
#
# Copyright (c) 2024 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Environment and Managers
========================
"""


# %%
# What is an Environment?
# -----------------------
#
# Environment defines the interaction between the agent and the world. It
# provides the agent with the necessary information to make decisions and
# take actions, and receives the actions from the agent to update the world
# state. The environment can be a simulation or a real-world system.
#
# In the context of reinforcement learning (RL), the environment provides
# interfaces for markov decision processes (MDP). In this case, the environment
# usually has the following components:
#
# - Observation Space: The set of all possible observations that the agent can
#   receive from the environment.
# - Action Space: The set of all possible actions that the agent can take.
# - Reward Function: A function that maps the state of the environment and the
#   action taken by the agent to a scalar reward.
# - Transition Dynamics: A function that maps the current state of the
#   environment and the action taken by the agent to the next state of the
#   environment.
# - Initial State Distribution: A distribution that defines the initial state
#   of the environment.
# - Termination Condition: A condition that defines when the episode ends.
#
# So the environment concept and interface can be simple, but the
# implementation can be complex. Most frameworks follow the `Env` interface
# from `OpenAI Gym <https://gymnasium.farama.org/introduction/basic_usage/>`_.
# Isaac Lab also provides a similar abstraction for the
# `Environment <https://https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/task_workflows.html>`_,
# which provides *Manager-based* interfaces and *Direct* interfaces.


# %%
# What is a Manager?
# ------------------
#
# `Manager` is a **design pattern** that handles multiple entities of the
# same type and provides a unified interface to interact with them.
# For example, in the context of the environment, the manager can be
# responsible for certain aspects of the environment, such as managing
# the actions, observations, and events.
#
# For `Manager` in `Environment`, it can also follows the `Observer` design
# pattern to notify the agents about the changes in the environment and update
# the states. The concept of `Manager` deals with the interaction and
# communication between the environment and the agents. It helps to focus
# on how to represent to the a certain aspect of the environment changes.
#
# Isaac Lab provides `Manager-based Env`, which is a good practice to manage
# the environment with the principles mentioned above. It provides a
# structured way to handle the environment and make it easy to extend and
# maintain. RoboOrchard also follows the same abstraction but in a more
# general way.


# %%
# Environment and Managers in RoboOrchard
# ---------------------------------------
#
# In RoboOrchard, we define the `Environment` and `Manager` as the basic
# interfaces.
#
# Environment
# ~~~~~~~~~~~
#
# Specifically, we provide :py:class:`~robo_orchard_core.envs.env_base.EnvBase`
# as the base class for environments, which only defines the basic interfaces
# and properties:
#
# - :py:meth:`~robo_orchard_core.envs.env_base.EnvBase.reset`: Reset the
#   environment.
# - :py:meth:`~robo_orchard_core.envs.env_base.EnvBase.step`: Execute one
#   step of the environment.
# - :py:meth:`~robo_orchard_core.envs.env_base.EnvBase.close`: Close the
#   environment.
#
# The interface of `EnvBase` can be extended by the specific environment
# implementation, for both Simulation and Real-world systems. The difference
# between the two is how the observations, actions, and events are generated
# and processed.
#
# Manager
# ~~~~~~~~~~~
#
# For the `Manager`, we provide the
# :py:class:`~robo_orchard_core.envs.managers.manager_base.ManagerBase` as the
# base class for managers, and the following specific managers:
#
# - :py:class:`~robo_orchard_core.envs.managers.observations.observation_manager.ObservationManager`:
#   The observation manager generates the observations of the environment. It
#   can be extended to different types of observations that the agents need
#   to make decisions. Observation is also a key component of the return from
#   :py:meth:`~robo_orchard_core.envs.env_base.EnvBase.step` function.
# - :py:class:`~robo_orchard_core.envs.managers.actions.action_manager.ActionManager`:
#   The action manager processes the actions from the agents. It is responsible
#   for validating the actions and executing them in the environment. The action
#   is also a key component as the input to the :py:meth:`~robo_orchard_core.envs.env_base.EnvBase.step`
#   function.
# - :py:class:`~robo_orchard_core.envs.managers.events.event_manager.EventManager`:
#   The event manager processes the events from the environment.
#
#
# Event can be a very general concept. For example, it can be a trigger to
# publish the observations to the agents(like the observation manager), or
# a trigger to update the environment state according to the received
# actions(like the action manager). But we already have the observation manager
# and the action manager, so the event manager is more
# like a general manager to *handle the events that are not directly related
# to the observations and actions, like the environment initialization,
# termination, etc*.
#
# Manager Term
# ~~~~~~~~~~~~~~~~~~~~~~
#
# The `Manager` handles a specific group of entities, called `ManagerTerm`. The
# `ManagerTerm` is the basic block to interact with the environment.
# Any operation that needs to interact with the environment should be
# implemented to extend the
# :py:class:`~robo_orchard_core.envs.managers.manager_term_base.ManagerTermBase`.
#
# Use Manager-based Environment
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Both `Manager` and `ManagerTerm` are designed to be independent of the
# simulation engines, or the real-world systems. They use `Env` as the
# interface to interact with the environment. This design makes it easy
# to extend and maintain the environment and managers, and also makes
# it easy to switch between different simulation engines and
# real-world systems.
#
# We expect and recommend users to use the
# :py:class:`~robo_orchard_core.envs.manager_based_env.TermManagerBasedEnv`
# as the base class for the environment, which provides modular and structured
# interfaces for flexible and scalable development. Package
# `robo_orchard_isaac` provides the adaptation of the
# `Manager-based Environment` for Isaac Lab, which is an example to follow.
#
