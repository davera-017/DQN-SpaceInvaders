from typing import Dict, Callable, Type

import os
import numpy as np

from tqdm import tqdm
import gymnasium as gym

from .agent import Agent, Experience
from .utils import epsilon_linear_scheduler, make_performance_plot

from hydra.utils import instantiate
from hydra import compose, initialize


def train(agent: Agent, envs: gym.Env, train_params: Dict):
    performance = []

    epsilon = train_params["epsilon"]
    epsilon_duration = train_params["exploration_fraction"] * train_params["total_timesteps"]
    epsilon_decay = (train_params["min_epsilon"] - train_params["epsilon"]) / epsilon_duration

    obs, _ = envs.reset()

    loop = tqdm(range(train_params["total_timesteps"]))

    for global_step in loop:
        epsilon = epsilon_linear_scheduler(epsilon, train_params["min_epsilon"], epsilon_decay)

        actions = agent.act(obs, epsilon)
        next_obs, rewards, terminateds, _, infos = envs.step(actions)

        if "final_info" in infos:
            for env_idx, info in enumerate(infos["final_info"]):
                # Skip the envs that are not done
                if not info or "episode" not in info:
                    continue
                performance.append((global_step, info["episode"]["r"][0], info["episode"]["l"][0]))

        experience = Experience(obs, actions, next_obs, terminateds, rewards, infos)

        begin_learning = global_step > train_params["learning_starts"]
        if begin_learning:
            train_local = global_step % train_params["train_frequency"] == 0
            sync_networks = global_step % train_params["target_update_frequency"] == 0
            agent.update(experience, train_local, sync_networks)

            # Save checkpoint
            if (global_step - train_params["learning_starts"]) % train_params["checkpoint_frequency"] == 0:
                agent.save(train_params["model_path"])
                np.save(os.path.join(train_params["model_path"], "performance.npy"), np.array(performance))

        obs = next_obs

    # Save at the end of the training
    agent.save(train_params["model_path"])
    np.save(os.path.join(train_params["model_path"], "performance.npy"), np.array(performance))


def evaluate(
    env_id: str,
    eval_episodes: int,
    make_env: Callable[[str, int, int], gym.Env],
    agent: Type[Agent],
    model_path: str,
    device: str,
    epsilon: float = 0.01,
    seed: int = 1234,
    **kwargs,
):
    n_envs = 4
    envs = make_env(env_id, n_envs, seed)
    agent = Agent(envs, model_path=model_path, device=device)

    agent.local_network.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        actions = agent.act(obs, epsilon)
        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for env_idx, info in enumerate(infos["final_info"]):
                # Skip the envs that are not done
                if not info or "episode" not in info:
                    continue
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    envs.close()
    episodic_returns = np.array(episodic_returns)

    return episodic_returns.mean(), episodic_returns.std()


def visualize(
    env_id: str,
    eval_episodes: int,
    make_env: Callable[[str, int, int], gym.Env],
    agent: Type[Agent],
    model_path: str,
    device: str,
    epsilon: float = 0.01,
    seed: int = 1234,
    **kwargs,
):
    human_env = make_env(env_id, 1, seed, capture_video=True)
    agent = Agent(human_env, model_path=model_path, device=device)
    agent.local_network.eval()

    obs, _ = human_env.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        actions = agent.act(obs, epsilon)
        next_obs, _, _, _, infos = human_env.step(actions)
        if "final_info" in infos:
            for _, info in enumerate(infos["final_info"]):
                # Skip the envs that are not done
                if not info or "episode" not in info:
                    continue
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs
    human_env.close()


if __name__ == "__main__":
    with initialize(version_base=None, config_path="conf/"):
        cfg = compose(config_name="config.yaml")

    train_params = cfg.train_params
    if train_params["do_train"]:
        envs = instantiate(cfg.envs)
        replay_buffer = instantiate(cfg.replay_buffer)(
            observation_space=envs.single_observation_space, action_space=envs.single_action_space
        )
        estimator = instantiate(cfg.estimator)
        local_network = instantiate(cfg.local_network)(n_actions=envs.single_action_space.n)
        target_network = instantiate(cfg.target_network)(n_actions=envs.single_action_space.n)

        agent = instantiate(cfg.agent)(
            envs=envs,
            replay_buffer=replay_buffer,
            target_network=target_network,
            local_network=local_network,
            estimator=estimator,
        )

        train(agent, envs, train_params)

    eval_params = cfg.eval_params
    if eval_params["do_eval"]:
        make_performance_plot(eval_params["model_path"])
        mean, std = evaluate(**instantiate(cfg.eval_params))
        print(f"Model performance on {eval_params['eval_episodes']} episodes: {round(mean)} ± {round(std)}")

    visualization_params = cfg.visualization_params
    if visualization_params["do_visualization"]:
        visualize(**instantiate(cfg.visualization_params))
