import torch


def render_episode_with_human_mode(envs, agent, device, args):
    state, info = envs.reset()
    next_obs = torch.Tensor(state).to(device)

    for step in range(0, args.num_steps):
        with torch.no_grad():
            action, _, _, value = agent.get_action_and_value(next_obs)

        # observations, rewards, terminations, truncations, infos
        next_obs, reward, terminations, truncations, info = envs.step(action.cpu().numpy())
        done = terminations or truncations
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

    envs.close()
