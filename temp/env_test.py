import gymnasium
from gymnasium.envs.registration import register
from gymnasium.wrappers import FlattenObservation
if __name__ == '__main__':
     register(
          id="GridWorld-v0",
          entry_point="temp.grid_world:GridWorldEnv",
          max_episode_steps=300,
     )
     env = FlattenObservation(gymnasium.make('GridWorld-v0',size = 10))
     print(env.observation_space)