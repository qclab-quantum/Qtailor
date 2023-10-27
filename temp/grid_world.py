import numpy
import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)
        self.step_cnt = 0
        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    # %%
    # Constructing Observations From Environment States
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    # Since we will need to compute observations both in ``reset`` and
    # ``step``, it is often convenient to have a (private) method ``_get_obs``
    # that translates the environment’s state into an observation. However,
    # this is not mandatory and you may as well compute observations in
    # ``reset`` and ``step`` separately:

    def _get_obs(self):
       return {"agent": self._agent_location, "target": self._target_location}

    # %%
    # We can also implement a similar method for the auxiliary information
    # that is returned by ``step`` and ``reset``. In our case, we would like
    # to provide the manhattan distance between the agent and the target:

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    # %%
    # Oftentimes, info will also contain some data that is only available
    # inside the ``step`` method (e.g. individual reward terms). In that case,
    # we would have to update the dictionary that is returned by ``_get_info``
    # in ``step``.

    # %%
    # Reset
    # ~~~~~
    #
    # The ``reset`` method will be called to initiate a new episode. You may
    # assume that the ``step`` method will not be called before ``reset`` has
    # been called. Moreover, ``reset`` should be called whenever a done signal
    # has been issued. Users may pass the ``seed`` keyword to ``reset`` to
    # initialize any random number generator that is used by the environment
    # to a deterministic state. It is recommended to use the random number
    # generator ``self.np_random`` that is provided by the environment’s base
    # class, ``gymnasium.Env``. If you only use this RNG, you do not need to
    # worry much about seeding, *but you need to remember to call
    # ``super().reset(seed=seed)``* to make sure that ``gymnasium.Env``
    # correctly seeds the RNG. Once this is done, we can randomly set the
    # state of our environment. In our case, we randomly choose the agent’s
    # location and the random sample target positions, until it does not
    # coincide with the agent’s position.
    #
    # The ``reset`` method should return a tuple of the initial observation
    # and some auxiliary information. We can use the methods ``_get_obs`` and
    # ``_get_info`` that we implemented earlier for that:

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    # %%
    # Step
    # ~~~~
    #
    # The ``step`` method usually contains most of the logic of your
    # environment. It accepts an ``action``, computes the state of the
    # environment after applying that action and returns the 4-tuple
    # ``(observation, reward, done, info)``. Once the new state of the
    # environment has been computed, we can check whether it is a terminal
    # state and we set ``done`` accordingly. Since we are using sparse binary
    # rewards in ``GridWorldEnv``, computing ``reward`` is trivial once we
    # know ``done``. To gather ``observation`` and ``info``, we can again make
    # use of ``_get_obs`` and ``_get_info``:

    def step(self, action:int):
        self.step_cnt = self.step_cnt+1
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        if isinstance(action, numpy.ndarray):
            print(action)
            print(self.step_cnt)
        direction = self._action_to_direction[action]

        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    # %%
    # Rendering
    # ~~~~~~~~~
    #
    # Here, we are using PyGame for rendering. A similar approach to rendering
    # is used in many environments that are included with Gymnasium and you
    # can use it as a skeleton for your own environments:

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

        def _render_frame(self):
            if self.window is None and self.render_mode == "human":
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (self.window_size, self.window_size)
                )
            if self.clock is None and self.render_mode == "human":
                self.clock = pygame.time.Clock()

            canvas = pygame.Surface((self.window_size, self.window_size))
            canvas.fill((255, 255, 255))
            pix_square_size = (
                self.window_size / self.size
            )  # The size of a single grid square in pixels

            # First we draw the target
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * self._target_location,
                    (pix_square_size, pix_square_size),
                ),
            )
            # Now we draw the agent
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                (self._agent_location + 0.5) * pix_square_size,
                pix_square_size / 3,
            )

            # Finally, add some gridlines
            for x in range(self.size + 1):
                pygame.draw.line(
                    canvas,
                    0,
                    (0, pix_square_size * x),
                    (self.window_size, pix_square_size * x),
                    width=3,
                )
                pygame.draw.line(
                    canvas,
                    0,
                    (pix_square_size * x, 0),
                    (pix_square_size * x, self.window_size),
                    width=3,
                )

            if self.render_mode == "human":
                # The following line copies our drawings from `canvas` to the visible window
                self.window.blit(canvas, canvas.get_rect())
                pygame.event.pump()
                pygame.display.update()

                # We need to ensure that human-rendering occurs at the predefined framerate.
                # The following line will automatically add a delay to keep the framerate stable.
                self.clock.tick(self.metadata["render_fps"])
            else:  # rgb_array
                return np.transpose(
                    np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
                )

    # %%
    # Close
    # ~~~~~
    #
    # The ``close`` method should close any open resources that were used by
    # the environment. In many cases, you don’t actually have to bother to
    # implement this method. However, in our example ``render_mode`` may be
    # ``"human"`` and we might need to close the window that has been opened:

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()