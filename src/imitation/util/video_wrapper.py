"""Wrapper to record rendered video frames from an environment."""

import os

import gym
from gym.wrappers.monitoring import video_recorder

from imitation.data import types
from typing import Optional

class VideoWrapper(gym.Wrapper):
    """Creates videos from wrapped environment by calling render after each timestep."""

    def __init__(
        self,
        env: gym.Env,
        directory: types.AnyPath,
        single_video: bool = True,
        save_interval: Optional[int] = None,
    ):
        """Builds a VideoWrapper.

        Args:
            env: the wrapped environment.
            directory: the output directory.
            single_video: if True, generates a single video file, with episodes
                concatenated. If False, a new video file is created for each episode.
                Usually a single video file is what is desired. However, if one is
                searching for an interesting episode (perhaps by looking at the
                metadata), then saving to different files can be useful.
            save_interval: the number of episodes skipped between video saving. Only
                used if `single_video == False`.
        """
        super().__init__(env)
        self.episode_id = 0
        self.video_recorder = None
        self.single_video = single_video
        self.save_interval = save_interval
        if save_interval is not None:
            assert not self.single_video, "save_interval not working for single video"
            if not isinstance(save_interval, int) or save_interval < 1:
                raise ValueError(f"save_interval {save_interval} must be positive int")

        self.directory = os.path.abspath(directory)
        os.makedirs(self.directory)

    def _reset_video_recorder(self) -> None:
        """Creates a video recorder if one does not already exist.

        Called at the start of each episode (by `reset`). When a video recorder is
        already present, it will only create a new one if `self.single_video == False`.
        """
        if self.video_recorder is not None:
            # Video recorder already started.
            if not self.single_video:
                # We want a new video for each episode, so destroy current recorder.
                self.video_recorder.close()
                self.video_recorder = None

        if self.video_recorder is None:
            # No video recorder -- start a new one.
            self.video_recorder = video_recorder.VideoRecorder(
                env=self.env,
                base_path=os.path.join(
                    self.directory,
                    "video.{:06}".format(self.episode_id),
                ),
                metadata={"episode_id": self.episode_id},
            )

    def reset(self):
        if self.episode_id % self.save_interval == 0:
            self._reset_video_recorder()
        self.episode_id += 1
        return self.env.reset()

    def step(self, action):
        res = self.env.step(action)
        if self.episode_id % self.save_interval == 0:
            self.video_recorder.capture_frame()
        return res

    def close(self) -> None:
        if self.video_recorder is not None:
            self.video_recorder.close()
            self.video_recorder = None
        super().close()
