# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import multiprocessing as mp
import threading
import time

import numpy as np
from libtorchbeast import rpcenv
from torchbeast.env_wrappers import create_env as create_other_env


# yapf: disable
parser = argparse.ArgumentParser(description='Remote Environment Server')

parser.add_argument("--pipes_basename", default="unix:/tmp/polybeast",
                    help="Basename for the pipes for inter-process communication. "
                    "Has to be of the type unix:/some/path.")
parser.add_argument('--num_servers', default=4, type=int, metavar='N',
                    help='Number of environment servers.')
parser.add_argument('--env', type=str, default='PongNoFrameskip-v4',
                    help='Gym environment.')
# For Coinrun Platforms: setting statics and dynamics to i+num_worlds holds
# out world i from training. E.g. --set-statics=7 --set-dynamics=7 means
# only worlds 0,1,3 will be used for RL+SVAE training.
parser.add_argument('--set_statics', type=int, default=0,
                    help='Statics for CoinRun world [0-3]')
parser.add_argument('--set_dynamics', type=int, default=0,
                    help='Dynamics for CoinRun world [0-3]')
parser.add_argument('--num_levels', type=int, default=500,
                    help='Number of levels per platforms world')
parser.add_argument('--any_custom_game', type=int, default=1,
                    help='Select any of the 4 custom games.')
parser.add_argument('--is_high_res', type=int, default=0,
                    help='Whether to render in high resolution.')
parser.add_argument('--default_zoom', type=int, default=5,
                    help='Default zoom for game visualization.')
# yapf: enable


class Env:
    def reset(self):
        print("reset called")
        return np.ones((4, 84, 84), dtype=np.uint8)

    def step(self, action):
        frame = np.zeros((4, 84, 84), dtype=np.uint8)
        return frame, 0.0, False, {}  # First three mandatory.


def create_env(env_name, flags, lock=threading.Lock()):
    with lock:  # envs might not be threadsafe at construction time.
        return create_other_env(env_name, flags)


def serve(env_name, server_address, flags):
    init = Env if env_name == "Mock" else lambda: create_env(env_name, flags)
    server = rpcenv.Server(init, server_address=server_address)
    server.run()


if __name__ == "__main__":
    flags = parser.parse_args()

    if not flags.pipes_basename.startswith("unix:"):
        raise Exception("--pipes_basename has to be of the form unix:/some/path.")

    processes = []
    for i in range(flags.num_servers):
        p = mp.Process(
            target=serve, args=(flags.env, f"{flags.pipes_basename}.{i}", flags), daemon=True
        )
        p.start()
        processes.append(p)

    try:
        # We are only here to listen to the interrupt.
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        pass
