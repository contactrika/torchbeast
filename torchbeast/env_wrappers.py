#
# @contactrika
#
# Wrappers for custom envs.
#
from torchbeast import atari_wrappers
from coinrun import coinrunenv


class CoinRunOneEnv(coinrunenv.CoinRunVecEnv):

    def seed(self, seed):
        print('CoinRun ignores seed()')

    def step(self, actions):
        self.step_async(actions)
        res = self.step_wait()
        return res


def create_env(env_name, flags):
    if env_name.startswith('Coin'):
        from coinrun import coinrunenv
        from coinrun import setup_utils as coinrun_setup_utils
        coinrun_setup_utils.setup_and_load(
            use_cmd_line_args=False,
            set_statics=flags.set_statics,
            set_dynamics=flags.set_dynamics,
            num_levels=flags.num_levels,
            any_custom_game=flags.any_custom_game,
            use_pytorch=True, paint_vel_info=0,
            is_high_res=flags.is_high_res,
            default_zoom=flags.default_zoom,
            float_obs=False)  # torchbeast divides by 255
        return CoinRunOneEnv('platform', 1,
                             default_zoom=flags.default_zoom, float_obs=False)
    else:
        return atari_wrappers.wrap_pytorch(
            atari_wrappers.wrap_deepmind(
                atari_wrappers.make_atari(env_name),
                clip_rewards=False,
                frame_stack=True,
                scale=False))
