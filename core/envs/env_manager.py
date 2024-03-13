from gymnasium import register
class EnvManager:
    @staticmethod

    def register_env():
        # 最简单的环境
        register(
            id='CircuitEnvTest-v0',
            # entry_point='core.envs.circuit_env:CircuitEnv',
            entry_point='temp.env.env_test_v0:CircuitEnvTest_v0',
            max_episode_steps=2000000,
        )

        # 田字格 5 比特
        register(
            id='CircuitEnvTest-v1',
            # entry_point='core.envs.circuit_env:CircuitEnv',
            entry_point='temp.env.env_test_v1:CircuitEnvTest_v1',
            max_episode_steps=2000000,
        )

        # 代码精简，action space 和 obs space 重构
        register(
            id='CircuitEnvTest-v2',
            # entry_point='core.envs.circuit_env:CircuitEnv',
            entry_point='temp.env.env_test_v2:CircuitEnvTest_v2',
            max_episode_steps=4000000,
        )
        register(
            id='CircuitEnvTest-v3',
            # entry_point='core.envs.circuit_env:CircuitEnv',
            entry_point='temp.env.env_test_v3:CircuitEnvTest_v3',
            max_episode_steps=4000000,
        )

        register(
            id='CircuitEnvTest-v7',
            # entry_point='core.envs.circuit_env:CircuitEnv',
            entry_point='temp.env.env_test_v7:CircuitEnvTest_v7',
            max_episode_steps=4000000,
        )