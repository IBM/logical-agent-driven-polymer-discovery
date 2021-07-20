from gym.envs.registration import register

register(
    id='eADPD-v1',
    entry_point='md_envs.envs:AgentDrivenPolymerDiscovery',
)

register(
    id='logical-eADPD-v1',
    entry_point='md_envs.envs:LogicalAgentDrivenPolymerDiscovery',
)

register(
    id='direct-eADPD-v1',
    entry_point='md_envs.envs:DirectAgentDrivenPolymerDiscovery',
)