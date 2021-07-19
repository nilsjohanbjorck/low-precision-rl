COMMON_TRAIN_FORMAT = [
    ('episode', 'E', 'int'),
    ('step', 'S', 'int'),
    ('episode_reward', 'R', 'float'),
    ('duration', 'D', 'time')
]

COMMON_EVAL_FORMAT = [
    ('episode', 'E', 'int'),
    ('step', 'S', 'int'),
    ('episode_reward', 'R', 'float')
]


AGENT_TRAIN_FORMAT = {
    'sac': [
        ('batch_reward', 'BR', 'float'),
        ('actor_loss', 'ALOSS', 'float'),
        ('critic_loss', 'CLOSS', 'float'),
        ('alpha_loss', 'TLOSS', 'float'),
        ('alpha_value', 'TVAL', 'float'),
        ('actor_entropy', 'AENT', 'float')
    ]
}
