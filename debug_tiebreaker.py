import numpy as np
from ztb.inference.decode import decode_action, InferenceConfig

logits = np.array([0.1, 0.0, 0.0])
mask = np.array([1, 0, 1])

action, info = decode_action(
    logits, mask, InferenceConfig(temperature=1.0, tiebreaker_tau=0.05, enable_tiebreaker=True)
)

print(f'Action: {action}')
print(f'Top2 actions: {info["top2_actions"]}')
print(f'Top2 probs: {info["top2_probs"]}')
print(f'Margin: {info["margin"]}')
print(f'Tiebreaker: {info["tiebreaker_activated"]}')
print(f'Probabilities: {info["probabilities"]}')
print(f'Mask: {mask}')
