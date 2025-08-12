```python
import torch
from stable_baselines3 import PPO

# Create an environment (you would replace this with your own environment)
env = "CartPole-v1"

# Create a model
model = PPO("MlpPolicy", env, verbose=1)

# Print the summary
print(model.summary())
```