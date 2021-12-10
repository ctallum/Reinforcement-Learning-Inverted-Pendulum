# Machine Learning Final Project
# Reinforcement Learning Game Project - Swing Up Two Link Pendulum
from ray.rllib.agents.ppo import PPOTrainer

def main():
    # Configure the algorithm.
    config = {
        # Environment (RLlib understands openAI gym registered strings).
        "env": "CartPole-v1",
        "num_workers": 2,
        "framework": "torch",
        "model": {
            "fcnet_hiddens": [64, 64],
            "fcnet_activation": "relu",
            },
        "evaluation_num_workers": 1,
        "evaluation_config": {
            "render_env": True,
            }
    }
    # Create our RLlib Trainer.
    trainer = PPOTrainer(config=config)

    for _ in range(20):
        print(trainer.train())

    trainer.evaluate()

if __name__ == "__main__":
    main()