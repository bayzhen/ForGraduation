# Unreal Engine Reinforcement Learning Plugin

This plugin enables you to integrate reinforcement learning in your Unreal Engine projects. It provides an interface between Unreal Engine and Python backend to communicate and control the training process.

## Class Diagram

Here's a class diagram showing the main classes in the plugin:

```plaintext
    +---------------------------+
    | UReinforcementLearningComponent |
    +---------------------------+
    ├─── URLAction
    ├─── URLState
    ├─── URLReward
    └─── URLDone

    +----------------+
    | RL_ENV_Controller |
    +----------------+
