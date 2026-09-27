# Ship control allocation

Generate thruster commands, calculate their forces, train the neural network,
and compare it with the NLP allocator and direct force input.

## Setup

Run everything from this folder. If `.venv` is already set up, skip this step.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run the flow

Run these commands in order. Wait for each one to finish before starting the next.

```bash
# 1. Generate 100,000 smooth thruster commands.
.venv/bin/python generate_random_commands.py --samples 100000

# 2. Calculate the forces produced by those commands.
.venv/bin/python compute_random_forces.py

# 3. Train the network and evaluate the selected model on the test data.
.venv/bin/python train_neural_allocator.py --epochs 200

# 4. Compare the network, NLP allocator, and direct forces in simulation.
.venv/bin/python simulate_allocation.py --raw-neural
```
# RBE577-F26-F01-HW1
