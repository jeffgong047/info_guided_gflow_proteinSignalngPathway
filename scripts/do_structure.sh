num_envs=1
num_iterations=1000
replay_capacity=1000
prefill=100
num_samples_posterior=100
update_target_every=100
num_workers=2
python train.py --num_envs=$num_envs --num_iterations=$num_iterations --replay_capacity=$replay_capacity --prefill=$prefill --num_samples_posterior=$num_samples_posterior --update_target_every=$update_target_every --num_workers=$num_workers do_structure 
