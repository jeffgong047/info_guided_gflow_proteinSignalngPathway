num_envs=8
num_iterations=10000
replay_capacity=3000
prefill=500
num_samples_posterior=1000
update_target_every=500
num_workers=4
output_folder=info_constraint_synthetic_15_3_40_10000
data=synthetic15_3_observational
info_constraint=True
python train.py --info_constraint=$info_constraint --output_folder=$output_folder --num_envs=$num_envs --num_iterations=$num_iterations --replay_capacity=$replay_capacity --prefill=$prefill --num_samples_posterior=$num_samples_posterior --update_target_every=$update_target_every --num_workers=$num_workers $data
