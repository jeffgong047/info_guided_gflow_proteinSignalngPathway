num_envs=8
num_iterations=1000
replay_capacity=1000
prefill=100
num_samples_posterior=100
update_target_every=100
num_workers=4
output_folder=info_constraint_nonlinear_gaussian
data=nonlinear_gaussian
info_constraint=True
python train.py --info_constraint=$info_constraint --output_folder=$output_folder --num_envs=$num_envs --num_iterations=$num_iterations --replay_capacity=$replay_capacity --prefill=$prefill --num_samples_posterior=$num_samples_posterior --update_target_every=$update_target_every --num_workers=$num_workers $data
