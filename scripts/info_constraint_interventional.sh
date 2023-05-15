iterations=$1
num_envs=8
num_iterations=2000
replay_capacity=2000
prefill=100
num_samples_posterior=100
update_target_every=100
num_workers=4
output_folder=./results/info_constraint_interventional/info_constraint_interventional_1000
data=sachs_interventional
counter=1
while [[ $counter -le $iterations ]]; do
  python train.py --output_folder=$output_folder$counter --num_envs=$num_envs --num_iterations=$num_iterations --replay_capacity=$replay_capacity --prefill=$prefill --num_samples_posterior=$num_samples_posterior --update_target_every=$update_target_every --num_workers=$num_workers $data
  counter=$((counter+1))
done
