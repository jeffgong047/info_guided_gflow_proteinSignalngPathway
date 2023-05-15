num_envs=8
num_iterations=1000
replay_capacity=1000
prefill=100
num_samples_posterior=100
update_target_every=100
num_workers=4
output_folder=erdos_renyi_5_5_100
data=erdos_renyi_lingauss
batch_size=256
num_variables=5
num_edges=5
num_samples=100
info_constraint=True
python train.py --info_constraint=$info_constraint --batch_size=$batch_size --output_folder=$output_folder --num_envs=$num_envs --num_iterations=$num_iterations \
--replay_capacity=$replay_capacity --prefill=$prefill --num_samples_posterior=$num_samples_posterior \
--update_target_every=$update_target_every --num_workers=$num_workers $data --num_variables=$num_variables \
--num_edges=$num_edges --num_samples=$num_samples

grep 'INFO' erdos_renyi.log > erdos_renyii.log
#mv erdos_renyii.log ./logs/
#rm erdos_renyi.log
