foreach ($algo in @('a2c','ppo','vpg50')) {
  python -m racing.train --algo $algo --seed 0 --steps 100000 --actor-lr 0.0001 --num-envs 4 --rollout 128 --batch-size 128 --eval-every 25000 --eval-episodes 5 --device cuda --out runs_tuning_lr1e4
}
