foreach ($algo in @('a2c','ppo','vpg50')) {
  python -m racing.train --algo $algo --steps 4096 --num-envs 4 --rollout 128 --batch-size 128 --eval-every 4096 --eval-episodes 1 --out runs\benchmark
  if ($LASTEXITCODE -ne 0) { throw "Benchmark failed: $algo" }
}
python -m racing.benchmark --root runs\benchmark --steps 1000000 --seeds 3 --hourly-price 0