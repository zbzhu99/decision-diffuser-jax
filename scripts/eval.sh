logdirs=(
  # "logs/diffuser_inv_d4rl/hopper-medium-expert-v2/h_20-r_400.0/100" \
  # "logs/diffuser_inv_d4rl/hopper-medium-expert-v2/h_20-r_400.0/200" \
  # "logs/diffuser_inv_d4rl/hopper-medium-expert-v2/h_20-r_400.0/300" \

  # "logs/diffuser_inv_d4rl/hopper-medium-replay-v2/h_20-r_350.0/100" \
  # "logs/diffuser_inv_d4rl/hopper-medium-replay-v2/h_20-r_350.0/200" \
  # "logs/diffuser_inv_d4rl/hopper-medium-replay-v2/h_20-r_350.0/300" \

  "logs/diffuser_inv_d4rl/hopper-medium-v2/h_20-r_350.0/100" \
  "logs/diffuser_inv_d4rl/hopper-medium-v2/h_20-r_350.0/200" \
  "logs/diffuser_inv_d4rl/hopper-medium-v2/h_20-r_350.0/300" \
)

while getopts ":g:" opt; do
  case $opt in
    g)
      gpu_index=$OPTARG
      ;;
    \?)
      echo "Unknown options: -$OPTARG" >&2
      ;;
  esac
done

for logdir in "${logdirs[@]}"; do
    eval $"python evaluate.py $logdir --epochs 0 100 200 300 400 600 700 800 900 1000 -g $gpu_index"
done