export PYTHONPATH=:$PYTHONPATH
source ./utils.sh

eval_ddpg "1 2 7 10 13 15 17 18" "find_treasure" "guide_original"

