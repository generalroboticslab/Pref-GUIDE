export PYTHONPATH=:$PYTHONPATH
source ./utils.sh

eval_ddpg_local "1 2 7 10 13 15 17 18 25 30 32 33 37 39 44" "find_treasure" "vote_ref"
eval_ddpg_local "1 2 7 10 13 15 17 18 25 30 32 33 37 39 44" "hide_and_seek_1v1" "vote_ref"

