export PYTHONPATH=:$PYTHONPATH
source ./utils.sh

run_ddpg_baseline1 "42 43 44 45 46" "hide_and_seek_1v1" "True"
run_ddpg_baseline1 "42 43 44 45 46" "hide_and_seek_1v1" "False"
