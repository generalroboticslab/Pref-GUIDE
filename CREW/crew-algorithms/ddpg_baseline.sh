export PYTHONPATH=:$PYTHONPATH
source ./utils.sh

run_ddpg_baseline1 "42 43 44 45 46" "hide_and_seek_1v1" "True"
run_ddpg_baseline1 "42 43 44 45 46" "hide_and_seek_1v1" "False"

source ./utils.sh

run_ddpg_baseline "1 2 7 10 13 15 17 18 25 30 32 33 37 39 44" "hide_and_seek_1v1" "True"
run_ddpg_baseline "1 2 7 10 13 15 17 18 25 30 32 33 37 39 44" "hide_and_seek_1v1" "False"