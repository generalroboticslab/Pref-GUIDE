# Test

A collection of test scripts

### Stress Test

Configure executable paths in `stress.json`.\
Then run following command to spawn `30` machine clients and `2` human client:
```bash
python stress.py -nm 30 -nh 2
```

### Spawn Human interfaces

Spawn a human interface executable and align it across a monitor

First setup environment
```bash
python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

Spawn 4x4 interfaces on monitor 1 (second monitor)
```bash
python spawn.py {YOUR_EXECUTABLE_PATH} -rows 4 -cols 4 -monitor 1
```