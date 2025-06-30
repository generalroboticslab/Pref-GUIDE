import os 
import paramiko
from scp import SCPClient

# Server details
hostname = 'bc298-cmp-06.egr.duke.edu'     # or IP address
port = 22                        # default SSH port
username = 'zj90'
password = 'JZR2679925804@jzr'      # Optional if using key

server_target_folder = 'Preference_based_guide_cont/CREW/crew-dojo'
local_target_folder = 'crew-dojo'

target_folder = 'Builds'

if not os.path.exists(local_target_folder):
    os.makedirs(local_target_folder)
    print(f"Created local target folder: {local_target_folder}")

# Create SSH client
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    # Connect using password
    client.connect(hostname, port=port, username=username, password=password)
    
    ### compress the ddpg_cont folder in server_target_folder 
    stdin, stdout, stderr = client.exec_command(f"cd {server_target_folder} && tar -cvzf {target_folder}.tar.gz {target_folder}")
    print(stdout.read().decode())

    ### scp to local_target_folder
    with SCPClient(client.get_transport()) as scp:
        scp.get(os.path.join(server_target_folder, f'{target_folder}.tar.gz'), os.path.join(local_target_folder, f'{target_folder}.tar.gz'))

    ### decompress the ddpg_cont folder in local_target_folder
    os.system(f"cd {local_target_folder} && tar -xvzf {target_folder}.tar.gz")
    print(f"Successfully transferred and extracted {target_folder} folder to {local_target_folder}")

    
finally:
    client.close()
