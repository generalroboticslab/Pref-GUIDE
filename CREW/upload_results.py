import os
import tarfile
import paramiko
from scp import SCPClient

# ---------------- CONFIGURATION ----------------
hostname = 'bc298-cmp-06.egr.duke.edu'
port = 22
username = 'zj90'
password = 'JZR2679925804@jzr'

local_target_folder = 'Data'  # Local folder with .json files
server_base_folder = 'Preference_based_guide_cont/Results'
target_folder = 'ddpg_cont'  # This is the subfolder name on the server

archive_name = f'{target_folder}.tar.gz'
archive_path = os.path.join(local_target_folder, archive_name)

server_target_folder = f'{server_base_folder}'  # Full path on server
server_archive_path = f'{server_target_folder}/{archive_name}'
# ------------------------------------------------

# STEP 1: Create a tar.gz file of all .json files in the local directory
with tarfile.open(archive_path, "w:gz") as tar:
    for root, dirs, files in os.walk(local_target_folder):
        for filename in files:
            if filename.endswith('.json'):
                filepath = os.path.join(root, filename)
                # Preserve folder structure relative to local_target_folder
                arcname = os.path.relpath(filepath, local_target_folder)
                tar.add(filepath, arcname=arcname)

print(f"✅ Compressed .json files into: {archive_path}")

# STEP 2: Connect via SSH and SCP to transfer and unpack
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    client.connect(hostname, port=port, username=username, password=password)

    # Create folder on server (if it doesn't exist)
    client.exec_command(f"mkdir -p {server_target_folder}")

    # Upload the tar.gz archive into the target folder
    with SCPClient(client.get_transport()) as scp:
        scp.put(archive_path, server_archive_path)
    print(f"✅ Archive uploaded to: {server_archive_path}")

    # STEP 3: Run untar command on server
    command = f"cd {server_target_folder} && tar -xvzf {archive_name}"
    stdin, stdout, stderr = client.exec_command(command)

    output = stdout.read().decode()
    error = stderr.read().decode()

    stdout.close()
    stderr.close()
    stdin.close()

    if error:
        print(f"❌ Error extracting archive:\n{error}")
    else:
        print(f"✅ Archive extracted to: {server_target_folder}")
        print(output)

finally:
    client.close()
    print("🔒 SSH connection closed.")
