import subprocess
import sys

def run_local_milvus_windows():
    print("Pulling Milvus Docker image...")
    subprocess.run(["docker", "pull", "milvusdb/milvus:v2.3.9"], shell=True)
    # Check if container exists
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", "name=milvus-standalone", "--format", "{{.Names}}"],
        capture_output=True, text=True, shell=True
    )
    if "milvus-standalone" in result.stdout:
        print("Removing existing container 'milvus-standalone'...")
        subprocess.run(["docker", "rm", "-f", "milvus-standalone"], shell=True)
    print("Running Milvus Standalone on ports 19530 and 9091...")
    subprocess.run([
        "docker", "run", "-d", "--name", "milvus-standalone",
        "-p", "19530:19530", "-p", "9091:9091", "milvusdb/milvus:v2.3.9"
    ], shell=True)
    print("Milvus Standalone should now be running.")

if __name__ == "__main__":
    if sys.platform.startswith("win"):
        run_local_milvus_windows()
    else:
        print("This script is intended for Windows. Use run_local_milvus.py for other platforms.")
