import os
import subprocess
import sys

def setup_hadoop_classpath():
    """
    Retrieves the Hadoop classpath and sets the CLASSPATH environment variable.
    Exits the program on failure.
    """
    try:
        hadoop_classpath = subprocess.check_output(['hadoop', 'classpath']).decode('utf-8').strip()
        os.environ['CLASSPATH'] = hadoop_classpath
        print("Hadoop classpath:")
        print(hadoop_classpath)
    except subprocess.CalledProcessError as e:
        print("Failed to obtain Hadoop classpath:", e)
        sys.exit(1)