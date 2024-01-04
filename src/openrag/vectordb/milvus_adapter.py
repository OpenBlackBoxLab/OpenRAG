from pymilvus import connections, utility
import os

def init_milvus_connection(alias="default"):
    """
    Initialize a connection to Milvus.

    Returns:
        None
    """
    connections.connect(host=os.environ["MILVUS_HOST"], port=os.environ["MILVUS_PORT"], alias=alias)
