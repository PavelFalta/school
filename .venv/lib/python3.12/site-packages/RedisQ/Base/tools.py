from RedisQ.Base.Qlist import HJ
import uuid

def fetch_job(connection, uuid):
    job_hash = HJ(connection)
    return job_hash.fetch(uuid)
