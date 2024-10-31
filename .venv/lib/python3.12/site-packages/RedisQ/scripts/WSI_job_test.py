import sys
sys.path.append('../../')
from RedisQ.Base.Qlist import LQ, LD, HJ, HS
from RedisQ.Base.tools import fetch_job
from redis import Redis
from RedisQ.Base.Qjob import BaseJob
from RedisQ.scripts.WSIpayload import WSIJob
import requests
import time


def run_test(host='127.0.0.1', port='6389', num=1):
    conn = Redis(host=host, port=port)
    hs = HS(connection=conn)
    ld = LD(connection=conn, job_hash=HJ)
    lq = LQ(connection=conn, job_hash=HJ)

    for i, job in enumerate(lq):
        print(job.id)
        print(conn.hget('job_hash', job.id))
        job = fetch_job(conn, job.id)
        print(job)
        lq.pop()

    for _ in range(num):
        job = WSIJob()
        job.content.api = 'http://{host}/api/v1/'.format(host=host)
        job.content.patient = 'patients/'
        job.content.patient_id = '1'
        job.content.case = 'cases/'
        job.content.case_id = 'e354c56c-5d4e-4e04-80b1-86d6c7e91316'
        job.content.accession = 'accessions/'
        job.content.accession_id = '1'
        job.content.image = 'images/?accession_id=1&limit=1&ordering=id'
        job.content.cad = 'cads/'
        data = {
            "accession_id": 1,
            "job_id": str(job.id),
            "status": 'queued',
            "receive_time": None,
            "start_time": None,
            "end_time": None
        }
        requests.post(url='http://52.80.160.224/api/v1/tasks/', json=data)
        lq = LQ(connection=conn, job_hash=HJ)
        lq.queue(job)

    while True:

        print('queue work {}'.format(len(lq)))
        print('done work {}'.format(len(ld)))
        print('=================')
        time.sleep(5)
if __name__ == '__main__':
    run_test()
