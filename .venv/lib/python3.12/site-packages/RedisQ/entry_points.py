import click
import sys
import os
import time
import traceback
import requests
from redis import Redis
from RedisQ.GPUWorker import GPUWorker
from RedisQ.Base.Qlist import LQ, LW, LD, HS, HJ, LF
from RedisQ.scripts.DICOMpayload import DICOMConfig
from RedisQ.scripts.DICOMpayload import payload as dicom_payload
from RedisQ.scripts.DICOM_job_test import run_test as DICOM_test
from RedisQ.scripts.WSIpayload import WSIConfig
from RedisQ.scripts.WSIpayload import payload as wsi_payload
from RedisQ.scripts.WSI_job_test import run_test as WSI_test
from RedisQ.WatchDog import watchOn


@click.command()
@click.option('--type', '-t', default='dicom', help="options: dicom, wsi")
@click.option('--retry', '-r', default=3, help="options: a int num to retry max")
@click.option('--host', '-h', default='127.0.0.1', help="options: server host")
@click.option('--port', '-p', default='6389', help="options: server port")
@click.option('--silence', '-s', default=True, type=bool, help="options: true or false to print or not")
@click.option('--path', '-p', default='', help="options: a path to your Algorithm")
def cli(type, retry, host, port, silence, path):

    if path:
        sys.path.append(path)
        print(sys.path)

    if not silence:
        sys.stdout = sys.__stdout__
    else:
        sys.stdout = open('./worker.log', 'a+')

    retry_count = 0

    if type == 'dicom':
        while True:
            worker = None
            try:
                print('#'*20)
                retry_count += 1
                conn = Redis(host=host, port=port)
                config = DICOMConfig()
                config.res_base = os.path.abspath('.') + '/res'
                config.des_base = os.path.abspath('.') + '/des'
                worker = GPUWorker(host=host, connection=conn, config=config, job_hash=HJ, sign_hash=HS,
                                   queue_list=LQ, working_list=LW, done_list=LD, fail_list=LF)
                print('polling...')
                worker.polling(config=config, payload=dicom_payload)
            except KeyboardInterrupt as e:
                break
            except Exception as e:
                traceback.print_exc(file=sys.stdout)
            if retry_count >= retry - 1:
                try:
                    if worker:
                        job = worker.working_list.pop_to(worker.fail_list)
                        if not job:
                            continue
                        os.remove(worker.pid_file)
                        response = requests.get('http://{api}/api/v1/{tasks}/{job_id}/?format={format}'.format(
                                        tasks='tasks',
                                        job_id=job.id,
                                        format='json',
                                        api=host),headers=worker.headers)
                        data = {}
                        data.update(response.json())
                        data.update({
                            'status': 'failed',
                            'end_time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        })
                        response = requests.put('http://{api}/api/v1/{tasks}/{job_id}/?format={format}'.format(
                                        tasks='tasks',
                                        job_id=job.id,
                                        format='json',
                                        api=host),headers=worker.headers,json=data)
                        if response.status_code != 200:
                            traceback.print_exc(file=sys.stdout)
                except Exception as e:
                    traceback.print_exc(file=sys.stdout)
                retry_count = -1

    elif type == 'wsi':
        while True:
            worker = None
            try:
                print('#'*20)
                retry_count += 1
                conn = Redis(host=host, port=port)
                config = WSIConfig()
                config.res_base = os.path.abspath('.') + '/res'
                config.des_base = os.path.abspath('.') + '/des'
                worker = GPUWorker(host=host, connection=conn, config=config, job_hash=HJ, sign_hash=HS,
                                   queue_list=LQ, working_list=LW, done_list=LD, fail_list=LF)
                print('polling...')
                worker.polling(config=config, payload=wsi_payload)
            except KeyboardInterrupt as e:
                break
            except Exception as e:
                traceback.print_exc(file=sys.stdout)

            if retry_count >= retry:
                try:
                    if worker:
                        job = worker.working_list.pop_to(worker.fail_list)
                        if not job:
                            continue
                        os.remove(worker.pid_file)
                        response = requests.get('http://{api}/api/v1/{tasks}/{job_id}/?format={format}'.format(
                                        tasks='tasks',
                                        job_id=job.id,
                                        format='json',
                                        api=host),headers=worker.headers)
                        data = {}
                        data.update(response.json())
                        data.update({
                            'status': 'failed',
                            'end_time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        })
                        response = requests.put('http://{api}/api/v1/{tasks}/{job_id}/?format={format}'.format(
                                        tasks='tasks',
                                        job_id=job.id,
                                        format='json',
                                        api=host),headers=worker.headers,json=data)
                        if response.status_code != 200:
                            traceback.print_exc(file=sys.stdout)
                except Exception as e:
                    traceback.print_exc(file=sys.stdout)
                    retry_count = -1


@click.command()
@click.option('--api', '-a', default='127.0.0.1', help="options: api host")
@click.option('--path', '-p', default='.', help="options: the path you want to watch on")
def watchdog(path, api):
    watchOn(path, api)


@click.command()
@click.option('--type', '-t', default='dicom', help='options: dicom, wsi')
@click.option('--host', '-h', default='127.0.0.1', help="options: host host")
@click.option('--port', '-p', default='6389', help="options: server port")
@click.option('--num', '-n', default=1, help='options: an integer num for sending how many tasks to queue')
def test(host, type, port, num):
    if type == 'dicom':
        DICOM_test(host, port,  num)
    if type == 'wsi':
        WSI_test(host, port, num)
