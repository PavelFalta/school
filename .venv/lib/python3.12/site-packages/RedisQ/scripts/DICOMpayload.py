import sys
from RedisQ.GPUWorker import GPUWorker
from redis import Redis
from RedisQ.Base.Qjob import BaseJob
from RedisQ.Base.Qlist import LQ, LW, LD, HS, HJ, LF
from pyseri.serializer import serializer
from threading import Thread
from imp import reload
import json
import csv
import requests
import os


class DICOMPredictionContent(object):
    # those data is used for testing
    api = 'http://192.168.1.150:8000/api/v1/'
    image = 'images/?accession_id=4&limit=200&ordering=id'
    api = 'http://192.168.1.150:8000/api/v1/'
    patient = 'patients/'
    patient_id = '1'
    case = 'cases/'
    case_id = '2351cb63-ea0a-4f40-8fed-5a82761adb7e'
    accession = 'accessions/'
    accession_id = '1'
    cad = 'cads/'


class DICOMJob(BaseJob):

    content = DICOMPredictionContent()


class DICOMConfig(object):

    gpu_load = 0.3
    gpu_memory = 0.3
    docker_path = './test'
    docker_image_tag = 'test_work'
    res_base = 'res'
    des_base = 'des'
    job = DICOMJob

    @property
    def gpu_fraction(self):
        return (self.gpu_memory + self.gpu_load) * 0.5


def payload(job, config):

    if not isinstance(job, DICOMJob):
        raise Exception('not DICOMJob')

    print('playload running...')
    # get job
    content = job.content
    # get api information
    base = '{base}/{accession_id}/'.format(
        base=config.res_base,
        accession_id=content.accession_id
    )

    res = '{base}{accession_id}/'.format(
        base=base,
        accession_id=content.accession_id,
    )

    if not os.path.isdir(base):
        print('create path')
        os.mkdir(base)

    if not os.path.isdir(res):
        print('create path')
        os.mkdir(res)

    print('get resource info')

    response = requests.get(content.api + content.image)

    if response.status_code != 200:
        raise Exception("can't get resource information", response.content)

    for image in response.json().get('results'):
        raw_uri = image.get('raw_file_path', None)
        id = image.get('id', None)
        if raw_uri and id is None:
            raise Exception(
                "can get resource information 'cause raw_uri or id is not in the response")
        # judge if the raw file has been down
        filename = raw_uri.split('/')[-1]
        res = '{base}{accession_id}/{id}_{filename}'.format(
            base=base,
            accession_id=content.accession_id,
            id=id,
            filename=filename
        )
        if not os.path.isfile(res):
            print('download raw file')
            response = requests.get(raw_uri)
            if response.status_code != 200:
                raise Exception(
                    "can't download raw files {},{}".format(raw_uri, id))
        # set file path
        with open(res, 'wb+') as f:
            f.write(response.content)

    print('ready to predict')
    des = '{base}/{accession_id}/'.format(
        base=config.des_base,
        accession_id=content.accession_id
    )
    # write content
    # make destenation path if there not be
    if not os.path.isdir(des):
        os.mkdir(des)
    # run algorithm if there is not result

    files = os.listdir(des)
    if not files:
        print('prediction...')
        from DICOM_AlgorithmAPI import api
        reload(api)
        api.main(data_dir=base, work_dir=des)
 
    # create metadata
    metadata = job.create_metadata()
    for file_name in files:
        if file_name.endswith('png') or file_name.endswith('jpg') or file_name.endswith('jpeg'):
            metadata.images.append(file_name)
        if file_name.endswith('csv'):
            file_name_noendfix = file_name.split('.')[0]
            with open('{des}/{file_name}'.format(des=des, file_name=file_name), 'r') as f:
                tag_name = '{file_name}_csv'.format(
                    file_name=file_name_noendfix)
                metadata.features[tag_name] = []
                reader = csv.DictReader(f)
                for row in reader:
                    metadata.features[tag_name].append(row)
        if file_name.endswith('txt'):
            file_name_noendfix = file_name.split('.')[0]
            with open('{des}/{file_name}'.format(des=des, file_name=file_name), 'r') as f:
                tag_name = '{file_name}_txt'.format(file_name=file_name_noendfix)
                result = f.read()
                metadata.features[tag_name] = result

    if not os.path.isfile('{des}/metadata.json'.format(des=des)):
        print('write metadata')
        with open('{des}/metadata.json'.format(des=des), 'w+') as f:
            j = serializer.dump(metadata)
            f.write(json.dumps(j))
    print('payload done')
    print('polling...')


def run_worker(host, port):
    conn = Redis(host=host, port=port)
    config = DICOMConfig()
    worker = GPUWorker(connection=conn, config=config, job_hash=HJ, sign_hash=HS,
                       queue_list=LQ, working_list=LW, done_list=LD, fail_list=LF)
    print('polling...')
    worker.polling(config=config, payload=payload)

if __name__ == '__main__':
    while True:
        try:
            run_worker(host='127.0.0.1', port='6389')
        except KeyboardInterrupt as e:
            break
        except Exception as e:
            print(e)
            with open('./worker.log', 'a+') as f:
                f.write(str(e))
