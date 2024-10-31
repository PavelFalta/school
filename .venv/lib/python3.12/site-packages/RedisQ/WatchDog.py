# Notifier example from tutorial
#
# See: http://github.com/seb-m/pyinotify/wiki/Tutorial
#
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler, FileSystemEventHandler, DirCreatedEvent, FileCreatedEvent
import sys
import time
import os
import logging
import json
import functools
import requests

watch_dir_name = 'workout'
watch_dir = './{0}/DICOM/'.format(watch_dir_name)


class my_handler(FileSystemEventHandler):

    def on_any_event(self, event):
        print(event)
        if not isinstance(event, FileCreatedEvent):
            return
        print("Create:{0}".format(event.src_path))
        base_path = '/'.join(event.src_path.split('/')[:-1])
        file_name = '/'.join(event.src_path.split('/')[-1:])
        if file_name == 'metadata.json':
            # if metadata.json create
            with open(event.src_path, 'r') as f:
                _json = f.read()
                j = json.loads(_json)
                job = j.get('job', None)
                api = job['api']
                cad_api = api + job['cad']
                image_api = api + job['image']
                accession_id = job['accession_id']

                if not job:
                    raise Exception("no job content")

                features = j.get('features', None)
                images = j.get('images', None)
                if features:
                    # upload features
                    headers = {
                        "Authorization": "Basic YWRtaW46YWRtaW4xMjM0NTY="}
                    response = requests.post(cad_api, headers=headers, json={
                                             "accession_id": accession_id, "result": features})
                    print(response.content)
                    if response.status_code != 201:
                        raise Exception('upload err!')
                    cad_id = response.json().get('id', None)
                    if cad_id:
                        if images:
                            for index, image in enumerate(images):
                                image_type = 'ROI' if image.split(
                                    '.')[-2].endswith('roi') else 'HEATMAP'
                                image = '/' + image
                                headers = {
                                    "Authorization": "Basic YWRtaW46YWRtaW4xMjM0NTY="}
                                data = {
                                    'cad_id': cad_id,
                                    'accession_id': accession_id,
                                    'type': image_type,
                                    'index': index,
                                }
                                files = [
                                    ('raw_file_path', (image, open(base_path + image, 'rb'), 'image/png'))]
                                response = requests.post(
                                    image_api, headers=headers, data=data, files=files)
                                print(response.content)
                                if response.status_code != 201:
                                    raise Exception('upload err!')


def watchOn(path, api):
    API = api
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    # event_handler = LoggingEventHandler()
    event_handler = my_handler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    print('watch on {}'.format(path))
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else watch_dir
    watchOn(path, '127.0.0.1')
