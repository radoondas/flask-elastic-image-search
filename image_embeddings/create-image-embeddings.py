import os
import sys
import glob
import time
import json
import argparse
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from elasticsearch.helpers import parallel_bulk
from PIL import Image
import exifread
from lat_lon_parser import parse
from datetime import datetime


from exif import Image as exifImage


ES_HOST = "https://127.0.0.1:9200/"
ES_USER = "elastic"
ES_PASSWORD = "changeme"
ES_TIMEOUT = 3600

DEST_INDEX = "my-image-embeddings"
DELETE_EXISTING = True
CHUNK_SIZE = 100

PATH_TO_IMAGES = "../app/static/images/**/*.jp*g"
PREFIX = "../app/static/images/"

CA_CERT='ca.crt'

parser = argparse.ArgumentParser()
parser.add_argument('--es_host', dest='es_host', required=False, default=ES_HOST,
                    help="Elasticsearch hostname. Must include HOST and PORT. Default: " + ES_HOST)
parser.add_argument('--es_user', dest='es_user', required=False, default=ES_USER,
                    help="Elasticsearch username. Default: " + ES_USER)
parser.add_argument('--es_password', dest='es_password', required=False, default=ES_PASSWORD,
                    help="Elasticsearch password. Default: " + ES_PASSWORD)
parser.add_argument('--verify_certs', dest='verify_certs', required=False, default=True,
                    action=argparse.BooleanOptionalAction,
                    help="Verify certificates. Default: True")
parser.add_argument('--thread_count', dest='thread_count', required=False, default=4, type=int,
                    help="Number of indexing threads. Default: 4")
parser.add_argument('--chunk_size', dest='chunk_size', required=False, default=CHUNK_SIZE, type=int,
                    help="Default: " + str(CHUNK_SIZE))
parser.add_argument('--timeout', dest='timeout', required=False, default=ES_TIMEOUT, type=int,
                    help="Request timeout in seconds. Default: " + str(ES_TIMEOUT))
parser.add_argument('--delete_existing', dest='delete_existing', required=False, default=True,
                    action=argparse.BooleanOptionalAction,
                    help="Delete existing indices if they are present in the cluster. Default: True")
parser.add_argument('--ca_certs', dest='ca_certs', required=False, default=False,
                    help="CA certificates. Default: ca.crt")
parser.add_argument('--extract_GPS_location', dest='gps_location', required=False, default=False,
                    action=argparse.BooleanOptionalAction,
                    help="[Experimental] Extract GPS location from photos if available. Default: False")

args = parser.parse_args()

def main():
    global args
    lst = []

    start_time = time.perf_counter()
    img_model = SentenceTransformer('clip-ViT-B-32')
    duration = time.perf_counter() - start_time
    print(f'Duration load model = {duration}')

    counter = 0 
    total = 0

    start_time = time.perf_counter()
    for filename in glob.glob(PATH_TO_IMAGES, recursive=True):
      total = total + 1
    for filename in glob.glob(PATH_TO_IMAGES, recursive=True):
        print("Image: ", filename)
        image = Image.open(filename)

        doc = {}
        embedding = image_embedding(image, img_model)
        doc['image_id'] = create_image_id(filename)
        doc['image_name'] = os.path.basename(filename)
        doc['image_embedding'] = embedding.tolist()
        doc['relative_path'] = os.path.relpath(filename).split(PREFIX)[1]
        # Experimental! Extract photo GPS location if available.
        if args.gps_location:
            doc['exif'] = get_image_exif(filename)

        lst.append(doc)
        counter = counter + 1
        print(str(counter) + ' / ' + str(total))
        print("Progress: {0:.0%}".format(counter / total))

    duration = time.perf_counter() - start_time
    print(f'Duration creating image embeddings = {duration}')
    
    es = Elasticsearch(
        hosts=[args.es_host],
        verify_certs=args.verify_certs,
        basic_auth=(args.es_user, args.es_password),
    )
    if(args.ca_certs != False):
      es = Elasticsearch(
        hosts=[args.es_host],
        verify_certs=args.verify_certs,
        basic_auth=(args.es_user, args.es_password),
        ca_certs=args.ca_certs
    )
    es.options(request_timeout=args.timeout)

    # index name to index data into
    index = DEST_INDEX
    with open("image-embeddings-mappings.json", "r") as config_file:
        config = json.loads(config_file.read())
        if args.delete_existing:
            if es.indices.exists(index=index):
                print("Deleting existing %s" % index)
                es.indices.delete(index=index, ignore=[400, 404])

        print("Creating index %s" % index)
        es.indices.create(index=index,
                          mappings=config["mappings"],
                          settings=config["settings"],
                          ignore=[400, 404],
                          request_timeout=args.timeout)

    count = 0
    for success, info in parallel_bulk(
            client=es,
            actions=lst,
            thread_count=4,
            chunk_size=args.chunk_size,
            timeout='%ss' % 120,
            index=index
    ):
        if success:
            count += 1
            if count % args.chunk_size == 0:
                print('Indexed %s documents' % str(count), flush=True)
                sys.stdout.flush()
        else:
            print('Doc failed', info)

    print('Indexed %s documents' % str(count), flush=True)
    duration = time.perf_counter() - start_time
    print(f'Total duration = {duration}')
    print("Done!\n")


def image_embedding(image, model):
    return model.encode(image)


def create_image_id(filename):
    # print("Image filename: ", filename)
    return os.path.splitext(os.path.basename(filename))[0]

# Experimental functionality
def get_date_taken(path):
  date_string = Image.open(path)._getexif()[36867]
  date_object = datetime.strptime(date_string, "%Y:%m:%d %H:%M:%S")
  return date_object.isoformat()


def get_image_exif(filename):
    with open(filename, 'rb') as f:
        print("here")
        image = exifImage(f)
        exif = {} 
        if image.has_exif:
          try:
            taken = f"{image.datetime_original}"
            date_object = datetime.strptime(taken, "%Y:%m:%d %H:%M:%S")
            exif['date'] = date_object.isoformat()
            print(exif['date'])
          except:
            print("unable to get date")
          try:
            lat = dms_coordinates_to_dd_coordinates(image.gps_latitude, image.gps_latitude_ref)
            lon = dms_coordinates_to_dd_coordinates(image.gps_longitude, image.gps_longitude_ref)
            exif['location'] = [lon, lat]
            print(exif['location'])
          except:
            print("unable to get location")
        return exif


def dms_coordinates_to_dd_coordinates(coordinates, coordinates_ref):
    decimal_degrees = coordinates[0] + \
                      coordinates[1] / 60 + \
                      coordinates[2] / 3600
    
    if coordinates_ref == "S" or coordinates_ref == "W":
        decimal_degrees = -decimal_degrees
    
    return decimal_degrees

if __name__ == '__main__':
    main()
