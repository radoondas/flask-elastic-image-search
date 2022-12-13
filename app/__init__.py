from flask import Flask
from config import Config
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch

app = Flask(__name__)
app.config.from_object(Config)
# Load model, run against the image and create image embedding
img_model = SentenceTransformer('clip-ViT-B-32')

es = Elasticsearch(hosts=app.config['ELASTICSEARCH_HOST'],
                   basic_auth=(app.config['ELASTICSEARCH_USER'], app.config['ELASTICSEARCH_PASSWORD']),
                   verify_certs= app.config['VERIFY_TLS'],
                   ca_certs='app/conf/ca.crt')

from app import routes

if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host="0.0.0.0", debug=True, port=5001)