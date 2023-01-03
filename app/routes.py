from app import app, img_model, es
from flask import render_template, redirect, url_for, request, send_file
from app.searchForm import SearchForm
from app.inputFileForm import InputFileForm
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import elasticsearch
import os
from PIL import Image

INFER_ENDPOINT = "/_ml/trained_models/{model}/deployment/_infer"
INFER_MODEL_IM_SEARCH = 'sentence-transformers__clip-vit-b-32-multilingual-v1'

INDEX_IM_EMBED = 'my-image-embeddings'

HOST = app.config['ELASTICSEARCH_HOST']
AUTH = (app.config['ELASTICSEARCH_USER'], app.config['ELASTICSEARCH_PASSWORD'])
HEADERS = {'Content-Type': 'application/json'}

TLS_VERIFY = app.config['VERIFY_TLS']

app_models = {}


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', title='Home')


@app.route('/image_search', methods=['GET', 'POST'])
def image_search():
    global app_models
    is_model_up_and_running(INFER_MODEL_IM_SEARCH)

    index_name = INDEX_IM_EMBED
    if not es.indices.exists(index=index_name):
        return render_template('image_search.html', title='Image search', model_up=False,
                               index_name=index_name, missing_index=True)

    if app_models.get(INFER_MODEL_IM_SEARCH) == 'started':
        form = SearchForm()

        # Check for  method
        if request.method == 'POST':

            if 'find_similar_image' in request.form and request.form['find_similar_image'] is not None:
                image_id_to_search_for = request.form['find_similar_image']
                form.searchbox.data = None

                image_info = es.search(
                    index=INDEX_IM_EMBED,
                    query={
                        "term": {
                            "image_id": {
                                "value": image_id_to_search_for,

                                "boost": 1.0
                            }
                        }
                    },
                    source=True)

                if (image_info is not None):

                    found_image = image_info['hits']['hits'][0]["_source"]
                    found_image_embedding = found_image['image_embedding']
                    search_response = knn_search_images(
                        found_image_embedding)

                    return render_template('image_search.html', title='Image Search', form=form,
                                           search_results=search_response['hits']['hits'],
                                           query=form.searchbox.data, model_up=True,
                                           image_id_to_search_for=image_id_to_search_for)

            if form.validate_on_submit():
                embeddings = sentence_embedding(form.searchbox.data)
                search_response = knn_search_images(embeddings['predicted_value'])

                return render_template('image_search.html', title='Image search', form=form,
                                       search_results=search_response['hits']['hits'],
                                       query=form.searchbox.data,  model_up=True)

            else:
                return redirect(url_for('image_search'))
        else:  # GET
            return render_template('image_search.html', title='Image search', form=form, model_up=True)
    else:
        return render_template('image_search.html', title='Image search', model_up=False, model_name=INFER_MODEL_IM_SEARCH)


@app.route('/similar_image', methods=['GET', 'POST'])
def similar_image():
    index_name = INDEX_IM_EMBED
    if not es.indices.exists(index=index_name):
        return render_template('similar_image.html', title='Similar image', index_name=index_name, missing_index=True)

    is_model_up_and_running(INFER_MODEL_IM_SEARCH)

    if app_models.get(INFER_MODEL_IM_SEARCH) == 'started':
        form = InputFileForm()
        if request.method == 'POST':
            if form.validate_on_submit():
                if request.files['file'].filename == '':
                    return render_template('similar_image.html', title='Similar image', form=form,
                                           err='No file selected', model_up=True)

                filename = secure_filename(form.file.data.filename)

                url_dir = 'static/tmp-uploads/'
                upload_dir = 'app/' + url_dir
                upload_dir_exists = os.path.exists(upload_dir)
                if not upload_dir_exists:
                    # Create a new directory because it does not exist
                    os.makedirs(upload_dir)

                # physical file-dir path
                file_path = upload_dir + filename
                # relative file path for URL
                url_path_file = url_dir + filename
                # Save the image
                form.file.data.save(upload_dir + filename)

                image = Image.open(file_path)
                embedding = image_embedding(image, img_model)

                # Execute KN search over the image dataset
                search_response = knn_search_images(embedding.tolist())

                # Cleanup uploaded file after not needed
                # if os.path.exists(file_path):
                #     os.remove(file_path)

                return render_template('similar_image.html', title='Similar image', form=form,
                                       search_results=search_response['hits']['hits'],
                                       original_file=url_path_file, model_up=True)
            else:
                return redirect(url_for('similar_image'))
        else:
            return render_template('similar_image.html', title='Similar image', form=form, model_up=True)
    else:
        return render_template('similar_image.html', title='Similar image', model_up=False,
                               model_name=INFER_MODEL_IM_SEARCH)


@app.route('/image/<path:image_name>')
def get_image(image_name):
    try:
        # Use os.path.join to handle subdirectories
        image_path = os.path.join('./static/images/', image_name)
        return send_file(image_path, mimetype='image/jpg')
    except FileNotFoundError:
        return 'Image not found.'


@app.errorhandler(413)
@app.errorhandler(RequestEntityTooLarge)
def app_handle_413(e):
    return render_template('error.413.html', title=e.name, e_name=e.name, e_desc=e.description,
                           max_bytes=app.config["MAX_CONTENT_LENGTH"])


def sentence_embedding(query: str):
    response = es.ml.infer_trained_model(model_id=INFER_MODEL_IM_SEARCH, docs=[{"text_field": query}])
    return response['inference_results'][0]


def knn_search_images(dense_vector: list):
    source_fields = ["image_id", "image_name", "relative_path"]
    query = {
        "field": "image_embedding",
        "query_vector": dense_vector,
        "k": 5,
        "num_candidates": 10
    }

    response = es.search(
        index=INDEX_IM_EMBED,
        fields=source_fields,
        knn=query, source=False)

    return response


def infer_trained_model(query: str, model: str):
    response = es.ml.infer_trained_model(model_id=model, docs=[{"text_field": query}])
    return response['inference_results'][0]


def image_embedding(image, model):
    return model.encode(image)


def is_model_up_and_running(model: str):
    global app_models

    try:
        rsp = es.ml.get_trained_models_stats(model_id=model)
        if "deployment_stats" in rsp['trained_model_stats'][0]:
            app_models[model] = rsp['trained_model_stats'][0]['deployment_stats']['state']
        else:
            app_models[model] = 'down'
    except elasticsearch.NotFoundError:
        app_models[model] = 'na'
