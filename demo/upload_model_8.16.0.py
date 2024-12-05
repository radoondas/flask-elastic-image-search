import elasticsearch
from pathlib import Path
from eland.common import es_version
from eland.ml.pytorch import PyTorchModel
from eland.ml.pytorch.transformers import TransformerModel

ca_certs_path = "../app/conf/ca.crt"
es = elasticsearch.Elasticsearch("https://elastic:xB9OzFwRC9-NW4-Ypknf@127.0.0.1:9200",
                                 ca_certs=ca_certs_path,
                                 verify_certs=True)
es_cluster_version = es_version(es)

# Load a Hugging Face transformers model directly from the model hub
tm = TransformerModel(model_id="sentence-transformers/clip-ViT-B-32-multilingual-v1", task_type="text_embedding", es_version=es_cluster_version)


# Export the model in a TorchScrpt representation which Elasticsearch uses
tmp_path = "models"
Path(tmp_path).mkdir(parents=True, exist_ok=True)
model_path, config, vocab_path = tm.save(tmp_path)

# Import model into Elasticsearch
ptm = PyTorchModel(es, tm.elasticsearch_model_id())
ptm.import_model(model_path=model_path, config_path=None, vocab_path=vocab_path, config=config)
