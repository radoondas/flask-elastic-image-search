{
  "settings": {
    "index.refresh_interval": "5s",
    "number_of_shards": 1
  },
  "mappings": {
    "properties": {
      "image_embedding": {
        "type": "dense_vector",
        "dims": 512,
        "index": true,
        "similarity": "cosine"
      },
      "image_id": {
        "type": "keyword"
      },
      "image_name": {
        "type" : "keyword"
      },
      "relative_path" : {
        "type" : "keyword"
      },
      "exif" : {
        "properties" : {
          "location": {
            "type": "geo_point"
          },
          "date": {
            "type": "date"
          }
        }
      }
    }
  }
}