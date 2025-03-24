
file_path="/mnt/local/ii/retriever/patents"
index_file=$file_path/patents_all_index/specter2_IVF100,PQ16.index
corpus_file=$file_path/patents_all.parquet

# Encoder parameters
retrieval_method=specter2 # specter2 | e5
retriever_model=allenai/specter2_base # allenai/specter2_base | intfloat/e5-base-v2
adapter_name=allenai/specter2 # this adapter is for retrieval

# Retrieval parameters
distance_metric=l2 # ip | l2
faiss_type=IVF100,PQ16 # Flat | IVF100,Flat | IVF100,PQ16 | HNSW32
topk=3
max_length=1024
batch_size=512
pooling_method=cls
doc_type=patent

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python retrieval_server.py \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --topk $topk \
    --retriever_model $retriever_model \
    --retrieval_method $retrieval_method \
    --adapter_name $adapter_name \
    --distance_metric $distance_metric \
    --faiss_type $faiss_type \
    --max_length $max_length \
    --batch_size $batch_size \
    --pooling_method $pooling_method \
    --doc_type $doc_type \
    --faiss_gpu \
    --use_fp16