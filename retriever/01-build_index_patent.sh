corpus_file='/mnt/local/ii/retriever/patents/patents_all.parquet' # parquet
save_dir='/mnt/local/ii/retriever/patents/patents_all_index'

# Encoder parameters
retriever_name=specter2 # specter2 | e5
retriever_model=allenai/specter2_base # allenai/specter2_base | intfloat/e5-base-v2
adapter_name=allenai/specter2 # this adapter is for retrieval

# FAISS DB parameters
distance_metric=l2 # ip | l2
faiss_type=IVF100,PQ16 # Flat | IVF100,Flat | IVF100,PQ16 | HNSW32

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python index_builder.py \
    --retrieval_method $retriever_name \
    --model_path $retriever_model \
    --adapter_name $adapter_name \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 256 \
    --batch_size 512 \
    --pooling_method cls \
    --distance_metric $distance_metric \
    --faiss_type $faiss_type \
    --save_embedding