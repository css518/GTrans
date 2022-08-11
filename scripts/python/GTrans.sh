#!/usr/bin/env bash

function make_dir () {
    if [[ ! -d "$1" ]]; then
        mkdir $1
    fi
}

SRC_DIR=../..
DATA_DIR=${SRC_DIR}/data
MODEL_DIR=${SRC_DIR}/python-tmp

make_dir $MODEL_DIR

DATASET=python
CODE_EXTENSION=original_subtoken
JAVADOC_EXTENSION=original


function train () {

echo "============TRAINING============"

RGPU=$1
MODEL_NAME=$2

PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$RGPU python -u -W ignore ${SRC_DIR}/main/train.py \
--data_workers 10 \
--dataset_name $DATASET \
--data_dir ${DATA_DIR}/ \
--model_dir $MODEL_DIR \
--model_name $MODEL_NAME \
--train_src train/code.${CODE_EXTENSION} \
--train_tgt train/javadoc.${JAVADOC_EXTENSION} \
--train_gnn train/train.0.gz \
--dev_src dev/code.${CODE_EXTENSION} \
--dev_tgt dev/javadoc.${JAVADOC_EXTENSION} \
--dev_gnn dev/dev.0.gz \
--uncase True \
--use_src_word True \
--use_src_char False \
--use_tgt_word True \
--use_tgt_char False \
--max_src_len 400 \
--max_tgt_len 30 \
--emsize 512 \
--fix_embeddings False \
--src_vocab_size 50000 \
--tgt_vocab_size 30000 \
--share_decoder_embeddings True \
--max_examples -1 \
--batch_size 32 \
--test_batch_size 64 \
--num_epochs 200 \
--model_type transformer \
--num_head 8 \
--d_k 64 \
--d_v 64 \
--d_ff 2048 \
--src_pos_emb False \
--tgt_pos_emb True \
--max_relative_pos 16 \
--use_neg_dist True \
--nlayers 6 \
--trans_drop 0.2 \
--dropout_emb 0.2 \
--dropout 0.2 \
--copy_attn True \
--early_stop 20 \
--warmup_steps 0 \
--optimizer adam \
--learning_rate 0.0001 \
--lr_decay 0.99 \
--valid_metric bleu \
--checkpoint True

}


function test () {

echo "============TESTING============"

RGPU=$1
MODEL_NAME=$2

PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$RGPU python -u -W ignore ${SRC_DIR}/main/train.py \
--only_test True \
--data_workers 10 \
--dataset_name $DATASET \
--data_dir ${DATA_DIR}/ \
--model_dir $MODEL_DIR \
--model_name $MODEL_NAME \
--dev_src test/code.${CODE_EXTENSION} \
--dev_tgt test/javadoc.${JAVADOC_EXTENSION} \
--dev_gnn test/test.0.gz \
--uncase True \
--max_src_len 400 \
--max_tgt_len 30 \
--max_examples -1 \
--test_batch_size 64

}

function beam_search () {

echo "============Beam Search TESTING============"

RGPU=$1
MODEL_NAME=$2

PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$RGPU python -u -W ignore ${SRC_DIR}/main/test.py \
--data_workers 10 \
--dataset_name $DATASET \
--data_dir ${DATA_DIR}/ \
--model_dir $MODEL_DIR \
--model_name $MODEL_NAME \
--dev_src test/code.${CODE_EXTENSION} \
--dev_tgt test/javadoc.${JAVADOC_EXTENSION} \
--dev_gnn test/test.0.gz \
--uncase True \
--max_examples -1 \
--max_src_len 400 \
--max_tgt_len 30 \
--test_batch_size 64 \
--beam_size 4 \
--n_best 1 \
--block_ngram_repeat 3 \
--stepwise_penalty False \
--coverage_penalty none \
--length_penalty none \
--beta 0 \
--gamma 0 \
--replace_unk

}

train 0 code2jdoc
test 0 code2jdoc
beam_search 0 code2jdoc
