mass_folder=/path/to/mass
data_folder=/path/to/data
dumped_folder=/path/to/dumped

# only needed for step 4 
bidict_file=/path/to/bidict.dsb-hsb

step1 () {
    python $mass_folder/train.py --exp_name mass_unmt \
    --dump_path $dumped_folder \
    --data_path $data_folder/  --lgs "cs-de-dsb-hsb" \
    --mass_steps "cs,de" \
    --mt_steps "cs-de,de-cs" \
    --encoder_only false \
    --emb_dim 1024 --n_layers 6 --n_heads 8 \
    --dropout 0.1 --attention_dropout 0.1 \
    --gelu_activation true \
    --tokens_per_batch 3000 \
    --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
    --epoch_size 100000 \
    --validation_metrics valid_cs-de_mt_bleu \
    --stopping_criterion valid_cs-de_mt_bleu,10  \
    --reload_model "$model" \
    --eval_bleu true \
    --word_mass 0.5 --min_len 5
}

step2 () {
    model=$1
    python $mass_folder/train.py --exp_name mass_unmt \
    --dump_path $dumped_folder \
    --data_path $data_folder/  --lgs "cs-de-dsb-hsb" \
    --mass_steps "de,hsb" \
    --mt_steps "hsb-de,de-hsb" \
    --encoder_only false \
    --emb_dim 1024 --n_layers 6 --n_heads 8 \
    --dropout 0.1 --attention_dropout 0.1 \
    --gelu_activation true \
    --tokens_per_batch 3000 \
    --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
    --epoch_size 100000 \
    --validation_metrics valid_hsb-de_mt_bleu \
    --stopping_criterion valid_hsb-de_mt_bleu,10 \
    --reload_model "$model" \
    --eval_bleu true \
    --word_mass 0.5 --min_len 5
}

step4 () {
    model=$1
    python $xlm_folder/train.py --exp_name mass_unmt \
    --dump_path $dumped_folder \
    --reload_model "$model" \
    --data_path $data_folder/ --lgs "cs-de-dsb-hsb" \
    --bt_steps "de-dsb-de,dsb-de-dsb,dsb-hsb-dsb,hsb-dsb-hsb" \
    --ae_steps "de,dsb,hsb" --xbt_steps "de-dsb-hsb,hsb-dsb-de" \
    --dont_eval_langs "dsb-hsb,hsb-dsb" \
    --word_shuffle 3 --word_dropout 0.1 --word_blank 0.1 \
    --lambda_ae '0:1,100000:0.1,300000:0' \
    --encoder_only false \
    --emb_dim 1024 --n_layers 6 --n_heads 8 \
    --dropout 0.1 --attention_dropout 0.1 \
    --gelu_activation true \
    --tokens_per_batch 3000 \
    --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
    --epoch_size 100000 \
    --eval_bleu true \
    --stopping_criterion valid_dsb-de_mt_bleu,10 \
    --validation_metrics valid_dsb-de_mt_bleu \
    --tie_lang_embs "dsb,hsb" --transfer_vocab $bidict_file
}

step5 () {
    model=$1
    python $xlm_folder/train.py --exp_name mass_unmt \
    --dump_path $dumped_folder \
    --data_path $data_folder/  --lgs "cs-de-dsb-hsb" \
    --mass_steps "de,dsb,hsb" \
    --mt_steps "de-dsb,dsb-de,hsb-de,de-hsb" \
    --encoder_only false \
    --emb_dim 1024 --n_layers 6 --n_heads 8 \
    --dropout 0.1 --attention_dropout 0.1 \
    --gelu_activation true \
    --tokens_per_batch 3000 \
    --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
    --epoch_size 100000 \
    --validation_metrics valid_dsb-de_mt_bleu \
    --stopping_criterion valid_dsb-de_mt_bleu,10 \
    --reload_model "$model" \
    --eval_bleu true \
    --word_mass 0.5 --min_len 5 
}

step6 () {
    model=$1
    python $xlm_folder/train.py --exp_name mass_unmt \
    --dump_path $dumped_folder \
    --reload_model "$model" \
    --data_path $data_folder/ --lgs "cs-de-dsb-hsb" \
    --bt_steps "de-dsb-de,dsb-de-dsb,dsb-hsb-dsb,hsb-dsb-hsb" \
    --ae_steps "de,dsb,hsb" --xbt_steps "de-dsb-hsb,hsb-dsb-de" \
    --dont_eval_langs "dsb-hsb,hsb-dsb" \
    --word_shuffle 3 --word_dropout 0.1 --word_blank 0.1 \
    --lambda_ae '0:1,100000:0.1,300000:0' \
    --encoder_only false \
    --emb_dim 1024 --n_layers 6 --n_heads 8 \
    --dropout 0.1 --attention_dropout 0.1 \
    --gelu_activation true \
    --tokens_per_batch 3000 \
    --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
    --epoch_size 100000 \
    --eval_bleu true \
    --stopping_criterion valid_dsb-de_mt_bleu,10 \
    --validation_metrics valid_dsb-de_mt_bleu 
}

"$@"