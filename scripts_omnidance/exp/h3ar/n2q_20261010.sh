while [ ! -e /apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/H3AR/val_zsbase_bi_s12n50/.done_0 ]; do sleep 60; done
bash /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/h3ar/evalq_ha2.sh /apdcephfs_gy4/share_302533218/cedricnie/diff/models/train/H3A2_decfgDS_lr1e-4_n1cur HA2dslr1e-4_ '100 300 500 700'
