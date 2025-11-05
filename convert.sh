# 进入你的正样本目录
# cd dataset/hey_aldelo/positive_train
# for f in *.wav; do
#   ffmpeg -y -i "$f" -ac 1 -ar 16000 -acodec pcm_s16le "/home/ubuntu/work/zxs/wake_word/openwake_test/dataset/hey_aldelo_pcm/positive_train/$f"
# done

# 进入负样本目录
cd /home/ubuntu/work/zxs/wake_word/openwake_test/dataset/negative_data/negative_test
for f in *.wav; do
  ffmpeg -y -i "$f" -ac 1 -ar 16000 -acodec pcm_s16le "/home/ubuntu/work/zxs/wake_word/openwake_test/dataset/negative_data_pcm/negative_test/$f"
done