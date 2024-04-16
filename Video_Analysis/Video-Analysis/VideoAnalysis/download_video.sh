set -e

DATA_DIR="/home/mq/data_disk2T/Hoang/AVA_Kinetics/video"
ANNO_DIR="/home/mq/data_disk2T/Hoang/AVA_Kinetics/Video_Analysis/Video-Analysis/anno"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} does not exist. Creating";
  mkdir -p ${DATA_DIR}
fi

# wget https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_trainval_v2.1.txt -P ${ANNO_DIR}

cat ${ANNO_DIR}/new.txt |
while read vid;
    do wget -c "https://s3.amazonaws.com/ava-dataset/trainval/${vid}" -P ${DATA_DIR}; done

echo "Downloading finished."