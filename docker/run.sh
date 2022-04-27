export DATA_PATH=net-ner-probing/data
export CODE_PATH=net-ner-probing/src
nvidia-docker run -ti --rm -v $CODE_PATH/:/src -v $DATA_PATH:/data --name net-ner-probing net-ner-probing /bin/bash
