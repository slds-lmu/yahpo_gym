FROM python:3.8-slim
WORKDIR /root
RUN \ 
  apt update -y && \
  apt install build-essential git wget -y && \
  /usr/local/bin/python -m pip install --upgrade pip && \
  pip install yahpo-gym && \
  git clone --depth 1 -b main https://github.com/slds-lmu/yahpo_data.git && \
  chown -R root yahpo_data && \
  mkdir ~/.config/ && \
  printf "data_path: ~/yahpo_data\n" >> ~/.config/yahpo_gym && \
  rm -rf /var/lib/apt/lists/* && \
  pip cache purge
COPY --chown=root:root scripts/yahpo.py yahpo.py
ENTRYPOINT ["/bin/bash"]
