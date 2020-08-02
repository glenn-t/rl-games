# openspiel made by cloning repo and building the base docker container
# as detailed here: https://github.com/deepmind/open_spiel/blob/master/docs/install.md
FROM openspiel:latest
WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt
