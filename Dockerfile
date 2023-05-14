FROM python:3.10

WORKDIR /srv

RUN git clone https://github.com/su77ungr/CASALIOY.git
RUN cd CASALIOY/

RUN python3 -m venv venv && . venv/bin/activate \
    && pip3 install -r /srv/CASALIOY/requirements.txt --upgrade pip

COPY ./example.env /srv/CASALIOY/.env
