FROM python:3.10

WORKDIR /srv

RUN git clone https://github.com/su77ungr/CASALIOY.git
COPY ./requirements.txt /srv/
COPY ./README.md /srv/

RUN pip3 install -r /srv/requirements.txt --upgrade pip
RUN rm ./requirements.txt
COPY ./example.env /srv/CASALIOY/.env
