FROM python:3.10.13 AS builder
COPY ./requirements.txt /home
RUN pip install -r /home/requirements.txt

FROM python:3.10.13-slim-bullseye
EXPOSE 8000
WORKDIR /home
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY . /home
COPY ModelMerge /home/ModelMerge
COPY md2tgmd /home/md2tgmd
ENTRYPOINT ["python", "bot.py"]
