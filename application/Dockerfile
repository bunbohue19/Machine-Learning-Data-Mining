FROM python:3.11-slim

RUN apt update && apt upgrade

WORKDIR /app

RUN adduser --disabled-password nonroot
RUN chown -R nonroot /app

USER nonroot

ENV PATH="/home/nonroot/.local/bin:$PATH"
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt --user

COPY . .

EXPOSE 7860
CMD [ "python3", "app.py" ]
