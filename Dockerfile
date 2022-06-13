FROM python:3.9.5-slim-buster

RUN apt-get update

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN ls -la $APP_HOME/

RUN pip install -r requirements.txt

# Run the streamlit on container startup
# CMD [ "streamlit", "run","--server.enableCORS","false","beanleaf_classification_app.py" ]
# CMD [ "streamlit", "run","beanleaf_classification_app.py" ]
# gunicorn -w 1 -b 0.0.0.0:9876 wsgi:app
# CMD ["gunicorn", "--workers=2", "--chdir=.", "--bind", "0.0.0.0:5000", "--access-logfile=-", "--error-logfile=-", "app:app"]
# gunicorn --chdir app app:app -w 2 --threads 2 -b 0.0.0.0:80

CMD ["gunicorn", "--workers=2", "--bind", "0.0.0.0:9876", "wsgi:app"]