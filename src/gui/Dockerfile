FROM python:3.10.14

RUN mkdir /app

COPY src/gui/requirements.txt /app/

WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

COPY src /app/src

EXPOSE 8501

CMD ["streamlit", "run", "src/gui/gui.py", "--server.port=8501", "--server.address=0.0.0.0"]
