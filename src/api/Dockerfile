FROM python:3.10.14

RUN mkdir /app

COPY src/api/requirements.txt /app/

WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

COPY src /app/src

COPY models/lstm_model.h5 /app/src/api/
COPY models/lstm_cnn.h5 /app/src/api/
COPY src/api/utils/words_dict.pkl /app/src/api/utils/
COPY src/api/utils/tokenizer.pickle /app/src/api/utils/

EXPOSE 8001

CMD ["uvicorn", "src.api.api:app", "--host", "0.0.0.0", "--port", "8001"]
