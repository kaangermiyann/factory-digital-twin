# .venv varsa ONU kullan. Sebep: egitilmis modeller joblib/pickle ile saklanir
# ve baska bir sklearn surumunde yuklenemez. Sistem python'uyla `make test`
# kosturmak, dashboard testlerinin SESSIZCE atlanmasina yol acar.
# Baska bir yorumlayici icin: make PYTHON=/yol/python3 test
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
export PYTHONPATH := src

.DEFAULT_GOAL := help
.PHONY: start help setup sim live train train-fast score recipe api dash demo test clean elastic-up elastic-down reset

start:  ## Tek komutla her seyi kur ve dashboard'u ac (yeni gelen buradan baslasin)
	./start.sh

help:  ## Komutlari listele
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Bagimliliklari kur (tercihen ./start.sh kullanin -- venv'i o kurar)
	$(PYTHON) -m pip install -r requirements.txt

sim:  ## 90 gunluk sentetik gecmis uret
	$(PYTHON) -m twin.simulator.run history --days 90

live:  ## Canli akis (dashboard hareket etsin diye) -- ayri terminalde
	$(PYTHON) -m twin.simulator.run live --speed 60

train:  ## Tum hedefler icin model egit (~20 dk)
	$(PYTHON) -m twin.models.train

train-fast:  ## Hizli egitim (gelistirme icin, metrikler rapora konmaz)
	$(PYTHON) -m twin.models.train --fast

score:  ## Son 48 saati skorla (canli vs tahmin grafigi icin)
	$(PYTHON) -m twin.serving.scorer --backfill 48

recipe:  ## Altin recete tablosu uret (urun x ortam x tarife)
	$(PYTHON) -m twin.optimize.golden_recipe -o data/models/golden_recipe.csv

api:  ## FastAPI servisi (http://localhost:8000/docs)
	$(PYTHON) -m uvicorn twin.serving.api:app --host 0.0.0.0 --port 8000 --reload

dash:  ## Streamlit dashboard (http://localhost:8501)
	$(PYTHON) -m streamlit run src/twin/dashboard/app.py

demo: sim train score  ## Sifirdan calisir demo: veri + model + tahmin
	@echo ""
	@echo "  Hazir. Simdi:  make dash"
	@echo ""

test:  ## Testleri calistir
	$(PYTHON) -m pytest tests -q

elastic-up:  ## Elasticsearch + Kibana + MySQL ayaga kaldir
	docker compose up -d elasticsearch kibana mysql
	@echo "Elasticsearch: http://localhost:9200   Kibana: http://localhost:5601"
	@echo "Backend'i degistirmek icin: export TWIN_STORAGE__BACKEND=elastic"

elastic-down:  ## Servisleri durdur
	docker compose down

reset:  ## Uretilmis veri ve modelleri sil
	rm -rf data/processed data/models
	mkdir -p data/processed data/models

clean: reset  ## reset + gecici dosyalar
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete
