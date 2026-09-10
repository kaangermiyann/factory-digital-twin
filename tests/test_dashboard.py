"""Dashboard duman testi.

Streamlit uygulamasi calisma aninda cok sey yapar: model yukler, tahmin okur,
hata hesaplar, grafik cizer. Bunlarin hicbiri import ile kontrol edilemez --
sayfa sozdizimsel olarak kusursuz olup ilk acilista patlayabilir. `AppTest`
betigi gercekten kosturur ve istisnalari yakalar.

Bu testin varlik sebebi: demo sirasinda ekranin acilmamasi, projede
yapilabilecek en pahali hatadir.
"""

from __future__ import annotations

from pathlib import Path

import pytest

DASHBOARD = Path(__file__).resolve().parents[1] / "src" / "twin" / "dashboard"
MAIN = DASHBOARD / "app.py"
PAGES = sorted(DASHBOARD.glob("pages/*.py"))


@pytest.fixture(scope="module")
def app_test():
    streamlit_testing = pytest.importorskip(
        "streamlit.testing.v1", reason="streamlit kurulu degil"
    )
    from twin.models.registry import load_all

    if not load_all():
        pytest.skip("Egitilmis model yok -- once `make train`")
    return streamlit_testing.AppTest


@pytest.fixture(scope="module")
def main(app_test):
    return app_test.from_file(str(MAIN), default_timeout=300).run()


def test_main_page_renders_without_exceptions(main):
    assert not main.exception, "\n".join(str(e.value) for e in main.exception)


def test_main_page_leads_with_comparison(main):
    """Ana ekran TEK bir soruya cevap vermeli: model olculeni tutturuyor mu?

    Model karsilastirma tablosu, ozellik onemi gibi egitim ici ciktilar burada
    OLMAMALI -- onlar alt sayfalarda.
    """
    text = " ".join(str(m.value) for m in main.markdown)
    assert "Gerçek vs Tahmin" in text
    assert "permutation" not in text.lower(), "egitim ici cikti ana ekrana sizmis"
    assert "naive" not in text.lower(), "model karsilastirmasi ana ekrana sizmis"


def test_main_page_states_the_data_was_unseen(main):
    """Grafigin hangi veri uzerinde ciziyor oldugu EKRANDA yazmali.

    Egitim verisi uzerinde cizilen bir 'gercek vs tahmin' grafigi guzel gorunur
    ve hicbir sey kanitlamaz. Bu ayrimi kullanicinin tahmin etmesi beklenemez.
    """
    banners = " ".join(str(s.value) for s in main.success) + \
              " ".join(str(w.value) for w in main.warning)
    assert "görmedi" in banners or "Eğitim dönemi" in banners, banners


def test_main_page_reports_error_metrics(main):
    labels = [m.label for m in main.metric]
    assert labels, "hata metrigi karti yok"
    deltas = [m.delta or "" for m in main.metric]
    assert any("MAPE" in d for d in deltas), deltas


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_subpages_render_without_exceptions(app_test, page):
    result = app_test.from_file(str(page), default_timeout=300).run()
    assert not result.exception, "\n".join(str(e.value) for e in result.exception)


def test_no_deprecated_api(app_test):
    """Kaldirilma tarihi gecmis API'ler sessizce calisir -- ta ki etmeyene kadar."""
    for path in [MAIN, *PAGES]:
        result = app_test.from_file(str(path), default_timeout=300).run()
        deprecated = [str(w.value) for w in result.warning if "deprecat" in str(w.value).lower()]
        assert not deprecated, f"{path.name}: {deprecated}"
