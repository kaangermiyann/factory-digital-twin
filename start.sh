#!/usr/bin/env bash
#
# Fabrika Dijital Ikizi -- tek komutla lokal kurulum ve calistirma.
#
#   ./start.sh                 kurulum + veri + model + dashboard
#   ./start.sh --fast          hizli egitim (~1 dk) -- metrikler rapora konmaz
#   ./start.sh --fresh         her seyi sifirdan uret
#   ./start.sh --no-dash       dashboard'u acma, sadece hazirla
#   ./start.sh --days 30       kac gunluk sentetik gecmis uretilsin
#   ./start.sh --port 8600     dashboard portu
#
# Script FIKIRLIDIR: her adim zaten yapilmissa atlanir. Ikinci calistirmada
# dogrudan dashboard acilir.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# --- ayarlar ---------------------------------------------------------------- #
VENV=".venv"
DAYS=90
PORT=8501
FAST=""
FRESH=0
LAUNCH=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fast)    FAST="--fast";  shift ;;
    --fresh)   FRESH=1;        shift ;;
    --no-dash) LAUNCH=0;       shift ;;
    --days)    DAYS="$2";      shift 2 ;;
    --port)    PORT="$2";      shift 2 ;;
    # Basligi satir numarasiyla degil, ilk kod satirina kadar okuyarak bas:
    # dosya degisince yardim metni sessizce bozulmasin.
    -h|--help) awk '/^#!/{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "$0"; exit 0 ;;
    *) echo "Bilinmeyen secenek: $1  (--help)"; exit 1 ;;
  esac
done

step()  { printf '\n\033[1;36m▸ %s\033[0m\n' "$*"; }
info()  { printf '  %s\n' "$*"; }
ok()    { printf '  \033[0;32m✓\033[0m %s\n' "$*"; }
die()   { printf '\n\033[0;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# --- 1. Python -------------------------------------------------------------- #
# Not: `python3` sistemde birden fazla surume isaret edebiliyor. Uygun olan
# ilkini ariyoruz; kullanici PYTHON=... ile zorlayabilir.
step "Python araniyor"
find_python() {
  local candidate
  for candidate in "${PYTHON:-}" python3.12 python3.11 python3.10 python3 python3.9; do
    [[ -z "$candidate" ]] && continue
    command -v "$candidate" >/dev/null 2>&1 || continue
    "$candidate" -c 'import sys, venv; sys.exit(0 if sys.version_info >= (3, 9) else 1)' \
      >/dev/null 2>&1 && { echo "$candidate"; return 0; }
  done
  return 1
}
PY_BIN="$(find_python)" || die "Python 3.9+ (venv modulu ile) bulunamadi. PYTHON=/yol/python3 ile belirtebilirsiniz."
ok "$("$PY_BIN" -V 2>&1)  ($(command -v "$PY_BIN"))"

# --- 2. Sanal ortam --------------------------------------------------------- #
step "Sanal ortam"
if [[ ! -x "$VENV/bin/python" ]]; then
  info "olusturuluyor: $VENV"
  "$PY_BIN" -m venv "$VENV"
fi
PY="$VENV/bin/python"
ok "$("$PY" -V 2>&1)"

# Bagimliliklar: requirements.txt degismisse yeniden kur.
STAMP="$VENV/.requirements.sha"
CURRENT="$("$PY" - <<'EOF'
import hashlib, pathlib
print(hashlib.sha256(pathlib.Path("requirements.txt").read_bytes()).hexdigest())
EOF
)"
if [[ ! -f "$STAMP" || "$(cat "$STAMP")" != "$CURRENT" ]]; then
  step "Bagimliliklar kuruluyor (ilk seferde birkac dakika surer)"
  "$PY" -m pip install --quiet --upgrade pip
  "$PY" -m pip install --quiet -r requirements.txt
  echo "$CURRENT" > "$STAMP"
  ok "kuruldu"
else
  ok "guncel"
fi

export PYTHONPATH=src

# --- 3. Sifirdan baslama istegi --------------------------------------------- #
if [[ "$FRESH" -eq 1 ]]; then
  step "Uretilmis veri ve modeller siliniyor (--fresh)"
  rm -rf data/processed data/models
  ok "silindi"
fi
mkdir -p data/processed data/models data/raw

# --- 4. Sentetik veri ------------------------------------------------------- #
step "Sentetik fabrika verisi"
if compgen -G "data/processed/telemetry/*.parquet" >/dev/null; then
  ok "mevcut (yeniden uretmek icin: ./start.sh --fresh)"
else
  info "$DAYS gunluk gecmis uretiliyor..."
  "$PY" -m twin.simulator.run history --days "$DAYS" 2>&1 | tail -1
  ok "uretildi"
fi

# --- 5. Model egitimi ------------------------------------------------------- #
# Modeller egitildikleri ortama baglidir (pickle). Baska bir makinede ya da
# sklearn surumu degistikten sonra klonlanirsa sessizce yuklenemezler --
# dashboard 4 hedef yerine 1 tanesini gosterir ve kimse fark etmez.
# Burada ONCEDEN kontrol edip gerekiyorsa yeniden egitiyoruz.
step "Model egitimi"
MODELS_OK=0
if compgen -G "data/models/*/*/latest.txt" >/dev/null; then
  if "$PY" - <<'EOF' >/dev/null 2>&1
import sys
from twin.models.registry import load_all_detailed
bundles, failures = load_all_detailed()
sys.exit(1 if failures or not bundles else 0)
EOF
  then
    MODELS_OK=1
  else
    info "mevcut modeller bu ortamda yuklenemiyor (sklearn surumu degismis) -- yeniden egitiliyor"
    rm -rf data/models && mkdir -p data/models
  fi
fi

if [[ "$MODELS_OK" -eq 1 ]]; then
  ok "mevcut (yeniden egitmek icin: ./start.sh --fresh)"
else
  [[ -n "$FAST" ]] && info "HIZLI MOD -- metrikler tam egitimden kotudur, rapora konmaz" \
                   || info "4 hedef egitiliyor (~4 dk)..."
  "$PY" -m twin.models.train $FAST 2>&1 | grep -E "secilen model|HEDEF" || true
  ok "egitildi"
fi

# --- 6. Skorlama ------------------------------------------------------------ #
# Dashboard "gercek vs tahmin" grafigini bu kayitlardan cizer; olmadan bos acilir.
step "Gecmis veri skorlaniyor"
"$PY" -m twin.serving.scorer --backfill 480 2>&1 | grep -o "[0-9]* tahmin yazildi" | tail -1
ok "hazir"

# --- 7. Dashboard ----------------------------------------------------------- #
if [[ "$LAUNCH" -eq 0 ]]; then
  step "Hazir"
  info "Dashboard'u acmak icin:  PYTHONPATH=src $PY -m streamlit run src/twin/dashboard/app.py"
  exit 0
fi

step "Dashboard aciliyor"
info "Adres:  http://localhost:$PORT"
info "Durdurmak icin: Ctrl-C"
echo
exec "$PY" -m streamlit run src/twin/dashboard/app.py \
  --server.port "$PORT" --browser.gatherUsageStats false
