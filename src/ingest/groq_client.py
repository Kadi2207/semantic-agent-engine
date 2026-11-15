"""Client Groq minimal avec cache disque, sur le modele de scripts/build_triplets_llm.py.

- cle lue dans l'environnement ou dans .env (jamais affichee) ;
- appels espaces pour respecter la limite de tokens par minute ;
- chaque reponse brute est ecrite dans un cache JSONL des sa reception : un nouveau run
  ne rappelle Groq que pour les requetes absentes du cache (resultats figes et reproductibles).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = ROOT / ".env"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
REQUEST_TIMEOUT = 120
DEFAULT_RETRY_WAIT = 30.0
MIN_RATE_LIMIT_WAIT = 30.0


class MissingKeyError(RuntimeError):
    pass


class GroqError(RuntimeError):
    pass


def load_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip().removeprefix("export ").strip() == "GROQ_API_KEY":
                key = value.strip().strip("\"'")
                if key:
                    return key
    raise MissingKeyError(
        "GROQ_API_KEY introuvable : ni dans l'environnement, ni (non vide) dans .env. "
        "Colle ta cle dans .env (GROQ_API_KEY=...) puis relance."
    )


class CachedGroq:
    def __init__(
        self,
        model: str,
        temperature: float,
        cache_path: Path,
        min_seconds_between_calls: float,
        json_mode: bool = False,
        cache_only: bool = False,
        max_completion_tokens: int | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.cache_path = cache_path
        self.min_seconds_between_calls = min_seconds_between_calls
        self.json_mode = json_mode
        self.max_completion_tokens = max_completion_tokens
        self.cache_only = cache_only
        self.calls = 0  # appels Groq reellement effectues pendant ce run
        self.cache_hits = 0
        self.prompt_tokens = 0  # tokens des appels de ce run
        self.completion_tokens = 0
        # Cout de production des reponses lues dans le cache (tokens enregistres lors de l'appel d'origine).
        self.cached_prompt_tokens = 0
        self.cached_completion_tokens = 0
        self._last_call_at = 0.0
        self._api_key: str | None = None
        self._cache: dict[str, str] = {}
        self._usage: dict[str, dict[str, int]] = {}
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self._cache[entry["key"]] = entry["raw"]
                    self._usage[entry["key"]] = entry.get("usage") or {}

    def snapshot(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "cached_completion_tokens": self.cached_completion_tokens,
        }

    def key(self, system_prompt: str, user_prompt: str) -> str:
        """Depend du modele, de la temperature, de la consigne et de l'entree : tout changement invalide."""
        material = json.dumps(
            [self.model, self.temperature, self.json_mode, self.max_completion_tokens, system_prompt, user_prompt],
            ensure_ascii=False,
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def complete(self, system_prompt: str, user_prompt: str, label: str = "", refresh: bool = False) -> str:
        """Reponse en cache si elle existe ; refresh=True force un nouvel appel (qui remplace l'entree)."""
        key = self.key(system_prompt, user_prompt)
        if key in self._cache and not refresh:
            self.cache_hits += 1
            usage = self._usage.get(key, {})
            self.cached_prompt_tokens += int(usage.get("prompt_tokens", 0))
            self.cached_completion_tokens += int(usage.get("completion_tokens", 0))
            return self._cache[key]
        if self.cache_only:
            raise GroqError("absente du cache (mode cache seul)")
        raw, usage, finish_reason = self._call(system_prompt, user_prompt)
        self.calls += 1
        self.prompt_tokens += int(usage.get("prompt_tokens", 0))
        self.completion_tokens += int(usage.get("completion_tokens", 0))
        self._cache[key] = raw
        self._usage[key] = usage
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "key": key,
            "label": label,
            "model": self.model,
            "usage": usage,
            "finish_reason": finish_reason,
            "raw": raw,
        }
        with self.cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return raw

    # ------------------------------------------------------------------ reseau

    def _pace(self) -> None:
        wait = self._last_call_at + self.min_seconds_between_calls - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_call_at = time.monotonic()

    def _call(self, system_prompt: str, user_prompt: str) -> tuple[str, dict[str, int], str]:
        """(contenu, usage {prompt_tokens, completion_tokens}, finish_reason)."""
        if self._api_key is None:
            self._api_key = load_api_key()
        body: dict[str, object] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.max_completion_tokens:
            body["max_completion_tokens"] = self.max_completion_tokens
        payload = json.dumps(body).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": "semantic-ingest/1.0",
        }
        for attempt in (1, 2):
            request = urllib.request.Request(GROQ_URL, data=payload, headers=headers, method="POST")
            self._pace()
            try:
                with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                    data = json.loads(response.read().decode("utf-8"))
                choice = data["choices"][0]
                usage = {k: int(v) for k, v in (data.get("usage") or {}).items() if k in ("prompt_tokens", "completion_tokens")}
                return str(choice["message"]["content"]), usage, str(choice.get("finish_reason", ""))
            except urllib.error.HTTPError as exc:
                detail = _error_message(exc)
                if exc.code in (401, 403):
                    raise MissingKeyError(f"Groq refuse la cle (HTTP {exc.code}, {detail}) : verifie GROQ_API_KEY.") from None
                if exc.code == 429 and attempt == 1:
                    wait = max(_retry_after(exc.headers.get("retry-after")), MIN_RATE_LIMIT_WAIT)
                    print(f"    [429] limite de debit, attente {wait:.0f}s puis nouvel essai")
                    time.sleep(wait)
                    continue
                raise GroqError(f"HTTP {exc.code} : {detail}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt == 1:
                    print(f"    [reseau] {type(exc).__name__}, nouvel essai dans 5s")
                    time.sleep(5)
                    continue
                raise GroqError(f"reseau : {type(exc).__name__}") from None
            except (KeyError, IndexError, ValueError):
                raise GroqError("reponse Groq illisible") from None
        raise GroqError("echec apres nouvel essai")


def _error_message(exc: urllib.error.HTTPError) -> str:
    """Message d'erreur renvoye par Groq (le corps ne contient jamais la cle)."""
    try:
        body = exc.read().decode("utf-8", "replace")
        return str(json.loads(body)["error"]["message"])[:300]
    except Exception:
        return "corps d'erreur illisible"


def _retry_after(value: str | None) -> float:
    try:
        return min(max(float(value), 1.0), 120.0) if value else DEFAULT_RETRY_WAIT
    except ValueError:
        return DEFAULT_RETRY_WAIT
