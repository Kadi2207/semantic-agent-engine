"""Detection d'intention par regles : rediger un email, ou repondre a une question."""
from __future__ import annotations

import re
import unicodedata
from typing import Literal

Intent = Literal["answer", "draft_email"]

# Verbes qui, a eux seuls, demandent d'ecrire a quelqu'un ("ecris a", "redige", "contacte").
_WRITE_VERBS = re.compile(r"\b(ecris|ecrire|ecrivez|redige|rediger|redigez|contacte|contacter|contactez)\b")
# Mention d'un mail : ne suffit que si la phrase est une demande, pas une question sur un mail
# ("Qui doit envoyer le mail aux utilisateurs ?" reste une question).
_MAIL_WORDS = re.compile(r"\b(e-?mails?|mails?|courriels?)\b")
_REQUEST_MARKERS = re.compile(r"\b(peux-tu|pourrais-tu|pouvez-vous|pourriez-vous|tu peux|merci de|envoie|envoyez|prepare|preparez)\b")
_QUESTION_START = re.compile(
    r"^(qui|quand|quel|quelle|quels|quelles|ou|comment|combien|pourquoi|est-ce|a-t-on|y a-t-il)\b"
)


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).replace("’", "'").strip()


# Formule de demande en tete de phrase ("Ecris un mail a la personne qui ...") : elle ne decrit pas
# le sujet et dilue la recherche, on la retire avant de chercher la tache.
_INSTRUCTION_PREFIX = re.compile(
    r"^\s*(?:(?:peux-tu|pourrais-tu|pouvez-vous|pourriez-vous|merci de)\s+)?"
    r"(?:[ée]cri(?:s|re|vez)|r[ée]dige[rz]?|contacte[rz]?|envoie[rz]?|envoyer|pr[ée]pare[rz]?)\s+"
    r"(?:(?:un|une)\s+)?(?:(?:e-?mail|mail|message|courriel|mot)\s+)?"
    r"(?:(?:à|a|au|aux|pour)\s+)?(?:(?:la|le|les)\s+)?"
    r"(?:(?:personnes?|responsables?|celui|celle|ceux)\s+)?"
    r"(?:(?:qui|charg[ée]e?s?\s+d[eu]|en\s+charge\s+d[eu]|d[eu]s?)\s+)?",
    re.IGNORECASE,
)


def strip_instruction(query: str) -> str:
    """'Ecris un mail a la personne qui doit signer le devis' -> 'doit signer le devis'."""
    topic = _INSTRUCTION_PREFIX.sub("", query, count=1).strip()
    return topic or query


def detect_intent(query: str) -> Intent:
    text = _normalize(query)
    if _WRITE_VERBS.search(text):
        return "draft_email"
    if _MAIL_WORDS.search(text):
        is_question = text.endswith("?") or bool(_QUESTION_START.match(text))
        if not is_question or _REQUEST_MARKERS.search(text):
            return "draft_email"
    return "answer"
