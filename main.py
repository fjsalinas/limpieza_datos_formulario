import spacy
import re
from typing import List
from spacy.lang.es.stop_words import STOP_WORDS
from unicodedata import normalize

extra_stopwords = []
extra_stopwords += list("abcdefghijklmnopqrstuvwxyz")
extra_stopwords += STOP_WORDS


# https://realpython.com/natural-language-processing-spacy-python/
def is_token_allowed(token: spacy.tokens.Token) -> bool:
    """ filtrar stopwords y puntuacion """
    if (
        not token
        or not token.string.strip()
        or token.is_stop
        or token.is_punct
        or token.text in extra_stopwords
    ):
        return False
    return True


def stem(token: spacy.tokens.Token) -> str:
    """ lemmatizar """
    return token.lemma_.strip().lower()


def remove_accents(text: str) -> str:
    """
    eliminar diacriticos
    https://es.stackoverflow.com/questions/135707/c%C3%B3mo-puedo-reemplazar-las-letras-con-tildes-por-las-mismas-sin-tilde-pero-no-l
    """
    clean = re.sub(
        r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+",
        r"\1",
        normalize("NFD", text),
        0,
        re.I,
    )

    clean = normalize("NFC", clean)
    return clean


def process_text(doc: spacy.tokens.Doc) -> List[str]:
    """main. toma un doc de spaCy y devuelve la lista de tokens limpios """
    stemmed = [stem(token) for token in doc if is_token_allowed(token)]
    normalized = [remove_accents(string) for string in stemmed]
    return normalized


if __name__ == "__main__":
    nlp = spacy.load("es_core_news_md")
    full_text = """
    El Hogar de Cristo puso en manos del Gobierno una batería de propuestas que califican como “críticas” para la atención de la población altamente vulnerable en el marco de la crisis del coronavirus en Chile.

    Para estas instituciones de la sociedad civil, el escenario es complejo, debido al riesgo de que el temido Covid-19 afecte tanto a los participantes de los programas residenciales y ambulatorios, a sus funcionarios, y en definitiva cause estragos en el funcionamiento de estas entidades.

    El llamado a favor de adultos mayores y personas con discapacidad mental en situación de pobreza y exclusión a estas alturas de la emergencia sanitaria ya es un clamor, tomando en cuenta que el foco de Gobierno ha estado puesto en otros sectores y faltan acciones concretas para este segmento.
    Los dos principales riesgos

    En el Hogar de Cristo constituyeron un comité de emergencia que realizó un análisis de la situación actual, advirtiendo que dos son los riesgos principales que podrían “profundizar la crisis sanitaria” en estos espacios.

    El primero son las limitaciones para implementar las recomendaciones de prevención del contagio del Covid-19 con población de riesgo atendida en los programas y con los equipos de trabajo, debido a los problemas del desabastecimiento y la especulación de precios que esto ha generado en el mercado de los insumos de la salud (termómetro, mascarillas, alcohol gel, guantes, entre otros).

    El segundo gran riesgo es que tanto el Hogar de Cristo como otras instituciones que trabajan con personas de alta vulnerabilidad queden inhabilitados de brindar los servicios sociales y de salud comprometidos, en aquellos programas donde exista confirmación de uno o más casos Covid-19, con la correspondiente activación de los protocolos dispuestos por el Ministerio de Salu para la cuarentena de los participantes de los programas y sus equipos de trabajo. De hecho, sólo en el Hogar de Cristo, un 60 por ciento de sus trabajadores desempeñan funciones del área de la salud.
    Las medidas
    """

    clean_sample = process_text(nlp(full_text))

    print(clean_sample)
