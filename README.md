# GRACE2026

This repository contains technical resources and information related to the [GRACE2026](https://www.codabench.org/competitions/13280/) shared task.


<p align="center">
	  <img src="https://img.shields.io/badge/📄_Paper-TBA-red.svg" alt="Paper Coming Soon">
</p>

---

## Technical Resources

This repository open-sources all the technical components, and code utilized in our methodology for the shared task:

*   **Formatting Engine & Ensembling Core:** The combined Python engine designed to preprocess raw data into the standardized GRACE schema and execute our decision-fusion algorithms to optimize final predictions.
*   **Experimental prompts:** The exact prompt engineering artifacts, system instructions, and templates utilized during experimentation, mapped specifically to each individual subtask.


---

## Dataset

The official GRACE competition records were provided through Codabench. They are
not redistributed in this repository. The repository therefore includes only public auxiliary data
and small example inputs for running the toolkit.

Our auxiliary data comes from
[CasiMedicos-Arg](https://huggingface.co/datasets/HiTZ/casimedicos-arg), a
CC-BY-4.0 multilingual medical question-answering argumentation dataset in
English, Spanish, French, and Italian, described by
[Sviridova et al. (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.1026/).
The processed release in
[casimedicos-dataset/](casimedicos-dataset/) contains normalized train/dev/test
JSONL files, aligned argument relations, and monolingual, bilingual, and
all-language split variants.

The [source unifier](toolkit/src/source-unifier/) converts these processed
CasiMedicos files into the GRACE JSON schema and can merge them with a local
GRACE JSON file. To reproduce unified GRACE + CasiMedicos inputs, users must
provide their own copy of the GRACE records.



## Toolkit

Utilities used to prepare data and submissions for the GRACE shared task.
This toolkit is intentionally small and focused on:

- splitting GRACE-format datasets;
- converting casimedicos-arg argumentation records into GRACE format;
- creating task-specific ensembles and assembling final submission-ready runs.

## Prompts

### Subtask 1

#### Sentence-by-Sentence Annotation Prompt

````text
Eres un experto médico. Tu tarea es la Detección de Oraciones de Evidencia.
Analiza la siguiente oración dentro del caso clínico y determina si es "relevant" o "not-relevant" para apoyar o refutar las opciones de respuesta.

Caso Clínico:
{context}

Opciones:
{choices_str}

Oración a evaluar:
{sentence}

Responde únicamente con "relevant" o "not-relevant".

````

#### All-in-One Inference Annotation Prompt
````text
Eres un experto clínico. Tu tarea es evaluar una lista numerada de oraciones de un caso clínico.
Debes determinar si cada oración contiene evidencia médica RELEVANTE (síntomas, historial, pruebas) o si es IRRELEVANTE (texto de relleno, saludos, la pregunta final).

Restricciones obligatorias:
- "sentence_relevancy" debe tener exactamente una etiqueta por cada oración recibida, y usa solo "relevant" o "not-relevant".
- Devuelve únicamente JSON válido.

Formato obligatorio de salida:
{
    "sentence_relevancy": [
        "relevant",
        "not-relevant"
    ]
}

````

### Subtask 2

#### Sentence-by-Sentence Annotation Prompt

````text
Eres un experto en razonamiento clínico y extracción de información. Tu tarea es identificar y extraer fragmentos de texto exactos que representen 'Premises' o 'Claims' dentro de una oración específica, utilizando el caso clínico completo solo como contexto de fondo.

Definiciones:
- Premise: Evidencia clínica objetiva (hechos, mediciones, síntomas, observaciones o antecedentes médicos del paciente).
- Claim: Opciones de respuesta o hipótesis (diagnósticos candidatos, propuestas de tratamiento o pronósticos).

Reglas de extracción:
    1. Evalúa ÚNICAMENTE la oración a analizar.
    2. El fragmento extraído debe ser una copia EXACTA (respetando mayúsculas, puntuación y espacios) de cómo aparece en la oración. Una 'Claim' puede abarcar la oración completa.
    3. Si la oración no contiene ninguna 'Premise' ni 'Claim' (ej. texto de relleno o preguntas genéricas), debes devolver un array vacío: []
    4. Responde ÚNICAMENTE con un array JSON válido, sin introducciones ni explicaciones previas.
    5. Formato de salida: [{"text": fragmento de texto, "type": Premise/Claim}]
    
Contexto:
{raw_text}

Oración a analizar:
{segment_text}

````

#### All-in-One Inference Annotation Prompt
````text
Eres un experto en razonamiento clínico y extracción de información. Tu tarea es identificar y extraer fragmentos de texto exactos que representen 'Premises' o 'Claims' dentro del caso clínico proporcionado.
            
Definiciones:
- Premise: Evidencia clínica objetiva (hechos, mediciones, síntomas, observaciones).
- Claim: Opciones de respuesta o hipótesis.

Reglas de extracción:
1. El fragmento extraído debe ser una copia EXACTA del texto original.
2. Asigna a cada Premise un 'local_id' correlativo (p1, p2...) y el 'source_index' de la oración donde aparece.
3. Extrae las Claims con su respectivo ID.

Formato obligatorio de salida:
{
    "premises": [
        {
            "local_id": "p1",
            "source_index": 0,
            "text": "fragmento exacto mínimo"
        }
    ],
    "claims": [
        {
            "id": "1",
            "text": "texto exacto de la opción"
        }
    ]
}

````

### Subtask 3

#### Sentence-by-Sentence Annotation Prompt

````text
Eres un experto clínico. Tu tarea es evaluar la relación argumentativa entre una evidencia (Premise) y una opción candidata (Claim) basándote en el caso clínico proporcionado.

Las posibles relaciones entre 'premise' y 'claim' son:
- Support: Si la premise apoya, confirma o es consistente con la claim.
- Attack: Si la premise contradice, refuta, descarta o hace improbable la claim.
- Nothing: Si la premise y la claim no tienen relación.
Responde únicamente con 'Support', 'Attack' o 'Nothing'.

Caso Clínico:
{raw_text}

Evidencia (Premise):
{premise_text}

Opción (Claim):
{claim_text}

````

#### All-in-One Inference Annotation Prompt
````text
Eres un razonador clínico. Se te dará una PREMISA (un hecho del paciente) y un CLAIM (una posible respuesta/diagnóstico). Tu tarea es determinar la relación argumentativa entre ellos:
- 'Support': La premisa apoya, confirma o es consistente con el claim.
- 'Attack': La premisa contradice, descarta o hace improbable el claim.

Restricciones obligatorias:
- Cada "premise_id" debe ser el ID proporcionado para la Premise analizada.
- Cada "claim_id" debe corresponder a la opción recibida.
- Usa solo "Support" o "Attack".
- Devuelve únicamente JSON válido.

Formato obligatorio de salida:
{
    "relations": [
        {
            "premise_id": "p1",
            "claim_id": "3",
            "relation_type": "Support"
        }
    ]
}

````

### Global (all subtask in one inference)

#### All-in-One Inference Annotation Prompt

````text
Eres un experto médico en razonamiento clínico MIR y extracción de argumentos clínicos.
Tu tarea es resolver tres subtareas de razonamiento clínico en una única inferencia.

Recibirás:
1. Un caso clínico completo.
2. Una lista de oraciones del contexto clínico, cada una con un índice.
3. Una lista de opciones de respuesta, cada una con un id. A las cuales denominaremos Claims.

IMPORTANTE:
Tu tarea consiste en:
1. Clasificar cada oración del contexto como "relevant" o "not-relevant".
2. Extraer únicamente Premises mínimas desde las oraciones relevantes.
3. Relacionar las Premises extraídas con las Claims/opciones usando el id de la opción.

Subtarea 1 — Evidence Sentence Detection:
Clasifica cada oración del contexto clínico como:
- "relevant": contiene evidencia clínica útil para apoyar o refutar alguna opción.
- "not-relevant": no aporta evidencia clínica útil para decidir entre las opciones.

Reglas para relevancia:
- Las oraciones con síntomas, signos, antecedentes, resultados de pruebas, evolución temporal, factores de riesgo, negaciones clínicas o hallazgos clínicos suelen ser "relevant".
- Las preguntas genéricas como "¿Cuál es el diagnóstico más probable?", "¿Cuál es el tratamiento indicado?" o frases que solo introducen la pregunta suelen ser "not-relevant".
- Una oración no debe ser "relevant" solo por introducir la pregunta.

Subtarea 2 — Minimal Premise Span Detection:
Extrae fragmentos exactos de texto que sean Premises.

Una Premise es una unidad mínima de evidencia clínica objetiva del caso:
- síntoma
- signo
- antecedente
- edad o sexo si son clínicamente relevantes
- duración o evolución temporal
- resultado de prueba
- hallazgo de exploración
- factor de riesgo
- ausencia o negación clínica relevante
- dato que apoye o refute una opción

Reglas estrictas para Premises:
1. Extrae Premises solo desde oraciones clasificadas como "relevant".
2. El texto debe ser una copia EXACTA de un fragmento de la oración original.
3. No reformules. No inventes.
5. No incluyas offsets.
6. No extraigas texto de las opciones como Premise.
7. No extraigas la pregunta final como Premise.
8. No devuelvas una oración completa si contiene varias unidades clínicas.
9. Si una oración contiene varias evidencias clínicas, divide la oración en varias Premises mínimas.
10. Cada Premise debe expresar una sola unidad clínica.
11. Prefiere el span más corto que conserve el significado clínico.
13. Mantén modificadores clínicamente importantes, por ejemplo duración, localización, severidad o resultado de prueba.
14. No incluyas introducciones, conectores ni relleno, solo la evidencia clínica concreta.

Subtarea 3 — Argumentative Relation Detection:
Relaciona cada Premise con una o varias Claims cuando exista una relación clara.

Las Claims son las opciones de respuesta recibidas.
Debes referirte a ellas usando su id.

Tipos de relación:
- "Support": la Premise apoya, favorece, confirma o es consistente con esa opción.
- "Attack": la Premise contradice, descarta, debilita o hace improbable esa opción.

No incluyas relaciones dudosas.

Formato obligatorio de salida:

{
	"sentence_relevancy": [
		"relevant",
		"not-relevant"
	],
	"premises": [
		{
			"local_id": "p1",
			"source_index": 0,
			"text": "fragmento exacto mínimo"
		}
	],
	"relations": [
		{
			"premise_id": "p1",
			"claim_id": "3",
			"relation_type": "Support"
		}
	]
}

Restricciones obligatorias:
- "sentence_relevancy" debe tener exactamente una etiqueta por cada oración recibida.
- Usa solo "relevant" o "not-relevant".
- Cada "source_index" debe ser el índice de la oración de la que sale la Premise.
- Cada "text" debe aparecer literalmente dentro de la oración indicada.
- Cada "text" debe ser el menor fragmento clínicamente suficiente, no la oración completa.
- Cada "premise_id" debe existir en "premises".
- Cada "claim_id" debe corresponder a una opción recibida.
- Usa solo "Support" o "Attack".
- Devuelve únicamente JSON válido.
- Limita tu bloque de pensamiento a un máximo de 3 párrafos cortos antes de dar la respuesta final.

````

## Citation
```
TBA
```
