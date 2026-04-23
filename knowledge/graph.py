from pathlib import Path

from rdflib import Graph, Namespace, RDF, RDFS

EX = Namespace("http://example.org/defect#")

_DEFAULT_TTL = Path(__file__).parent / "ontology.ttl"


def load_defect_types(ttl_path: str | Path = _DEFAULT_TTL, class_name: str = "") -> dict[str, dict]:
    """
    Parse the RDF ontology and return all ex:DefectType nodes.

    Returns a dict keyed by IRI string:
        {
            "http://example.org/defect#Crack": {
                "iri":     "http://example.org/defect#Crack",
                "label":   "crack",
                "prompts": ["a photo of a {} with a crack", ...],
            },
            ...
        }

    Prompt templates contain '{}' as a placeholder for the object class name.
    """
    g = Graph()
    g.parse(str(ttl_path), format="turtle")

    defect_types: dict[str, dict] = {}

    for subject in g.subjects(RDF.type, EX.DefectType):
        iri = str(subject)
        label_node = g.value(subject, RDFS.label)
        label = str(label_node) if label_node is not None else iri.split("#")[-1]
        prompts = [str(o) for o in g.objects(subject, EX.promptTemplate)]

        if len(class_name) > 0 and str(g.value(subject, EX.applicableToClass)) != class_name:
            continue

        if not prompts:
            raise ValueError(
                f"DefectType <{iri}> has no ex:promptTemplate triples in {ttl_path}"
            )

        defect_types[iri] = {"iri": iri, "label": label, "prompts": prompts}

    if not defect_types:
        raise ValueError(f"No ex:DefectType instances found in {ttl_path}")

    return defect_types
