from src.interpretation import suggest_topic, topics_for_clusters


def test_suggest_topic_recognizes_support_domains():
    assert suggest_topic(["shared mailbox", "mailbox access"]).name == "Buzones compartidos"
    assert suggest_topic(["file share", "share access"]).name == "Accesos y permisos"
    assert suggest_topic(["basis user", "light user"]).name == "Gestión de cuentas de usuario"
    assert suggest_topic(["nexthink", "reservations"]).name == "Fin de vida de equipos"
    assert suggest_topic(["necesito acceso a una carpeta compartida"]).name == (
        "Accesos y permisos"
    )


def test_suggest_topic_exposes_evidence_and_safe_fallback():
    suggestion = suggest_topic(["email", "skype", "missing"])
    assert suggestion.name == "Correo y colaboración"
    assert set(suggestion.matched_keywords) == {"email", "skype"}
    assert suggestion.score > 0
    assert 0 <= suggestion.confidence <= 1

    fallback = suggest_topic(["unknown", "miscellaneous"])
    assert fallback.name == "Soporte general"
    assert fallback.matched_keywords == ()
    assert fallback.review_required


def test_topics_for_clusters_preserves_cluster_ids():
    suggestions = topics_for_clusters({0: ["file share"], 3: ["shared mailbox"]})
    assert set(suggestions) == {0, 3}
    assert suggestions[3].name == "Buzones compartidos"


def test_access_clusters_receive_distinct_subtypes():
    brief = suggest_topic(["file share", "share access", "remove file"])
    detailed = suggest_topic(["file share", "username", "following user", "read write"])
    portuguese = suggest_topic(["acesso pasta", "pasta", "rede", "acesso vpn"])

    assert brief.subtype == "Solicitudes breves de archivos compartidos"
    assert detailed.subtype == "Gestión detallada de permisos"
    assert portuguese.subtype == "Accesos diversos en portugués"
