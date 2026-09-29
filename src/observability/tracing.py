from phoenix.otel import register


tracer_provider = register(
    project_name="rag-financial",
    endpoint="http://localhost:6006/v1/traces",
)

tracer = tracer_provider.get_tracer(
    "rag-financial"
)