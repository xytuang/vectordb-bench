import os
import pytest
from pymilvus import MilvusClient, DataType


from opentelemetry import trace
from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

# Set up environment variables
os.environ['OTEL_EXPORTER_OTLP_ENDPOINT'] = 'http://localhost:4317'
os.environ['OTEL_SERVICE_NAME'] = 'milvus-client'

# Create a resource with the service name and application attributes
resource = Resource.create({
    "service.name": "milvus-client",
    "application": "milvus-otel"
})

# Set up the tracer provider with OTLP exporter
otlp_exporter = OTLPSpanExporter()
span_processor = BatchSpanProcessor(otlp_exporter)

trace.set_tracer_provider(
    TracerProvider(resource=resource)
)
trace.get_tracer_provider().add_span_processor(span_processor)

# Instrument gRPC
grpc_client_instrumentor = GrpcInstrumentorClient()
grpc_client_instrumentor.instrument()

# Get a tracer
tracer = trace.get_tracer(__name__)


def test_milvus_otel():
    with tracer.start_as_current_span("milvus_otel"):
        milvus_client = MilvusClient(
            uri="http://localhost:19530",
        )
        collection_name = "quick_setup"

        # Drop the collection if it exists
        if milvus_client.has_collection(collection_name):
            milvus_client.drop_collection(collection_name)

        # Create a collection
        milvus_client.create_collection(
            collection_name=collection_name,
            dimension=5
        )

        res = milvus_client.get_load_state(
            collection_name=collection_name
        )

        milvus_client.close()
