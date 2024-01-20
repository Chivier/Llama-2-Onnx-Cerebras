import onnxruntime as ort
import onnxruntime
import numpy as np
import onnx

"""
Type mapping:
Bfloat16	0
Bool	1
Complex128	2
Complex64	3
Double	4
Float	5
Float16	6
Int16	7
Int32	8
Int64	9
Int8	10
String	11
Uint16	12
Uint32	13
Uint64	14
Uint8	15
Undefined 16
"""

TypeMappingToNumpy = {
    0: np.float16,  # Bfloat16
    1: np.int8,  # Bool
    2: np.complex128,  # Complex128
    3: np.complex64,  # Complex64
    4: np.float64,  # Double
    5: np.float32,  # Float
    6: np.float16,  # Float16
    7: np.int64,  # strange
    8: np.int32,  # Int32
    9: np.int64,  # Int64
    10: np.float16,  # strange
    11: np.float16,  # strange
    12: np.uint16,  # Uint16
    13: np.uint32,  # Uint32
    14: np.uint64,  # Uint64
    15: np.uint8,  # Uint8
    16: np.float64,  # Undefined
}


def donnx_get_output_statistics(model):
    # Get the graph from the model
    graph = model.graph

    # Create a dictionary to store the output statistics
    output_statistics = {}

    # Iterate over the output nodes in the graph
    for output_node in graph.output:
        output_name = output_node.name

        # Get the shape of the output tensor
        output_shape = [dim.dim_value for dim in output_node.type.tensor_type.shape.dim]

        # Store the output name and shape in the dictionary
        output_statistics[output_name] = output_shape

        # Print the output name and shape
        # print(f"Output: {output_name}, Shape: {output_shape}")

    return output_statistics


def donnx_get_input_statistics(model):
    # Get the graph from the ONNX model
    graph = model.graph

    # Create a dictionary to store the input statistics
    input_statistics = {}

    # Iterate through the inputs in the graph
    for input_node in graph.input:
        # Get the name, shape, and tensor type of the input
        input_name = input_node.name
        input_shape = [dim.dim_value for dim in input_node.type.tensor_type.shape.dim]
        input_type = input_node.type.tensor_type.elem_type

        # Store the input name, shape, and type in the dictionary as a tuple
        input_statistics[input_name] = (input_shape, input_type)

    return input_statistics


def generate_random_inputs(input_statistics, generate_random=False):
    # Create a dictionary to store the random inputs
    random_inputs = {}

    # Iterate through the input statistics
    for input_name, input_info in input_statistics.items():
        input_shape, input_type = input_info
        # Generate random data based on the input shape
        if generate_random:
            random_data = np.random.randn(*input_shape).astype(
                TypeMappingToNumpy[input_type]
            )
        else:
            # Fill 0 for all input
            random_data = np.zeros(input_shape).astype(TypeMappingToNumpy[input_type])

        # Store the random data in the dictionary
        random_inputs[input_name] = random_data

    return random_inputs


class DummyEngine:
    def __init__(self, onnx_file: str):
        self.model = onnx.load(onnx_file)
        self.model_input_stat = donnx_get_input_statistics(self.model)
        self.model_output_stat = donnx_get_output_statistics(self.model)
        self.input = {}
        self.output = {}
        self.reverse_node_index = {}
        self.value_cache = {}

    def set_input(self, inputs):
        self.input = inputs

    def donnx_get_topological_order(self):
        graph = self.model.graph
        
        # Create a dictionary to store the topological order of each node
        node_order = {}
        
        # Perform a topological sort of the nodes
        for i, node in enumerate(graph.node):
            for input_name in node.input:
                if input_name not in node_order:
                    node_order[input_name] = 0
            node_order[node.output[0]] = i
        
        # Sort the nodes based on their topological order
        sorted_nodes = sorted(graph.node, key=lambda node: node_order[node.output[0]])

        for i, node in enumerate(sorted_nodes):
            for output_name in node.output:
                self.reverse_node_index[output_name] = i
        
        # Print the nodes with their topological order
        for node in sorted_nodes:
            print(f"Node {node.name} (Topological Order: {node_order[node.output[0]]}):")
            # Print dependencies
            for input_name in node.input:
                if input_name in self.reverse_node_index:
                    print(self.reverse_node_index[input_name], end=' ')
                else:
                    print("from_input", end=' ')
            print()



# Load the ONNX model
# Load the ONNX model
model_path = "./7B_FT_float16/ONNX/LlamaV2_7B_FT_float16.onnx"
dummy_engine = DummyEngine(model_path)

# Generate random inputs
input_statistics = donnx_get_input_statistics(dummy_engine.model)

random_inputs = generate_random_inputs(input_statistics, False)
dummy_engine.set_input(random_inputs)

dummy_engine.donnx_get_topological_order()

# Create an ONNX runtime session
# session = ort.InferenceSession(model_path)

# Run the ONNX model with the random inputs
# output = session.run(["output_hidden_states", "output_past_key", "output_past_value"], random_inputs)

# Process the output as needed
# ...
