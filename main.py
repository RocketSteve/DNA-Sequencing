import os

def load_sequences_from_file(file_path):
    """
    Load DNA sequences from a given file.

    Args:
    file_path (str): The path to the file containing the sequences.

    Returns:
    list: A list of DNA sequences read from the file.
    """
    sequences = []
    try:
        with open(file_path, 'r') as file:
            sequences = [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred while reading the file {file_path}: {e}")
    
    return sequences

def parse_filename(file_name):
    """
    Parse the filename to extract the sequence length, total length, and error information.

    Args:
    file_name (str): The filename that encodes the sequence and error info.

    Returns:
    dict: A dictionary containing the parsed parameters.
    """
    base_name = os.path.basename(file_name)
    try:
        parts = base_name.split('.')
        if len(parts) > 1:
            length_info, n_errors = parts
            n, errors = n_errors.split('-') if '-' in n_errors else n_errors.split('+')
            
            return {
                'length': int(length_info),
                'n': int(n),
                'error_type': 'negative' if '-' in n_errors else 'positive',
                'error_count': int(errors)
            }
    except ValueError as e:
        print(f"Error parsing filename '{file_name}': {e}")

    return {}

def create_graph(sequences, k):
    """
    Create a directed graph from a list of DNA sequences, allowing for flexible overlaps.

    This function constructs a graph where each node is a DNA sequence, and directed edges
    are created between nodes if the suffix of one sequence overlaps with the prefix of
    another sequence with a given overlap length minus potential mismatches or indels.

    Args:
        sequences (list): A list of DNA sequences.
        k (int): The number of characters each sequence overlaps with the next one, allowing
                 for a potential mismatch or indel within the overlap.

    Returns:
        tuple: Returns a tuple containing three items:
            - edges (defaultdict of list): A dictionary where keys are sequences and values are lists of sequences
              that can be reached directly from the key sequence based on overlap criteria.
            - in_degree (defaultdict of int): A dictionary counting the number of incoming edges for each sequence.
            - out_degree (defaultdict of int): A dictionary counting the number of outgoing edges for each sequence.
    
    The graph is built by checking for overlaps between the suffix of each sequence and the prefix of any
    other sequences. The overlaps are defined as the first (k-2) characters of the suffix matching the second to
    (k-1)th characters of any possible prefix, thus allowing for one character of flexibility in the matching process.
    """
    from collections import defaultdict

    edges = defaultdict(list)
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)

    prefix_map = defaultdict(list)
    for seq in sequences:
        prefix = seq[:-1] 
        suffix = seq[1:]
        prefix_map[prefix].append(seq)  

    for seq in sequences:
        prefix = seq[:-1]
        suffix = seq[1:]
        for possible_match in prefix_map:
            if suffix[:k-2] == possible_match[1:k-1]:  # allow one mismatch or indel in overlap
                for target_seq in prefix_map[possible_match]:
                    if target_seq != seq:  # prevent self-loop
                        edges[seq].append(target_seq)
                        out_degree[seq] += 1
                        in_degree[target_seq] += 1
                        print(f"Connecting {seq} to {target_seq}") 

    return edges, in_degree, out_degree

def create_graph_with_positive_error_handling(sequences, k):
    from collections import defaultdict
    import numpy as np  # Import numpy for average calculation

    edges = defaultdict(list)
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)

    prefix_map = defaultdict(list)
    for seq in sequences:
        prefix = seq[:-1]
        suffix = seq[1:]
        prefix_map[prefix].append(seq)

    # Initially connect sequences based on flexible overlaps
    for seq in sequences:
        prefix = seq[:-1]
        suffix = seq[1:]
        for possible_match in prefix_map:
            if suffix[:k-2] == possible_match[1:k-1]:  # Allowing one mismatch or indel
                for target_seq in prefix_map[possible_match]:
                    if target_seq != seq:
                        edges[seq].append(target_seq)
                        out_degree[seq] += 1
                        in_degree[target_seq] += 1

    # Filter out sequences with connectivity lower than a threshold
    average_in_degree = np.mean(list(in_degree.values()))
    average_out_degree = np.mean(list(out_degree.values()))

    # Define thresholds as half of average (tweak based on dataset specifics)
    in_degree_threshold = average_in_degree / 2
    out_degree_threshold = average_out_degree / 2

    # Create new filtered graph structures
    filtered_edges = defaultdict(list)
    filtered_in_degree = defaultdict(int)
    filtered_out_degree = defaultdict(int)

    # Only include nodes that meet the degree threshold
    for seq in sequences:
        if in_degree[seq] >= in_degree_threshold and out_degree[seq] >= out_degree_threshold:
            for target_seq in edges[seq]:
                if in_degree[target_seq] >= in_degree_threshold and out_degree[target_seq] >= out_degree_threshold:
                    filtered_edges[seq].append(target_seq)
                    filtered_out_degree[seq] += 1
                    filtered_in_degree[target_seq] += 1

    return filtered_edges, filtered_in_degree, filtered_out_degree


def print_graph_details(graph, in_degree, out_degree):
    multi_in = {k: v for k, v in in_degree.items() if v > 1}
    multi_out = {k: v for k, v in out_degree.items() if v > 1}
    print("Nodes with multiple incoming edges:", multi_in)
    print("Nodes with multiple outgoing edges:", multi_out)


def print_dict(dict):
    for i, (key, data) in enumerate(dict.items()):
        print(f'Key: {key} - {data}')
        
        
# Changed used algo to dfs with pruning as the proposed algo was impossible to use (not a eulerian graph)
def dfs(graph, start, visited=None, path=None, longest=None):
    if visited is None:
        visited = set()
    if path is None:
        path = []
    if longest is None:
        longest = {'length': 0, 'path': []}

    visited.add(start)
    path.append(start)

    if len(path) > longest['length']:
        longest['length'] = len(path)
        longest['path'] = path.copy()

    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            # Prune: only continue if the potential path could be longer than the longest found
            if len(path) + 1 > longest['length']: 
                dfs(graph, neighbor, visited, path, longest)

    path.pop()
    visited.remove(start)


def find_longest_path(graph):
    longest = {'length': 0, 'path': []}
    for start_node in graph:
        visited = set()
        dfs(graph, start_node, visited, [], longest)
    return longest['path']

def concatenate_dna_path(path, k):
    """
    Concatenate a path of DNA sequences into a single DNA sequence based on overlap.

    Args:
    path (list): A list of DNA sequences in the order they are connected.
    k (int): The number of characters each sequence overlaps with the next one.

    Returns:
    str: A concatenated DNA sequence.
    """
    if not path:
        return "" 

    full_sequence = path[0]

    for i in range(1, len(path)):
        if k > 1:
            full_sequence += path[i][-1]
        else:
            full_sequence += path[i]

    return full_sequence


def main():
    file_path = "random_positive/9.200+80"
    sequences = load_sequences_from_file(file_path)
    file_info = parse_filename(file_path)

    #print("Loaded sequences:", sequences)
    print("File info:", file_info)
    k = file_info['length'] - 6 # Changing this parameter allows for the biggest chain length gains
    graph, in_degree, out_degree = create_graph_with_positive_error_handling(sequences, k)
    print_graph_details(graph,in_degree,out_degree)
    #print_dict(graph)
    print('<><><><>')
    print_dict(in_degree)
    print('<><><><>')
    print_dict(out_degree)
    
    longest = find_longest_path(graph)
    print(len(find_longest_path(graph)))
    
    print(concatenate_dna_path(longest, k))
    
if __name__ == "__main__":
    retcode = main()
    exit()
    

    