import os
import sys

# Ensure src can be imported if not in PYTHONPATH (though it likely is)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.utils import read_binary_qubit_op

def main():
    base_dir = "../molecules/thomas"
    
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return

    # Get all subdirectories
    subdirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    
    print(f"{'Molecule':<15} | {'Qubits':<8} | {'Pauli Terms':<12}")
    print("-" * 41)

    for molecule in subdirs:
        ham_path = os.path.join(base_dir, molecule, "qubit_op.data")
        if os.path.exists(ham_path):
            try:
                n_qubits, qubit_op = read_binary_qubit_op(ham_path)
                print(f"{molecule:<15} | {n_qubits:<8} | {len(qubit_op.terms):<12}")
            except Exception as e:
                print(f"{molecule:<15} | {'Error':<8} | {str(e)}")

if __name__ == "__main__":
    main()