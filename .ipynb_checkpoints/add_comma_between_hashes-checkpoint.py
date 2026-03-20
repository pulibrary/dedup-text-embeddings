import sys

# Usage: python add_comma_between_hashes.py input.json output.json


def fix_json(input_path, output_path):
    with open(input_path, "r") as infile:
        lines = infile.readlines()

    lines = [line.strip() for line in lines if line.strip()]

    for i in range(len(lines) - 1):
        lines[i] += ","

    lines.insert(0, "[")
    lines.append("]")

    with open(output_path, "w") as outfile:
        for line in lines:
            outfile.write(line + "\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python add_comma_between_hashes.py input.json output.json")
    else:
        fix_json(sys.argv[1], sys.argv[2])
