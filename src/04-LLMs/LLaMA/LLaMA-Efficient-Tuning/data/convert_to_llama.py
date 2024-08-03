import sys

def convert_to_json_list(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()

    # Add a comma at the end of each line, except for the last line
    lines = [line.strip() + ',' for line in lines[:-1]] + [lines[-1].strip()]

    # Add square brackets at the beginning and end of the list
    lines.insert(0, '[')
    lines.append(']')

    # Write the modified content back to the original file
    with open(file_name, 'w') as f:
        f.write('\n'.join(lines))

# Get the file name from the command line arguments
file_name = sys.argv[1]

# Call the function with the provided file name
convert_to_json_list(file_name)
