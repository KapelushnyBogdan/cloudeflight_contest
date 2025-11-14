import csv
import os

def word_to_number(value):
    """Convert word or numeric string to integer."""
    # Try to convert directly if it's already a number
    try:
        return int(value)
    except ValueError:
        pass

    # Word to number mapping
    word_map = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
        'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
        'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30,
        'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
        'eighty': 80, 'ninety': 90
    }

    value_lower = value.lower().strip()

    # Handle simple words
    if value_lower in word_map:
        return word_map[value_lower]

    # Handle compound numbers like "twenty-one" or "twenty one"
    parts = value_lower.replace('-', ' ').split()
    if len(parts) == 2:
        return word_map.get(parts[0], 0) + word_map.get(parts[1], 0)

    return int(value)  # Fallback

def solve_bop_sorting(input_file, output_file):
    """Sort Bird Observation Points by popularity."""
    bops = []

    # Read and parse the CSV file
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        for row in reader:
            if row:  # Skip empty rows
                bop_id = int(row[0])
                temperature = word_to_number(row[1])
                humidity = word_to_number(row[2])
                bops.append((bop_id, temperature, humidity))

    # Sort by:
    # 1. Temperature descending (higher is better)
    # 2. Humidity ascending (lower is better)
    # 3. BOP ID ascending (lower ID wins ties)
    sorted_bops = sorted(bops, key=lambda x: (-x[1], x[2], x[0]))

    # Extract just the BOP IDs
    result = ' '.join(str(bop[0]) for bop in sorted_bops)

    # Write to output file
    with open(output_file, 'w') as f:
        f.write(result + '\n')

    return result

# Process all input files
level_1_dir = 'level_1'
input_files = [f for f in os.listdir(level_1_dir) if f.endswith('.in')]

for input_file in sorted(input_files):
    input_path = os.path.join(level_1_dir, input_file)
    output_file = input_file.replace('.in', '.out')
    output_path = os.path.join(level_1_dir, output_file)

    result = solve_bop_sorting(input_path, output_path)
    print(f"{input_file}: {result}")
