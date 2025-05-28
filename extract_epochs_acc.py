import json
import sys
import re

def parse_and_output_csv(input_str):
    # Clean and wrap the input to make it a valid JSON list
    input_str = input_str.strip()
    if not input_str.startswith('['):
        input_str = '[' + input_str
    if not input_str.endswith(']'):
        input_str = input_str.rstrip(',') + ']'

    # Remove trailing commas in JSON objects (invalid in JSON)
    input_str = re.sub(r',\s*}', '}', input_str)
    input_str = re.sub(r',\s*]', ']', input_str)

    try:
        data = json.loads(input_str)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return

    # Print CSV header
    print("epoch,top1_accuracy,top5_accuracy")

    # Print each row
    for entry in data:
        epoch = entry.get("epoch", "")
        val = entry.get("validation", {})
        top1 = val.get("top1", "")
        top5 = val.get("top5", "")
        print(f"{epoch},{top1},{top5}")

if __name__ == "__main__":
    input_str = sys.stdin.read()
    parse_and_output_csv(input_str)