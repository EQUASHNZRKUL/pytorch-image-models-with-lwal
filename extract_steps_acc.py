import sys
import re

def extract_train_acc(log_text):
    # Regex to extract step and accuracy
    pattern = r"train_acc @ (\d+) steps tensor\(([\d.]+),"
    matches = re.findall(pattern, log_text)

    # Prepare tab-separated output for Google Sheets
    header = "Step,Train Accuracy"
    rows = [f"{step},{acc}" for step, acc in matches]

    return "\n".join([header] + rows)

if __name__ == "__main__":
    # Read from stdin
    log_input = sys.stdin.read()
    output_table = extract_train_acc(log_input)
    print(output_table)